"""Cylindrical-tree MPC for height, azimuth, passive tilt and six arm joints.

Wheel commands drive height and tree-axis rotation. Roll and pitch are passive."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
import hashlib
import heapq
import json
from time import perf_counter

import casadi as cs
import mujoco
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from envs.common import (ARM_JOINTS, BASE_JOINTS, WHEEL_ANGLES, CONTACT_STIFFNESS, rolling_directions,
                                   wheel_commands, wheel_force_map, wheel_mixer, wrap_angle)


@dataclass
class TreeWorkMPCConfig:
    horizon: int = 8
    prediction_dt: float = .12
    solve_interval: int = 3
    wheel_max: float = 30.
    nominal_friction: float = .60
    normal_force: float = 45.
    traction_gain: float = 35.
    contact_stiffness: float = CONTACT_STIFFNESS
    wheel_radius: float = .032
    tree_radius: float = .15
    arm_speed_max: float = .8
    max_base_speed: float = .35
    max_azimuth_speed: float = .65
    max_work_base_speed: float = .06
    max_work_azimuth_speed: float = .15
    # Native KP220/KV2.5 ramp probes fit .0172-.0232 s on the three main
    # pitch axes. Use a local first-order approximation for prediction.
    arm_velocity_time_constant: float = .022
    max_tilt: float = .20
    min_height: float = .30
    max_height: float = 1.9
    height_tolerance: float = .035
    azimuth_tolerance: float = .06
    ee_position_weight: float = 1800.
    ee_orientation_weight: float = 35.
    height_residual_scale: float = .06
    azimuth_residual_scale: float = .15
    ee_residual_scale: float = .025
    model_residual_scales: tuple[float, ...] = (1.5, 3., 8., 8.)
    residual_use_adaptation: bool = True
    residual_model_scale: float = .25
    adaptation_rate: float = .06
    ik_iterations: int = 40
    planner_height_step: float = .025
    planner_theta_bins: int = 72
    footprint_height: float = .015
    footprint_azimuth: float = .075
    terrain_footprint_margin: float = .035
    terrain_theta_margin: float = .08
    planner_lookahead: float = .09
    navigation_approach_first: bool = True
    work_height_radius: float = .15
    work_azimuth_radius: float = .45
    recovery_tilt_radius: float = .02
    recovery_clearance: float = .005
    sensor_work_safety_enabled: bool = False
    sensor_work_arm_speed: float = .24
    sensor_work_arm_acceleration: float = .65
    sensor_work_load_margin: float = .10
    sensor_work_azimuth_search: float = 1.0


def _unit(value):
    value = np.asarray(value, dtype=float)
    return value/max(float(np.linalg.norm(value)), 1e-12)


def _goal_rotation(axis):
    x = _unit(axis)
    up = np.array([0., 0., 1.]) if abs(x[2]) < .95 else np.array([0., 1., 0.])
    y = _unit(np.cross(up, x))
    return np.column_stack((x, y, np.cross(x, y)))


class PeriodicSurfacePlanner:
    """A* on the supplied periodic surface map, covering all six wheel contacts."""
    def __init__(self, config, height_limits, tilt_limits):
        self.config = config
        self.height_limits = np.asarray(height_limits, dtype=float)
        maximum_tilt = float(np.max(np.abs(tilt_limits)))
        self.height_margin = max(config.footprint_height, config.terrain_footprint_margin)+2*config.tree_radius*np.sin(maximum_tilt)
        self.theta_margin = max(config.footprint_azimuth, config.terrain_theta_margin)+maximum_tilt**2
        self._cache = {}
        self.reset_recovery()

    def reset_recovery(self):
        self._escape_goal = None
        self._escape_axis = None
        self._escape_map = None
        self.recovery_active = False
        self.recovery_geometry = None
        self.recovery_reason = None
        self._detour = None
        self._detour_index = 0
        self._detour_map = None
        self.navigation_phase = 'direct_route'
        self.navigation_obstacle = None
        self.navigation_pause_verified = False
        self.navigation_pause_state = None

    def _approach_route(self, start, goal, terrain):
        """Map-derived cardinal detour with a safe braking point.

        This chooses a route from geometry, rather than applying a timed
        animation. All segments retain the original six-wheel worst-tilt
        envelope. The ordinary A* remains the fallback for more complex maps.
        """
        direction = np.sign(goal[0]-start[0])
        if not direction or not self.safe(*start, terrain):
            return None
        obstacles = []
        for h0, h1, t0, t1 in self._rectangles(start, terrain):
            if t0 <= start[1] <= t1:
                entry, exit = (h0, h1) if direction > 0 else (h1, h0)
                if direction*(entry-start[0]) > .03 and direction*(goal[0]-exit) > .03:
                    obstacles.append((direction*(entry-start[0]), entry, exit, t0, t1))
        if not obstacles:
            return None
        _, entry, exit, t0, t1 = min(obstacles)
        approach_h = entry-direction*.030
        clear_h = exit+direction*.030
        goal_theta = start[1]+float(wrap_angle(goal[1]-start[1]))
        candidates = []
        for bypass_theta in (t0-.12, t1+.12):
            corners = np.array([[start[0], start[1]], [approach_h, start[1]],
                [approach_h, bypass_theta], [clear_h, bypass_theta],
                [clear_h, goal_theta], [goal[0], goal_theta]])
            # A segment may be safe yet have no whole-box QP corridor. Check
            # both properties before accepting the route.
            if not all(self.segment_safe(a,b,terrain) and
                       np.all(np.diff(self.corridor(a,b,terrain).reshape(2,2),axis=1) > 1e-8)
                       for a,b in zip(corners,corners[1:])):
                continue
            cost = 0.
            for a,b in zip(corners,corners[1:]):
                delta=b-a
                length=abs(delta[0])+self.config.tree_radius*abs(delta[1])
                count=max(2,int(np.ceil(length/.01)))
                support_cost=0.
                for f in np.linspace(0.,1.,count):
                    p=a+f*delta
                    for patch in self.patches(terrain):
                        if patch['kind']=='depression' and abs(p[0]-patch['height']) <= patch['height_half_width']+self.height_margin:
                            if np.any(np.abs(wrap_angle(p[1]+WHEEL_ANGLES-patch['theta'])) <= patch['theta_half_width']+self.theta_margin):
                                support_cost += 4*(1-patch['support_fraction'])/count
                cost += length*(1+support_cost)
            candidates.append((cost, corners))
        if not candidates:
            return None
        corners=min(candidates,key=lambda candidate:candidate[0])[1]
        return corners, dict(entry_height=float(entry),exit_height=float(exit),
            approach_height=float(approach_h),clear_height=float(clear_h),
            original_azimuth=float(start[1]),bypass_azimuth=float(corners[2,1]),
            target_azimuth=float(goal_theta),planning_uses_public_map=True)

    def _navigation_route(self, start, goal, terrain, fingerprint, base_rates):
        if fingerprint != self._detour_map:
            self._detour = None
        if self._detour is None:
            candidate=self._approach_route(start,goal,terrain)
            if candidate is None:
                return None
            self._detour,self.navigation_obstacle=candidate
            self._detour_index=1
            self._detour_map=fingerprint
            self.navigation_pause_verified=False
            self.navigation_pause_state=None
        target=self._detour[self._detour_index]
        rates=np.zeros(2) if base_rates is None else np.asarray(base_rates,dtype=float)
        if rates.shape!=(2,) or not np.isfinite(rates).all():
            raise ValueError('Navigation base rates must contain two finite measured values')
        arrived=abs(start[0]-target[0])<.012 and abs(float(wrap_angle(start[1]-target[1])))<.025
        stopped=abs(rates[0])<.006 and abs(rates[1])<.025
        if arrived and stopped:
            if self._detour_index==1:
                self.navigation_pause_verified=True
                self.navigation_pause_state=dict(height=float(start[0]),azimuth=float(start[1]),
                    height_speed=float(rates[0]),azimuth_speed=float(rates[1]),
                    height_speed_threshold=.006,azimuth_speed_threshold=.025,
                    outside_conservative_six_contact_obstacle=True)
            self._detour_index+=1
            if self._detour_index>=len(self._detour):
                self._detour=None
                self.navigation_phase='target_arrival'
                return None
            target=self._detour[self._detour_index]
        names={1:'approach_blocked_edge',2:'rotate_around_blocked',
               3:'climb_past_blocked',4:'return_to_target_azimuth',5:'resume_target_climb'}
        self.navigation_phase='brake_before_blocked' if self._detour_index==1 and arrived else names[self._detour_index]
        # Check the actual pose's connector again; drift or a new map can
        # invalidate a previously accepted route. Do not cross an unsafe box.
        if not self.segment_safe(start,target,terrain):
            self._detour=None
            self.navigation_phase='replan_after_drift'
            return None
        corridor=self.corridor(start,target,terrain)
        if corridor[1]-corridor[0]<1e-8 or corridor[3]-corridor[2]<1e-8:
            self._detour=None
            self.navigation_phase='replan_after_drift'
            return None
        return np.vstack([start,self._detour[self._detour_index:]]),True

    def _rectangles(self,current,terrain,geometry=None):
        """Include neighboring periodic copies, also for wide seam patches."""
        for patch in self.patches(terrain):
            if patch['kind']!='blocked':continue
            for i,beta in enumerate(WHEEL_ANGLES):
                hm=self.height_margin if geometry is None else geometry['height_margin']
                tm=self.theta_margin if geometry is None else geometry['theta_margin']
                h0=patch['height']-patch['height_half_width']-hm-(0 if geometry is None else geometry['height_high'][i])
                h1=patch['height']+patch['height_half_width']+hm-(0 if geometry is None else geometry['height_low'][i])
                offset=beta if geometry is None else geometry['theta_center'][i]
                center=current[1]+float(wrap_angle(patch['theta']-offset-current[1]))
                for shift in (-2*np.pi,0.,2*np.pi):
                    yield h0,h1,center+shift-patch['theta_half_width']-tm,center+shift+patch['theta_half_width']+tm

    def _contact_corridor(self, current, waypoint, terrain, geometry):
        """Whole-box separation using all six measured contact envelopes.

        This is used only for an escape connector. Normal routes continue to
        use the original worst-tilt envelope, including its original margins.
        """
        c=self.config.recovery_clearance
        box=np.array([max(self.height_limits[0],min(current[0],waypoint[0])-c),
            min(self.height_limits[1],max(current[0],waypoint[0])+c),
            min(current[1],waypoint[1])-c/self.config.tree_radius,
            max(current[1],waypoint[1])+c/self.config.tree_radius])
        for h0,h1,t0,t1 in self._rectangles(current,terrain,geometry):
            sides=[]
            if max(current[0],waypoint[0])<h0-1e-6:sides.append((h0-max(current[0],waypoint[0]),0,h0-1e-5))
            if min(current[0],waypoint[0])>h1+1e-6:sides.append((min(current[0],waypoint[0])-h1,1,h1+1e-5))
            if max(current[1],waypoint[1])<t0-1e-6:sides.append((self.config.tree_radius*(t0-max(current[1],waypoint[1])),2,t0-1e-5))
            if min(current[1],waypoint[1])>t1+1e-6:sides.append((self.config.tree_radius*(min(current[1],waypoint[1])-t1),3,t1+1e-5))
            if not sides:return None
            _,side,bound=max(sides)
            if side==0:box[1]=min(box[1],bound)
            elif side==1:box[0]=max(box[0],bound)
            elif side==2:box[3]=min(box[3],bound)
            else:box[2]=max(box[2],bound)
        return box if box[0]<=box[1] and box[2]<=box[3] else None

    def _monotone_escape(self,start,end,terrain):
        """Cardinal escape may exit existing buffers, never enter a new one."""
        delta=np.array([end[0]-start[0],float(wrap_angle(end[1]-start[1]))])
        if np.count_nonzero(np.abs(delta)>1e-10)!=1:return False
        for h0,h1,t0,t1 in self._rectangles(start,terrain):
            inside=h0<=start[0]<=h1 and t0<=start[1]<=t1
            if inside:
                axis=0 if abs(delta[0])>1e-10 else 1
                midpoint=(h0+h1)/2 if axis==0 else (t0+t1)/2
                if (start[axis]-midpoint)*delta[axis]<-1e-12:return False
            elif not(max(start[0],end[0])<h0 or min(start[0],end[0])>h1 or
                     max(start[1],end[1])<t0 or min(start[1],end[1])>t1):return False
        return True

    def _clear_connector(self,point,terrain):
        c=self.config.recovery_clearance
        if point[0]-c<self.height_limits[0] or point[0]+c>self.height_limits[1]:return False
        for patch in self.patches(terrain):
            if patch['kind']!='blocked':continue
            overlap_h=abs(point[0]-patch['height'])<=patch['height_half_width']+self.height_margin+c
            overlap_t=np.any(np.abs(wrap_angle(point[1]+WHEEL_ANGLES-patch['theta']))<=
                patch['theta_half_width']+self.theta_margin+c/self.config.tree_radius)
            if overlap_h and overlap_t:return False
        return True

    def plan(self,start,goal,terrain,recovery_geometry=None,*,navigation=False,base_rates=None):
        start=np.asarray(start,dtype=float);goal=np.asarray(goal,dtype=float)
        fingerprint=hashlib.sha256(json.dumps(self.patches(terrain),sort_keys=True).encode()).hexdigest()
        self.recovery_active=False;self.recovery_geometry=None;self.recovery_reason=None
        if not navigation:
            self._detour=None
            self.navigation_phase='work_positioning'
        elif self.config.navigation_approach_first and self.safe(*start,terrain):
            navigation_route=self._navigation_route(start,goal,terrain,fingerprint,base_rates)
            if navigation_route is not None:
                return navigation_route
        elif not self.safe(*start,terrain):
            self._detour=None
            self.navigation_phase='conservative_margin_recovery'
        if fingerprint!=self._escape_map:self._escape_goal=None
        if self._escape_goal is not None:
            self._escape_goal[1]=start[1]+float(wrap_angle(self._escape_goal[1]-start[1]))
            self._escape_goal[1-self._escape_axis]=start[1-self._escape_axis]
            if (abs(start[0]-self._escape_goal[0])<.006 and abs(start[1]-self._escape_goal[1])<.015
                    and self._clear_connector(start,terrain)):
                self._escape_goal=None
        if self._escape_goal is None and self.safe(*start,terrain):
            route,found=self._plan_regular(start,goal,terrain)
            if navigation and self.navigation_phase!='target_arrival':
                self.navigation_phase='no_safe_route' if not found else ('map_detour' if len(route)>2 else 'direct_route')
            return route,found
        if recovery_geometry is None:
            self._escape_goal=None
            return np.asarray([start]),False
        if self._contact_corridor(start,start,terrain,recovery_geometry) is None:
            self._escape_goal=None;self.recovery_reason='measured_contact_envelope_blocked'
            return np.asarray([start]),False
        latched=None if self._escape_goal is None else self._escape_goal.copy()
        candidates=[]
        # Local cardinal candidates avoid a diagonal crossing or a sampled
        # thin obstacle. Their complete rectangular connectors are checked.
        for axis,step in ((0,self.config.planner_height_step),(1,2*np.pi/self.config.planner_theta_bins)):
            for sign in (-1,1):
                for distance in range(1,7):
                    point=start.copy();point[axis]+=sign*distance*step
                    candidates.append(point)
        candidates.sort(key=lambda p:abs(p[0]-start[0])+self.config.tree_radius*abs(p[1]-start[1]))
        if latched is not None:candidates.insert(0,latched)
        for connector in candidates:
            if not self._clear_connector(connector,terrain) or not self._monotone_escape(start,connector,terrain):continue
            corridor=self._contact_corridor(start,connector,terrain,recovery_geometry)
            if corridor is None:continue
            route,found=self._plan_regular(connector,goal,terrain)
            if not found:continue
            self._escape_goal=connector.copy();self._escape_map=fingerprint
            self._escape_axis=int(abs(connector[1]-start[1])>1e-10)
            self.recovery_active=True;self.recovery_geometry=recovery_geometry
            self.recovery_reason='escape_conservative_margin'
            return np.vstack([start,route]),True
        self._escape_goal=None;self.recovery_reason='no_safe_monotone_escape_connector'
        return np.asarray([start]),False

    @staticmethod
    def patches(terrain):
        if terrain is None:
            return []
        if isinstance(terrain, dict):
            terrain = terrain.get('patches', terrain.get('regions', []))
        patches = []
        for item in terrain:
            center_theta = item.get('theta', item.get('theta_center'))
            center_height = item.get('height', item.get('height_center'))
            half_theta = item.get('theta_half_width', item.get('theta_width', 0.)/2)
            half_height = item.get('height_half_width', item.get('height_width', 0.)/2)
            values = np.array([center_theta, center_height, half_theta, half_height], dtype=float)
            if not np.isfinite(values).all() or min(half_theta, half_height) < 0:
                raise ValueError('Surface rectangles require finite centers and nonnegative half widths')
            patches.append(dict(kind=item['kind'], theta=float(center_theta), height=float(center_height),
                                theta_half_width=float(half_theta), height_half_width=float(half_height),
                                support_fraction=float(item.get('support_fraction', 1.))))
        return patches

    def safe(self, height, theta, terrain):
        if not self.height_limits[0] <= height <= self.height_limits[1]:
            return False
        for patch in self.patches(terrain):
            if patch['kind'] != 'blocked':
                continue
            if abs(height-patch['height']) <= patch['height_half_width']+self.height_margin:
                delta = wrap_angle(theta+WHEEL_ANGLES-patch['theta'])
                if np.any(np.abs(delta) <= patch['theta_half_width']+self.theta_margin):
                    return False
        return True

    def segment_safe(self, start, end, terrain):
        delta = np.array([end[0]-start[0], float(wrap_angle(end[1]-start[1]))])
        steps = max(2, int(np.ceil(max(abs(delta[0])/self.config.planner_height_step,
                                      abs(delta[1])/(2*np.pi/self.config.planner_theta_bins))*3)))
        return all(self.safe(*(np.asarray(start)+fraction*delta), terrain) for fraction in np.linspace(0, 1, steps+1))

    def _plan_regular(self, start, goal, terrain):
        patches = self.patches(terrain)
        unwrapped_goal=np.array([goal[0],start[1]+float(wrap_angle(goal[1]-start[1]))])
        direct_corridor=self.corridor(start,unwrapped_goal,patches)
        direct_box_has_extent=direct_corridor[1]-direct_corridor[0]>1e-8 and direct_corridor[3]-direct_corridor[2]>1e-8
        if direct_box_has_extent and self.segment_safe(start, goal, patches):
            return np.asarray([start,unwrapped_goal]), True
        heights = np.arange(self.height_limits[0], self.height_limits[1]+1e-10, self.config.planner_height_step)
        thetas = np.arange(self.config.planner_theta_bins)*2*np.pi/self.config.planner_theta_bins-np.pi
        fingerprint = hashlib.sha256(json.dumps(patches, sort_keys=True).encode()).hexdigest()
        if fingerprint not in self._cache:
            occupancy = np.array([[self.safe(h, t, patches) for t in thetas] for h in heights], dtype=bool)
            self._cache = {fingerprint: occupancy}
        occupancy = self._cache[fingerprint]
        def nearest(point):
            return int(np.argmin(np.abs(heights-point[0]))), int(np.argmin(np.abs(wrap_angle(thetas-point[1]))))
        def point(node):
            return np.array([heights[node[0]], thetas[node[1]]])
        def attach(position):
            center=nearest(position)
            if occupancy[center] and self.segment_safe(position,point(center),patches):return center
            candidates=[]
            for dh in range(-2,3):
                for dt in range(-2,3):
                    node=(center[0]+dh,(center[1]+dt)%len(thetas))
                    if not 0<=node[0]<len(heights) or not occupancy[node]:continue
                    p=point(node)
                    if self.segment_safe(position,p,patches):
                        cost=abs(p[0]-position[0])+self.config.tree_radius*abs(float(wrap_angle(p[1]-position[1])))
                        candidates.append((cost,node))
            return min(candidates)[1] if candidates else None
        if not self.safe(*start,patches) or not self.safe(*goal,patches):return np.asarray([start]),False
        # Rounding a safe continuous start onto an occupied grid node must
        # not declare the entire route disconnected. Every alternate connector
        # still satisfies the original full conservative contact envelope.
        source,destination=attach(start),attach(goal)
        if source is None or destination is None:return np.asarray([start]),False
        def heuristic(node):
            p = point(node)
            return abs(p[0]-goal[0])+self.config.tree_radius*abs(float(wrap_angle(p[1]-goal[1])))
        queue = [(heuristic(source), 0., source)]
        costs, parents = {source: 0.}, {}
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost > costs[node]+1e-12:
                continue
            if node == destination:
                nodes = [node]
                while nodes[-1] != source:
                    nodes.append(parents[nodes[-1]])
                path = [np.asarray(start)]
                for selected in reversed(nodes):
                    p = point(selected)
                    p[1] = path[-1][1]+float(wrap_angle(p[1]-path[-1][1]))
                    path.append(p)
                finish = np.array([goal[0], path[-1][1]+float(wrap_angle(goal[1]-path[-1][1]))])
                if self.segment_safe(path[-1], finish, patches):
                    path.append(finish)
                # Grid endpoints must also connect safely to the actual pose.
                if not all(self.segment_safe(a, b, patches) for a, b in zip(path, path[1:])):
                    return np.asarray([start]), False
                return np.asarray(path), True
            for dh, dt in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (node[0]+dh, (node[1]+dt) % len(thetas))
                if not 0 <= neighbor[0] < len(heights) or not occupancy[neighbor]:
                    continue
                if not self.segment_safe(point(node), point(neighbor), patches):
                    continue
                weight = self.config.planner_height_step if dh else self.config.tree_radius*2*np.pi/len(thetas)
                p = point(neighbor)
                for patch in patches:
                    if patch['kind'] == 'depression' and abs(p[0]-patch['height']) < patch['height_half_width']+self.height_margin:
                        if np.any(np.abs(wrap_angle(p[1]+WHEEL_ANGLES-patch['theta'])) < patch['theta_half_width']+self.theta_margin):
                            weight *= 1+4*(1-np.clip(patch['support_fraction'], 0, 1))
                next_cost = cost+weight
                if next_cost < costs.get(neighbor, np.inf):
                    costs[neighbor], parents[neighbor] = next_cost, node
                    heapq.heappush(queue, (next_cost+heuristic(neighbor), next_cost, neighbor))
        return np.asarray([start]), False

    def waypoint(self, path, terrain=None):
        if self.recovery_active:
            # Complete the short latched escape before ordinary lookahead.
            return self._escape_goal.copy()
        if self._detour is not None:
            return self._detour[self._detour_index].copy()
        if len(path) < 2:
            return path[0].copy()
        # Replanning adds a short actual-pose -> nearest-grid connector. A
        # collinearity test on that connector can trap the robot indefinitely.
        # Advance by path length, retaining only points directly visible through
        # the same six-contact occupancy check used by A*.
        result = path[0].copy()
        remaining = self.config.planner_lookahead
        for previous, following in zip(path, path[1:]):
            delta = following-previous
            length = abs(delta[0])+self.config.tree_radius*abs(delta[1])
            fraction = min(1., remaining/max(length, 1e-12))
            candidate = previous+fraction*delta
            if not self.segment_safe(path[0], candidate, terrain):
                break
            corridor=self.corridor(path[0],candidate,terrain)
            if corridor[1]-corridor[0]<1e-8 or corridor[3]-corridor[2]<1e-8:
                # A safe diagonal around a rectangle corner may still have no
                # safe rectangular MPC corridor. Preserve the earlier point
                # on the cardinal route instead of freezing at the actual pose.
                break
            result = candidate.copy()
            remaining -= length*fraction
            if fraction < 1 or remaining <= 1e-10:
                break
        return result

    def corridor(self, current, waypoint, terrain, work=False):
        if self.recovery_active:
            corridor=self._contact_corridor(current,waypoint,terrain,self.recovery_geometry)
            if corridor is not None:return corridor
            return np.array([current[0],current[0],current[1],current[1]])
        h_low, h_high = self.height_limits
        t_low, t_high = current[1]-np.pi+.01, current[1]+np.pi-.01
        if work:
            h_low, h_high = max(h_low, current[0]-self.config.work_height_radius), min(h_high, current[0]+self.config.work_height_radius)
            t_low, t_high = current[1]-self.config.work_azimuth_radius, current[1]+self.config.work_azimuth_radius
        for patch in self.patches(terrain):
            if patch['kind'] != 'blocked':
                continue
            for beta in WHEEL_ANGLES:
                center = current[1]+float(wrap_angle(patch['theta']-beta-current[1]))
                h0, h1 = patch['height']-patch['height_half_width']-self.height_margin, patch['height']+patch['height_half_width']+self.height_margin
                t0, t1 = center-patch['theta_half_width']-self.theta_margin, center+patch['theta_half_width']+self.theta_margin
                candidates = []
                if max(current[0], waypoint[0]) < h0-1e-6:
                    candidates.append(('below', (h0-max(current[0], waypoint[0]))))
                if min(current[0], waypoint[0]) > h1+1e-6:
                    candidates.append(('above', (min(current[0], waypoint[0])-h1)))
                if max(current[1], waypoint[1]) < t0-1e-6:
                    candidates.append(('left', self.config.tree_radius*(t0-max(current[1], waypoint[1]))))
                if min(current[1], waypoint[1]) > t1+1e-6:
                    candidates.append(('right', self.config.tree_radius*(min(current[1], waypoint[1])-t1)))
                if not candidates:
                    # Keep a point corridor when a continuous diagonal cannot
                    # be represented by a safe axis-aligned local corridor.
                    return np.array([current[0], current[0], current[1], current[1]])
                side = max(candidates, key=lambda item: item[1])[0]
                if side == 'below': h_high = min(h_high, h0-1e-5)
                elif side == 'above': h_low = max(h_low, h1+1e-5)
                elif side == 'left': t_high = min(t_high, t0-1e-5)
                else: t_low = max(t_low, t1+1e-5)
        return np.array([h_low, h_high, t_low, t_high])


class TreeWorkController:
    def __init__(self, model, mode='mpc', config=None):
        self.model = model
        self.mode = 'residual_mpc' if mode == 'mpc_rl' else mode
        if self.mode not in ('mpc', 'adaptive_mpc', 'residual_mpc', 'sequential'):
            raise ValueError('Unknown controller mode')
        if config is None:
            self.config = TreeWorkMPCConfig()
        elif isinstance(config, TreeWorkMPCConfig):
            self.config = config
        elif isinstance(config, dict):
            unknown = set(config)-{item.name for item in fields(TreeWorkMPCConfig)}
            if unknown:raise ValueError(f'Unknown MPC configuration: {sorted(unknown)}')
            self.config = TreeWorkMPCConfig(**config)
        else:
            raise TypeError('config must be a configuration or dict')
        c = self.config
        for item in fields(TreeWorkMPCConfig):
            value = getattr(c, item.name)
            if isinstance(value, (float, int, np.floating, np.integer)) and not np.isfinite(value):
                raise ValueError(f'{item.name} must be finite')
        for name in ('horizon','solve_interval','ik_iterations','planner_theta_bins'):
            value = getattr(c,name)
            if isinstance(value, (bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('prediction_dt','wheel_max','nominal_friction','normal_force','traction_gain','contact_stiffness','wheel_radius',
                     'tree_radius','arm_speed_max','max_base_speed','max_azimuth_speed','max_work_base_speed',
                     'max_work_azimuth_speed','arm_velocity_time_constant',
                     'max_tilt','height_tolerance','azimuth_tolerance','ee_position_weight','ee_orientation_weight',
                     'planner_height_step','planner_lookahead','work_height_radius','work_azimuth_radius',
                     'sensor_work_arm_speed','sensor_work_arm_acceleration','sensor_work_load_margin',
                     'sensor_work_azimuth_search'):
            if getattr(c,name) <= 0:raise ValueError(f'{name} must be positive')
        if not 0<c.recovery_tilt_radius<=.05 or not 0<c.recovery_clearance<=.025:
            raise ValueError('Recovery tube and connector clearance must be bounded and positive')
        for name in ('height_residual_scale','azimuth_residual_scale','ee_residual_scale','footprint_height',
                     'footprint_azimuth','terrain_footprint_margin','terrain_theta_margin'):
            if getattr(c,name) < 0:raise ValueError(f'{name} must be nonnegative')
        if c.horizon < 2 or c.solve_interval < 1 or c.prediction_dt <= 0 or c.planner_theta_bins < 12:
            raise ValueError('Invalid horizon, solve schedule or periodic grid')
        if c.min_height>=c.max_height:raise ValueError('Height safety limits must be ordered')
        limits = np.asarray(c.model_residual_scales, dtype=float)
        if limits.shape != (4,) or not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError('model correction needs four positive finite scales')
        if not isinstance(c.residual_use_adaptation, bool) or not 0 <= c.residual_model_scale <= 1 or not 0 < c.adaptation_rate <= 1:
            raise ValueError('Invalid adaptation configuration')
        if not isinstance(c.navigation_approach_first, bool):
            raise ValueError('navigation_approach_first must be bool')
        if not isinstance(c.sensor_work_safety_enabled, bool):
            raise ValueError('sensor_work_safety_enabled must be bool')
        if c.sensor_work_arm_speed > c.arm_speed_max or c.sensor_work_azimuth_search > 1.2:
            raise ValueError('Cooperative work speeds/search must stay inside the declared actuator/workspace envelope')
        self._scratch = mujoco.MjData(model)
        self.base_joints = np.array([self._name(mujoco.mjtObj.mjOBJ_JOINT, name) for name in BASE_JOINTS])
        self.arm_joints = np.array([self._name(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS])
        if any(model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE for joint in self.arm_joints):
            raise ValueError('All six arm pose axes must be genuine revolute joints')
        self.base_qpos = model.jnt_qposadr[self.base_joints]
        self.base_dofs = model.jnt_dofadr[self.base_joints]
        self.arm_qpos = model.jnt_qposadr[self.arm_joints]
        self.arm_dofs = model.jnt_dofadr[self.arm_joints]
        self.pose_dofs = np.r_[self.base_dofs, self.arm_dofs]
        self.pose_qpos = np.r_[self.base_qpos, self.arm_qpos]
        self.site_id = self._name(mujoco.mjtObj.mjOBJ_SITE, 'cut_site')
        self.nozzle_id = self._name(mujoco.mjtObj.mjOBJ_SITE, 'nozzle_site')
        self.root_id = self._name(mujoco.mjtObj.mjOBJ_BODY, 'climber_root')
        self.q_min = model.jnt_range[self.arm_joints, 0]+.015
        self.q_max = model.jnt_range[self.arm_joints, 1]-.015
        native_height=model.jnt_range[self.base_joints[0]]
        self.height_limits=np.array([max(native_height[0],c.min_height)+.015,
                                     min(native_height[1],c.max_height)-.015])
        if self.height_limits[0]>=self.height_limits[1]:raise ValueError('No valid safe height interval')
        self.tilt_limits = model.jnt_range[self.base_joints[2:]].copy()
        if np.any(model.jnt_stiffness[self.base_joints[2:]] != 0):
            raise ValueError('passive tilt must not have a spring to zero')
        actuators = [self._name(mujoco.mjtObj.mjOBJ_ACTUATOR, f'arm_{i}') for i in range(1, 7)]
        self.arm_kp = model.actuator_gainprm[actuators, 0].copy()
        wheels = [self._name(mujoco.mjtObj.mjOBJ_ACTUATOR, f'motor_wheel{i}') for i in range(1, 7)]
        wheel_joints = [self._name(mujoco.mjtObj.mjOBJ_JOINT, f'wheel{i}_spin') for i in range(1, 7)]
        self.wheel_dofs=model.jnt_dofadr[wheel_joints]
        self.wheel_torque_limit=np.minimum(np.abs(model.actuator_ctrlrange[wheels,0]),model.actuator_ctrlrange[wheels,1])
        self._base_body=self._name(mujoco.mjtObj.mjOBJ_BODY,'climber_root')
        moving_root=self._name(mujoco.mjtObj.mjOBJ_BODY,'height_carriage')
        nominal_force=float(model.body_subtreemass[moving_root])*9.81/(6*np.cos(np.pi/4))
        self._holding_torque=np.full(6,nominal_force*c.wheel_radius)
        self._traction_sites = [self._name(mujoco.mjtObj.mjOBJ_SITE,f'traction{i}') for i in range(1,7)]
        self.mixer = wheel_mixer(c.tree_radius, c.wheel_radius)
        self.force_map = wheel_force_map(c.tree_radius)
        self.rolling_force_input=self.mixer@np.linalg.inv(self.force_map@self.mixer)
        self.rolling_force_velocity=np.zeros((6,4));self.rolling_force_bias=np.zeros(6)
        # Wheel-speed constraints concern rolling speed; physical commands
        # are separately allocated motor torques, with no slip-speed bias.
        self.command_force_map=np.zeros((6,2))
        self.command_velocity_map=np.zeros((6,4));self.command_velocity_map[:,:2]=self.mixer
        self.command_bias=np.zeros(6)
        # Updated from the measured native pose before every optimization.
        self.contact_velocity_map = np.zeros((6,4))
        self.contact_velocity_map[:,:2] = c.wheel_radius*self.mixer
        self._actuation_map = np.zeros((4, 2));self._actuation_map[:2] = np.eye(2)
        self.planner = PeriodicSurfacePlanner(c, self.height_limits, self.tilt_limits)
        self._tilt_axis_limit=c.max_tilt*np.cos(np.pi/8)
        self._coupled_tilt_map=np.zeros((2,20))
        self._coupled_tilt_map[:,[4,6]]=[[1,1],[1,-1]]
        self._coupled_tilt_limit=np.sqrt(2)*self._tilt_axis_limit
        nz = c.horizon*8
        self._solver = cs.conic(f'tree_work_qp_{id(self)}', 'daqp',
                                {'h':cs.Sparsity.dense(nz,nz), 'a':cs.Sparsity.dense(c.horizon*34,nz)},
                                {'print_time':False,'error_on_fail':False})
        self.reset()

    def _name(self, kind, name):
        value=mujoco.mj_name2id(self.model,kind,name)
        if value<0:raise ValueError(f'model lacks {name}')
        return value

    def reset(self):
        self.planner.reset_recovery()
        self._step=0;self._last_stage=None;self._last_solution=np.zeros(self.config.horizon*8)
        self._base_force=np.zeros(2);self._arm_velocity=np.zeros(6)
        self._disturbance=np.zeros(4);self._learned=np.zeros(4)
        self._previous_predicted=None;self._q_command=None;self._last_arm_command=None
        self._arrival_pose=None;self._arrival_ik=None;self._last_ik=None;self._diagnostics={}
        self._operation_base_pose=None
        self._work_base_pose=None;self._work_ik=None;self._work_phase=None
        self._work_plan_diagnostics={};self._work_retract_yaw=None
        self._issued_arm_velocity=np.zeros(6)
        self._recovery_was_active=False
        self._solve_count=0;self._fallback_count=0;self._held_reference=None
        self._feedforward=np.zeros(6)
        self._command_history=deque()
        self._sensor_step=None;self._sensor_x=None
        self._sensor_delay_steps=None;self._sensor_dt=None
        self._observer_last_stiction=None

    def _delay_steps(self,state,dt):
        """The public fixed sampling delay, expressed in controller calls.

        Reset supplies repeated copies of its first sample while the delay
        queue fills. Their capture step is zero, rather than a negative time.
        The environment calls this controller once per declared dt; no live
        MuJoCo state or material/load parameter is available here.
        """
        delay=float(state.get('sensor_delay_seconds',0.))
        if not np.isfinite(delay) or delay<0:raise ValueError('Invalid sensor delay')
        steps=int(round(delay/dt))
        if abs(delay-steps*dt)>1e-8:raise ValueError('Sensor delay must be an integer number of samples')
        if self._sensor_delay_steps is None:
            self._sensor_delay_steps=steps;self._sensor_dt=dt
        elif steps!=self._sensor_delay_steps or abs(dt-self._sensor_dt)>1e-10:
            raise ValueError('Sampling interval or sensor delay changed; reset the controller')
        return steps

    def _replay_commands(self,x,start,end):
        """Replay only issued inputs over their actual sampling intervals.

        Each transition is rebuilt at the measurement/predicted pose. A
        cached affine dry-friction branch is invalid at a different pose or
        velocity and can otherwise invent passive motion during sticking.
        Predictions approximate an unknown plant; they are never a measured
        current pose, a hidden payload estimate, or a contact-force oracle.
        """
        result=x.copy();records={item['step']:item for item in self._command_history}
        for step in range(start,end):
            if step not in records:raise RuntimeError('Missing issued-input history for delayed sample')
            record=records[step]
            result=self._observer_transition(result,record,self._sensor_dt or record.get('dt',.04))
        return result

    def _observer_stiction(self,x,terms,drive,arm_velocity,correction,dt):
        """Nominal input-aware dry-friction reactions, in native joint units.

        Enumerate both passive axes' stick/slip subsets. A stationary axis
        holds its measured angle only when the coupled required reaction is
        within the original nominal friction limit (or its mechanical stop
        supports an outward load). There is no zero-angle objective or new
        actuator. Unknown payload/contact material are never consulted.
        """
        mass,bias,_=terms;rates=x[[1,3,5,7]]
        decay=np.exp(-dt/self.config.arm_velocity_time_constant)
        arm_acceleration=(arm_velocity-x[14:20])*(1-decay)/dt
        net=drive-bias-self._effective_base_damping@rates-mass[:4,4:]@arm_acceleration+mass[:4,:4]@correction
        friction=self.model.dof_frictionloss[self.base_dofs]
        stop_sign=np.zeros(2)
        for dim,(angle,rate) in enumerate(((4,5),(6,7))):
            low,high=self.tilt_limits[dim]
            if x[angle]>=high-.004 and x[rate]>=-.01:stop_sign[dim]=1.
            elif x[angle]<=low+.004 and x[rate]<=.01:stop_sign[dim]=-1.
        eligible=(np.abs(rates[2:])<=.015)|(stop_sign!=0)
        for subset in ((True,True),(True,False),(False,True),(False,False)):
            held=np.asarray(subset,dtype=bool)
            if np.any(held&~eligible):continue
            free=np.r_[0,1,np.flatnonzero(~held)+2]
            inverse=np.linalg.inv(mass[np.ix_(free,free)])
            unresisted=inverse@net[free]
            resisting=np.zeros(4)
            resisting[free]=friction[free]*np.where(np.abs(rates[free])>.015,np.sign(rates[free]),np.sign(unresisted))
            acceleration=np.zeros(4);acceleration[free]=inverse@(net[free]-resisting[free])
            reaction=net[2:]-mass[2:4,:4]@acceleration
            # A tiny conservative margin avoids asserting sticking exactly
            # at an uncertain threshold; all original friction limits stay.
            within=np.abs(reaction)<=np.maximum(0.,friction[2:]-1e-4)
            at_stop=(stop_sign!=0)&(stop_sign*reaction>=0)
            if np.all(~held|within|at_stop):
                resisting[2:]=np.where(held,reaction,resisting[2:])
                return dict(static_axes=held,friction=resisting,correction=correction,
                    stop_supported_axes=held&at_stop&~within,
                    required_resisting_friction=reaction,friction_limits=friction[2:].copy())
        raise RuntimeError('No nominal passive friction subset')

    def _observer_transition(self,x,record,dt):
        # Nominal rolling rates are inferred from this sample's pose/rates.
        # Motor torque is the actual issued, commonly saturated command.
        self._copy_state(x);self._update_traction_inverse()
        wheel_rates=self.contact_velocity_map@x[[1,3,5,7]]/self.config.wheel_radius
        terms=self._model_terms(x,{'wheel_rates':wheel_rates})
        drive=self.contact_velocity_map.T@record['wheel_torque']/self.config.wheel_radius
        arm_velocity=record['input'][2:]
        correction=np.asarray(record['model_correction'])
        observer=self._observer_stiction(x,terms,drive,arm_velocity,correction,dt)
        forces=self.contact_velocity_map[:,:2].T@record['wheel_torque']/self.config.wheel_radius
        ad,bd,cd=self._linear_model(x,terms,dt,observer=observer)
        applied=np.r_[forces,arm_velocity]
        nominal=ad@x+bd@applied+cd
        ad,bd,cd=self._passive_projection(ad,bd,cd,nominal)
        self._observer_last_stiction=observer
        return ad@x+bd@applied+cd

    def _current_sensor_state(self,state,measured,predicted,lag):
        """Move delayed sensor geometry coherently with the nominal estimate.

        Position/orientation residuals in the raw sensors are retained. Only
        the kinematic change from measured to predicted q is added. Moving
        targets use their public measured velocity, without sway phase or
        target-center access. Original delayed observations remain unchanged
        for the policy and the environment's measured-stage gates.
        """
        result=dict(state)
        sites=[self.site_id,self.nozzle_id,*self._traction_sites]
        self._copy_state(measured,state);self._update_traction_inverse()
        before_position=self._scratch.site_xpos[sites].copy()
        before_rotation=self._scratch.site_xmat[sites[:2]].reshape(2,3,3).copy()
        before_rolling=self.contact_velocity_map@measured[[1,3,5,7]]/self.config.wheel_radius
        self._copy_state(predicted,state);self._update_traction_inverse()
        after_position=self._scratch.site_xpos[sites].copy()
        after_rotation=self._scratch.site_xmat[sites[:2]].reshape(2,3,3).copy()
        after_rolling=self.contact_velocity_map@predicted[[1,3,5,7]]/self.config.wheel_radius
        result.update(base_height=float(predicted[0]),base_azimuth=float(predicted[2]),
            base_rates=predicted[[1,3]].copy(),base_tilt=predicted[[4,6]].copy(),
            tilt_rates=predicted[[5,7]].copy(),arm_q=predicted[8:14].copy(),arm_dq=predicted[14:20].copy())
        for index,prefix in enumerate(('cut','nozzle')):
            key=prefix+'_pos'
            if key in state:result[key]=np.asarray(state[key])+after_position[index]-before_position[index]
            key=prefix+'_axis'
            if key in state:result[key]=after_rotation[index]@before_rotation[index].T@np.asarray(state[key])
        if 'wheel_positions' in state:
            result['wheel_positions']=np.asarray(state['wheel_positions'])+after_position[2:]-before_position[2:]
            # A conservative public-map estimate retains any low support in
            # the delayed sensor instead of declaring uncertain support safe.
            from envs.tree_work_env import terrain_support
            support=terrain_support(result['wheel_positions'],state.get('terrain_map',[]),
                self.config.terrain_footprint_margin,self.config.terrain_theta_margin)[0]
            result['measured_contact']=np.minimum(support,np.asarray(state.get('measured_contact',np.ones(6))))
        if 'wheel_rates' in state:
            result['wheel_rates']=np.asarray(state['wheel_rates'])+after_rolling-before_rolling
        target_change=lag*np.asarray(state.get('target_velocity',np.zeros(3)),dtype=float)
        result['ref_ee']=np.asarray(state['ref_ee'])+target_change
        if 'target_pos' in state:result['target_pos']=np.asarray(state['target_pos'])+target_change
        return result

    def _compensate_delay(self,state,x,steps,dt):
        capture=max(0,self._step-steps);innovation=np.zeros(4)
        if self._sensor_step is not None and capture>self._sensor_step:
            expected=self._replay_commands(self._sensor_x,self._sensor_step,capture)
            innovation=(x[[1,3,5,7]]-expected[[1,3,5,7]])/((capture-self._sensor_step)*dt)
        self._sensor_step=capture;self._sensor_x=x.copy()
        predicted=self._replay_commands(x,capture,self._step)
        current=self._current_sensor_state(state,x,predicted,(self._step-capture)*dt)
        return current,predicted,innovation,capture

    def _state_vector(self,state):
        required=('base_height','base_azimuth','base_rates','base_tilt','tilt_rates','arm_q','arm_dq','ref_height','ref_azimuth','ref_ee','ref_axis','stage','arm_home','dt')
        missing=set(required)-state.keys()
        if missing:raise ValueError(f'Missing control state: {sorted(missing)}')
        for key,size in [('base_rates',2),('base_tilt',2),('tilt_rates',2),('arm_q',6),('arm_dq',6),
                         ('ref_ee',3),('ref_axis',3),('arm_home',6)]:
            value=np.asarray(state[key],dtype=float)
            if value.shape!=(size,) or not np.isfinite(value).all():raise ValueError(f'Invalid {key}')
        for key in ('ref_height','ref_azimuth'):
            if not np.isscalar(state[key]) or not np.isfinite(state[key]):raise ValueError(f'Invalid {key}')
        if np.linalg.norm(state['ref_axis'])<1e-8:raise ValueError('ref_axis must be nonzero')
        if state['stage'] not in ('navigate','extend','realign','operate','retract','done'):
            raise ValueError('Unknown task stage')
        for key in ('measured_contact','wheel_rates'):
            if key in state and (np.asarray(state[key]).shape!=(6,) or not np.isfinite(state[key]).all()):
                raise ValueError(f'Invalid {key}')
        values=[state['base_height'],state['base_rates'][0],state['base_azimuth'],state['base_rates'][1],
                state['base_tilt'][0],state['tilt_rates'][0],state['base_tilt'][1],state['tilt_rates'][1]]
        x=np.r_[values,state['arm_q'],state['arm_dq']].astype(float)
        if x.shape!=(20,) or not np.isfinite(x).all() or not np.isfinite(state['dt']) or state['dt']<=0:
            raise ValueError('state requires twenty finite values and positive dt')
        for key,size in [('ref_ee',3),('ref_axis',3),('arm_home',6)]:
            if np.asarray(state[key]).shape!=(size,) or not np.isfinite(state[key]).all():
                raise ValueError(f'Invalid {key}')
        return x

    def _copy_state(self,x,state=None):
        self._scratch.qpos[self.base_qpos]=x[[0,2,4,6]]
        self._scratch.qvel[self.base_dofs]=x[[1,3,5,7]]
        self._scratch.qpos[self.arm_qpos]=x[8:14];self._scratch.qvel[self.arm_dofs]=x[14:20]
        if state is not None and 'wheel_rates' in state:
            wheel_dofs=[self.model.joint(f'wheel{i}_spin').dofadr[0] for i in range(1,7)]
            self._scratch.qvel[wheel_dofs]=state['wheel_rates']
        mujoco.mj_forward(self.model,self._scratch)

    def _kinematics(self):
        jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jacSite(self.model,self._scratch,jp,jr,self.site_id)
        return self._scratch.site_xpos[self.site_id].copy(),self._scratch.site_xmat[self.site_id].reshape(3,3).copy(),jp[:,self.pose_dofs],jr[:,self.pose_dofs]

    def _ik(self,target,rotation,x,initial):
        original=self._scratch.qpos.copy()
        q=np.clip(initial,self.q_min,self.q_max);best=q.copy();score=np.inf
        for _ in range(self.config.ik_iterations):
            self._scratch.qpos[self.arm_qpos]=q;mujoco.mj_forward(self.model,self._scratch)
            position,actual,jp,jr=self._kinematics()
            error=np.r_[target-position,.12*Rotation.from_matrix(rotation@actual.T).as_rotvec()]
            norm=float(error@error)
            if norm<score:best,score=q.copy(),norm
            jacobian=np.vstack((jp[:,4:],.12*jr[:,4:]))
            increment=jacobian.T@np.linalg.solve(jacobian@jacobian.T+.0003*np.eye(6),error)
            q=np.clip(q+np.clip(increment,-.18,.18),self.q_min,self.q_max)
            if norm<1e-7:break
        self._scratch.qpos[:]=original;mujoco.mj_forward(self.model,self._scratch)
        return best,float(np.sqrt(score))

    def _model_terms(self,x,state):
        self._copy_state(x,state)
        self._update_traction_inverse()
        full=np.zeros((self.model.nv,self.model.nv));mujoco.mj_fullM(self.model,full,self._scratch.qM)
        # No-slip projection reflects shaft inertia and damping into all four
        # native base coordinates. Omitting these six real rotors understates
        # climbing inertia by about 1.61 kg at the upright nominal pose.
        transform=np.zeros((self.model.nv,10));transform[self.pose_dofs]=np.eye(10)
        transform[self.wheel_dofs,:4]=self.contact_velocity_map/self.config.wheel_radius
        self._rolling_projection=transform
        mass=transform.T@full@transform
        self._effective_base_damping=(transform.T@np.diag(self.model.dof_damping)@transform)[:4,:4]
        bias=self._scratch.qfrc_bias[self.base_dofs].copy()
        derivative=np.zeros((4,10));saved=self._scratch.qpos.copy()
        for column,qpos in enumerate(self.pose_qpos):
            self._scratch.qpos[qpos]=saved[qpos]+1e-4;mujoco.mj_forward(self.model,self._scratch)
            high=self._scratch.qfrc_bias[self.base_dofs].copy()
            self._scratch.qpos[qpos]=saved[qpos]-1e-4;mujoco.mj_forward(self.model,self._scratch)
            low=self._scratch.qfrc_bias[self.base_dofs].copy()
            derivative[:,column]=(high-low)/2e-4
            self._scratch.qpos[qpos]=saved[qpos]
        self._scratch.qpos[:]=saved;mujoco.mj_forward(self.model,self._scratch)
        self._update_traction_inverse()
        self._feedforward=np.clip(self._scratch.qfrc_bias[self.arm_dofs]/self.arm_kp,-.10,.10)
        return mass,bias,derivative

    def _update_traction_inverse(self):
        """Measured-pose traction inverse; wheel commands remain rank two.

        Native site Jacobians include yaw and passive roll/pitch velocities.
        Their reaction forces may physically disturb tilt, but the optimizer
        has no independent tilt torque or wheel nullspace available to it.
        """
        c=self.config
        directions=rolling_directions(float(self._scratch.qpos[self.base_qpos[1]]),self._scratch.xmat[self._base_body])
        jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
        for i,site in enumerate(self._traction_sites):
            mujoco.mj_jacSite(self.model,self._scratch,jp,jr,site)
            self.contact_velocity_map[i]=directions[i]@jp[:,self.base_dofs]
        generalized=self.contact_velocity_map[:,:2].T
        # Only the two physical climbing/azimuth force coordinates are
        # allocated. Six independent torques or a tilt-control nullspace are
        # never offered to the optimizer or feedback policy.
        self.rolling_force_input=self.mixer@np.linalg.inv(generalized@self.mixer)
        self.torque_force_map=c.wheel_radius*self.rolling_force_input
        self._actuation_map=self.contact_velocity_map.T@self.rolling_force_input
        self._traction_velocity_force=np.zeros((4,4));self._traction_bias_force=np.zeros(4)

    def _correction(self):
        estimate=self._disturbance if (self.mode=='adaptive_mpc' or (self.mode=='residual_mpc' and self.config.residual_use_adaptation)) else np.zeros(4)
        learned=self._learned if self.mode=='residual_mpc' else np.zeros(4)
        return np.clip(estimate+learned,-np.asarray(self.config.model_residual_scales),self.config.model_residual_scales)

    def _linear_model(self,x,terms,dt,observer=None):
        mass,bias,derivative=terms;c=self.config
        # An outward-loaded native stop supplies its constraint reaction. The
        # remaining base coordinates use the corresponding constrained mass,
        # rather than the free four-coordinate inverse followed by angle-only
        # clipping (which leaves incorrect h/theta accelerations).
        locked=np.zeros(4,dtype=bool)
        for dim,(angle,rate) in enumerate(((4,5),(6,7))):
            low,high=self.tilt_limits[dim]
            direction=1. if x[angle]>=high-.004 else (-1. if x[angle]<=low+.004 else 0.)
            locked[dim+2]=bool(direction and direction*(-bias[dim+2])>0 and direction*x[rate]>=-.01)
        native_stop_active=locked[2:].copy()
        if observer is not None:locked[2:]=observer['static_axes']
        free=np.flatnonzero(~locked);inverse=np.zeros((4,4))
        inverse[np.ix_(free,free)]=np.linalg.inv(mass[np.ix_(free,free)])
        self._stop_active=native_stop_active;self._effective_base_inverse=inverse.copy()
        reaction=inverse@mass[:4,4:]
        a=np.zeros((20,20));b=np.zeros((20,8));offset=np.zeros(20)
        positions=np.r_[0,2,4,6,np.arange(8,14)];rates=np.r_[1,3,5,7,np.arange(14,20)]
        a[positions,rates]=1.
        a[np.ix_(rates[:4],positions)]=-inverse@derivative
        a[np.ix_(rates[:4],rates[:4])]-=inverse@self._effective_base_damping
        a[np.ix_(rates[:4],rates[:4])]+=inverse@self._traction_velocity_force
        a[np.ix_(rates[:4],rates[4:])]+=reaction/c.arm_velocity_time_constant
        b[np.ix_(rates[:4],[0,1])]=inverse@self._actuation_map
        b[np.ix_(rates[:4],np.arange(2,8))]=-reaction/c.arm_velocity_time_constant
        a[np.ix_(rates[4:],rates[4:])]=-np.eye(6)/c.arm_velocity_time_constant
        b[np.ix_(rates[4:],np.arange(2,8))]=np.eye(6)/c.arm_velocity_time_constant
        # Dry friction opposes motion, and static dry friction absorbs bias up
        # to its native limit. It never targets a zero tilt angle.
        friction=self.model.dof_frictionloss[self.base_dofs]
        velocity=x[rates[:4]]
        resisting=friction*np.where(np.abs(velocity)>.015,np.sign(velocity),-np.sign(bias))
        resisting=np.where(np.abs(velocity)>.015,resisting,np.clip(-bias,-friction,friction))
        correction=self._correction()
        if observer is not None:
            resisting=observer['friction']
            correction=inverse@mass[:4,:4]@observer['correction']
        offset[rates[:4]]=inverse@(-bias+derivative@x[positions]-resisting+self._traction_bias_force)+correction
        augmented=np.zeros((29,29));augmented[:20,:20]=a;augmented[:20,20:28]=b;augmented[:20,28]=offset
        discrete=expm(augmented*dt)
        for dim,(angle,rate) in enumerate(((4,5),(6,7))):
            if locked[dim+2]:
                # Hold this measured passive angle, never a zero-angle target.
                discrete[angle,:]=0.;discrete[angle,angle]=1.
                discrete[rate,:]=0.
        return discrete[:20,:20],discrete[:20,20:28],discrete[:20,28]

    def _passive_projection(self,ad,bd,cd,nominal):
        ad,bd,cd=ad.copy(),bd.copy(),cd.copy()
        for dim,(angle,rate) in enumerate(((4,5),(6,7))):
            low,high=self.tilt_limits[dim]
            if nominal[angle]<low or nominal[angle]>high:
                stop=np.clip(nominal[angle],low,high)
                ad[angle]=0.;bd[angle]=0.;cd[angle]=stop
                if (nominal[angle]>high and nominal[rate]>0) or (nominal[angle]<low and nominal[rate]<0):
                    ad[rate]=0.;bd[rate]=0.;cd[rate]=0.
        return ad,bd,cd

    def _refs(self,state,action,x):
        delta=np.zeros(6) if action is None else np.array(action,dtype=float,copy=True)
        if delta.shape!=(6,) or not np.isfinite(delta).all():raise ValueError('reference_action must have six finite entries')
        delta=np.clip(delta,-1,1)
        if self.mode!='residual_mpc':delta[:]=0
        height=float(state['ref_height'])+delta[0]*self.config.height_residual_scale
        theta=x[2]+float(wrap_angle(float(state['ref_azimuth'])+delta[1]*self.config.azimuth_residual_scale-x[2]))
        ee=np.asarray(state['ref_ee'])+delta[2:5]*self.config.ee_residual_scale
        speed=float(np.clip(1+.5*delta[5],.5,1.5))
        return height,theta,ee,_goal_rotation(state['ref_axis']),speed

    def _plan(self,state,x,reference):
        if state.get('sensor_only_navigation', False):
            # Deployable actors supply a local waypoint, never the hidden map.
            if state.get('terrain_map') or state.get('local_terrain_map'):
                raise ValueError('Sensor-only MPC refuses privileged terrain coordinates')
            start = x[[0, 2]].copy()
            supplied = state.get('navigation_waypoint') if state['stage'] == 'navigate' else None
            waypoint = np.asarray(reference[:2] if supplied is None else supplied, dtype=float)
            if waypoint.shape != (2,) or not np.isfinite(waypoint).all():
                raise ValueError('Learned navigation waypoint must contain two finite values')
            waypoint = waypoint.copy()
            waypoint[0] = np.clip(waypoint[0], self.config.min_height + .01, self.config.max_height - .01)
            waypoint[1] = start[1] + float(wrap_angle(waypoint[1] - start[1]))
            # Generic local workspace tube, not an obstacle-derived corridor.
            work = state['stage'] != 'navigate'
            dh = self.config.work_height_radius if work else .30
            dtheta = self.config.work_azimuth_radius if work else .80
            corridor = np.array([max(self.config.min_height, start[0] - dh),
                                 min(self.config.max_height, start[0] + dh),
                                 start[1] - dtheta, start[1] + dtheta])
            if work:
                waypoint = np.array([reference[0], start[1] + float(wrap_angle(reference[1] - start[1]))])
            waypoint = np.clip(waypoint, corridor[[0, 2]], corridor[[1, 3]])
            self.planner.reset_recovery()
            self.planner.navigation_phase = 'learned_sensor_waypoint' if not work else 'sensor_only_work'
            return np.vstack((start, waypoint)), True, waypoint, corridor
        start=x[[0,2]];goal=np.array(reference[:2])
        geometry=self._recovery_geometry(state,x)
        path,found=self.planner.plan(start,goal,state.get('terrain_map',[]),geometry,
            navigation=state['stage']=='navigate',base_rates=x[[1,3]])
        waypoint=self.planner.waypoint(path,state.get('terrain_map',[]))
        work=state['stage'] in ('realign','operate','retract','done')
        corridor=self.planner.corridor(start,waypoint if self.planner.recovery_active or not work else start,state.get('terrain_map',[]),work)
        return path,found,waypoint,corridor

    def _recovery_geometry(self,state,x):
        """Measured contact envelopes for a short locally bounded escape.

        No true material/load parameter or live plant state is used. Rigid
        rotations of both passive axes by at most radius move a site by at
        most 4*r*sin(radius/2). Three-sigma sensor and delayed-motion buffers
        supplement the unchanged physical map footprint margins. These are
        local measured-model bounds, not a nonlinear hardware certificate.
        """
        if 'wheel_positions' not in state:return None
        positions=np.asarray(state['wheel_positions'],dtype=float)
        if positions.shape!=(6,3) or not np.isfinite(positions).all():raise ValueError('Invalid measured wheel_positions')
        sigma=float(state.get('sensor_noise_sigma',0.));delay=float(state.get('sensor_delay_seconds',0.))
        if not np.isfinite([sigma,delay]).all() or min(sigma,delay)<0:raise ValueError('Invalid public sensor uncertainty metadata')
        radial=np.linalg.norm(positions[:,:2],axis=1)
        if np.min(radial)<.05:raise ValueError('Measured contact radius is invalid')
        offset=positions-np.array([0.,0.,x[0]])
        tilt_radius=self.config.recovery_tilt_radius+3*sigma+delay*np.max(np.abs(x[[5,7]]))
        if not np.isfinite(tilt_radius) or tilt_radius>=np.pi:return None
        position_error=3*np.sqrt(3)*sigma
        displacement=4*(np.max(np.linalg.norm(offset,axis=1))+position_error+3*sigma)*np.sin(tilt_radius/2)
        denominator=np.min(radial)-position_error
        if displacement+position_error>=denominator:return None
        # Sensor delay is buffered by the declared controller speed envelope;
        # current measured zero rates are not treated as a no-motion oracle.
        h_error=6*sigma+delay*self.config.max_base_speed
        theta_error=3*sigma+delay*self.config.max_azimuth_speed
        return dict(height_low=positions[:,2]-x[0]-displacement,
            height_high=positions[:,2]-x[0]+displacement,
            theta_center=wrap_angle(np.arctan2(positions[:,1],positions[:,0])-x[2]),
            height_margin=max(self.config.footprint_height,self.config.terrain_footprint_margin)+h_error,
            theta_margin=max(self.config.footprint_azimuth,self.config.terrain_theta_margin)+theta_error+
                float(np.arcsin((displacement+position_error)/denominator)),
            local_tilt_radius=self.config.recovery_tilt_radius,
            sensor_noise_sigma=sigma,sensor_delay_seconds=delay)

    def _work_static_load(self, x, state):
        """Nominal passive-axis load at a sensed pose, including climbing support.

        This is a robot-model calculation, not a terrain/payload query. The
        scratch data are independent of the plant. Only h/theta forces are
        allocated; the residual roll/pitch loads remain physically passive.
        """
        self._copy_state(x, state)
        self._update_traction_inverse()
        bias = self._scratch.qfrc_bias[self.base_dofs].copy()
        return (self._actuation_map @ bias[:2] - bias)[2:]

    def _initialize_cooperative_work(self, state, x, reference):
        """Find a reachable work posture without assuming the base can level.

        Folded shoulder yaw is moved first, then the other arm axes extend.
        The nominal gravity loads along that two-part path determine whether
        a measured passive axis would run to its mechanical stop. A bounded
        base-yaw search can redistribute the load between the passive axes.
        There is no zero-tilt objective, additional base input, or plant write.
        """
        c = self.config
        home = np.asarray(state['arm_home'], dtype=float)
        friction = self.model.dof_frictionloss[self.base_dofs[2:]]
        candidates = []
        grid = np.linspace(0., c.sensor_work_azimuth_search,
                           max(2, int(np.ceil(c.sensor_work_azimuth_search / .05)) + 1))
        for direction in (-1., 1.):
            seed = x[8:14].copy()
            for offset in grid:
                if direction > 0 and offset == 0:
                    continue
                pose = x.copy();pose[2] += direction * offset
                self._copy_state(pose, state)
                q, error = self._ik(reference[2], reference[3], pose, seed)
                seed = q.copy()
                if error > .008:
                    continue
                folded = home.copy();folded[0] = q[0]
                loads = []
                for qa, qb in ((home, folded), (folded, q)):
                    for fraction in np.linspace(0., 1., 13):
                        pose[8:14] = (1 - fraction) * qa + fraction * qb
                        loads.append(self._work_static_load(pose, state))
                loads = np.asarray(loads)
                peak = np.max(np.abs(loads), axis=0)
                predicted = np.tile(x[[4, 6]], (len(loads), 1))
                outward = np.abs(loads) > np.maximum(.01, friction - c.sensor_work_load_margin)
                stop = np.where(loads > 0., self.tilt_limits[:, 1], self.tilt_limits[:, 0])
                predicted[outward] = stop[outward]
                predicted_norm = float(np.max(np.linalg.norm(predicted, axis=1)))
                # A stop on one axis can be safe when the other measured axis
                # is near zero; a previously tilted ring may require a rotated
                # work posture so both axes remain under their friction limits.
                feasible = predicted_norm <= c.max_tilt - .007
                score = abs(offset) + 50 * max(0., predicted_norm - (c.max_tilt - .007)) + error
                candidates.append((not feasible, score, pose[[0, 2]].copy(), q.copy(),
                                   peak, predicted_norm, float(error)))
        self._copy_state(x, state);self._update_traction_inverse()
        if not candidates:
            self._work_base_pose = x[[0, 2]].copy()
            self._work_ik = home.copy()
            self._work_phase = 'unreachable_hold'
            self._work_plan_diagnostics = {'cooperative_work_plan_feasible': False,
                                          'cooperative_work_plan_reason': 'no reachable nominal pose'}
            return
        selected = min(candidates, key=lambda item: (item[0], item[1]))
        unsafe, _, self._work_base_pose, self._work_ik, peak, norm, error = selected
        self._work_phase = 'unreachable_hold' if unsafe else 'base_reposition'
        self._work_plan_diagnostics = dict(cooperative_work_plan_feasible=not unsafe,
            cooperative_work_plan_reason='nominal path and measured passive-tilt envelope',
            cooperative_work_base_pose=self._work_base_pose.tolist(),
            cooperative_work_base_reposition=(self._work_base_pose-x[[0, 2]]).tolist(),
            cooperative_nominal_path_peak_passive_load=peak.tolist(),
            cooperative_nominal_path_predicted_stop_tilt_norm=norm,
            cooperative_nominal_ik_error=error, cooperative_work_candidates=len(candidates),
            cooperative_work_uses_true_payload_or_terrain=False)

    def _cooperative_work_reference(self, state, x, reference):
        """Measured arm sequence and base anchor for the opt-in work controller."""
        home = np.asarray(state['arm_home'], dtype=float)
        stage = state['stage']
        base = self._work_base_pose.copy() if self._work_base_pose is not None else x[[0, 2]].copy()
        if stage == 'extend':
            if self._work_phase == 'base_reposition':
                error = base-x[[0, 2]]
                if abs(error[0]) < .014 and abs(error[1]) < .035 and np.linalg.norm(x[[1, 3]]) < .065:
                    self._work_phase = 'folded_shoulder_yaw'
            if self._work_phase == 'folded_shoulder_yaw':
                folded = home.copy();folded[0] = self._work_ik[0]
                if abs(x[8]-folded[0]) < .035 and np.max(np.abs(x[14:20])) < .09:
                    self._work_phase = 'extend_pitch_axes'
                else:
                    return base, folded
            if self._work_phase in ('extend_pitch_axes', 'ready'):
                self._copy_state(x, state)
                q, error = self._ik(reference[2], reference[3], x, self._work_ik)
                if error < .01:
                    self._work_ik = q.copy()
                if np.max(np.abs(x[8:14]-self._work_ik)) < .07 and np.max(np.abs(x[14:20])) < .15:
                    self._work_phase = 'ready'
                return base, self._work_ik.copy()
            return base, home.copy()
        if stage in ('realign', 'operate'):
            self._copy_state(x, state)
            q, _ = self._ik(reference[2], reference[3], x, x[8:14])
            return (self._operation_target(state, reference) if stage == 'operate' else base), q
        if stage == 'retract':
            q = home.copy();q[0] = self._work_retract_yaw
            if np.max(np.abs(x[9:14]-home[1:])) < .065 and np.max(np.abs(x[15:20])) < .12:
                self._work_phase = 'retract_shoulder_yaw'
            if self._work_phase == 'retract_shoulder_yaw':
                q = home.copy()
            else:
                self._work_phase = 'retract_pitch_axes'
            return (self._operation_base_pose.copy() if self._operation_base_pose is not None else base), q
        return base, home.copy()

    def _cooperative_arm_velocity(self, velocity, state, x, dt):
        """Slew physical arm commands; hold on predicted outward tilt drift."""
        c = self.config
        value = np.clip(velocity, -c.sensor_work_arm_speed, c.sensor_work_arm_speed)
        forecast = x[[4, 6]] + .16 * x[[5, 7]]
        if np.linalg.norm(forecast) > c.max_tilt-.004 and np.dot(x[[4, 6]], x[[5, 7]]) > 0:
            value[:] = 0.
        if self._work_phase in ('base_reposition', 'unreachable_hold'):
            value[:] = 0.
        change = c.sensor_work_arm_acceleration * dt
        return np.clip(value, self._issued_arm_velocity-change, self._issued_arm_velocity+change)

    def _solve(self,state,x,reference,terms,planning):
        started=perf_counter();c=self.config;n=c.horizon
        height,theta,ee,rotation,speed=reference
        path,found,waypoint,corridor=planning
        stage=state['stage'];home=np.asarray(state['arm_home'])
        recovering=self.planner.recovery_active
        position,actual,jp,jr=self._kinematics()
        move=stage in ('extend','realign','operate')
        ik_error=None
        cooperative = c.sensor_work_safety_enabled and stage != 'navigate'
        cooperative_target = self._cooperative_work_reference(state, x, reference) if cooperative else None
        if cooperative:
            qref = cooperative_target[1]
        elif stage=='extend':
            qref=self._arrival_ik.copy()
        elif move:
            # Seed from the actual measured posture. An old fixed-base IK
            # branch can be far from the arm once h/theta have repositioned.
            qref,ik_error=self._ik(ee,rotation,x,x[8:14]);self._last_ik=qref.copy()
        else:qref=home.copy()
        if recovering:qref=home.copy() if stage in ('navigate','done') else x[8:14].copy()
        if cooperative:base_target=cooperative_target[0]
        elif stage=='navigate' or recovering:base_target=waypoint
        elif stage=='extend':base_target=self._arrival_pose.copy()
        elif stage=='operate':base_target=self._operation_target(state,reference)
        else:base_target=np.array([height,theta])
        desired=np.r_[base_target[0],0.,base_target[1],0.,x[4],0.,x[6],0.,qref,np.zeros(6)]
        weights=np.array([80,8,35,3,0,0,0,0]+[2.]*6+[.2]*6)
        if stage=='extend':weights[[0,2]]=200.;weights[8:14]=60.
        if stage in ('realign','operate'):weights[[0,2]]=.8;weights[8:14]=.7
        if cooperative and stage == 'realign':weights[[0,2]]=[80.,35.]
        if stage=='operate':weights[[0,2]]=[80.,35.]
        if recovering:weights[[0,2]]=[80.,35.]
        if stage in ('navigate','retract','done'):weights[8:14]=50.
        pose_map=np.zeros((6,20));pose_map[:3,np.r_[0,2,4,6,np.arange(8,14)]]=jp
        pose_map[3:,np.r_[0,2,4,6,np.arange(8,14)]]=jr
        orientation_error=Rotation.from_matrix(actual@rotation.T).as_rotvec()
        track_pose=stage in ('realign','operate') and not recovering
        pose_weights=np.diag([c.ee_position_weight]*3+[c.ee_orientation_weight]*3)
        objective=np.diag(weights)+(pose_map.T@pose_weights@pose_map if track_pose else 0)
        nz=n*8;hessian=np.eye(nz)*1e-7;linear=np.zeros(nz)
        constraints=[];lower=[];upper=[];s=np.zeros((20,nz));v=x.copy()
        nominal_state=x.copy()
        nominal_inputs=self._last_solution.reshape(n,8).copy()
        if self._step==0:nominal_inputs[:,:2]=terms[1][:2]
        ad,bd,cd=self._linear_model(x,terms,c.prediction_dt)
        low=np.full(20,-np.inf);high=np.full(20,np.inf)
        low[[0,2]],high[[0,2]]=[corridor[0],corridor[2]],[corridor[1],corridor[3]]
        base_speed_limits=self._base_speed_limits(stage)
        low[[1,3]],high[[1,3]]=-base_speed_limits,base_speed_limits
        # The octagon is inscribed in the configured tilt-norm bound. Native
        # stops limit each axis, which alone cannot bound their combined norm.
        low[[4,6]],high[[4,6]]=-self._tilt_axis_limit,self._tilt_axis_limit
        if recovering:
            low[[4,6]]=np.maximum(low[[4,6]],x[[4,6]]-c.recovery_tilt_radius)
            high[[4,6]]=np.minimum(high[[4,6]],x[[4,6]]+c.recovery_tilt_radius)
        low[8:14]=np.maximum(self.q_min,self.q_min-self._feedforward)
        high[8:14]=np.minimum(self.q_max,self.q_max-self._feedforward)
        arm_speed = min(c.arm_speed_max*speed, c.sensor_work_arm_speed) if cooperative else c.arm_speed_max*speed
        low[14:20],high[14:20]=-arm_speed,arm_speed
        # Current/strain sensors do not provide a true friction/support label.
        support=np.ones(6) if state.get('sensor_only_navigation', False) else np.asarray(state.get('measured_contact',np.ones(6)),dtype=float)
        if support.shape!=(6,) or not np.isfinite(support).all():raise ValueError('Invalid measured_contact')
        force_limits=np.minimum(c.nominal_friction*c.normal_force*np.clip(support,0,1),self.wheel_torque_limit/c.wheel_radius)
        force_selector=np.zeros((2,8));force_selector[:,:2]=np.eye(2)
        velocity_selector=np.zeros((4,20));velocity_selector[:,[1,3,5,7]]=np.eye(4)
        maps=[];offsets=[]
        control_weight=np.diag([.003,.04]+[.08]*6)
        support_target=np.r_[terms[1][:2],np.zeros(6)]
        for step in range(n):
            selector=np.zeros((8,nz));selector[:,step*8:(step+1)*8]=np.eye(8)
            nominal=ad@nominal_state+bd@nominal_inputs[step]+cd
            local_a,local_b,local_c=self._passive_projection(ad,bd,cd,nominal)
            nominal_state=local_a@nominal_state+local_b@nominal_inputs[step]+local_c
            previous_s,previous_v=s.copy(),v.copy()
            s=local_a@s+local_b@selector;v=local_a@v+local_c
            offsets.append(v.copy());maps.append(s.copy())
            factor=3. if step==n-1 else 1.
            hessian+=2*factor*s.T@objective@s
            linear+=2*factor*s.T@(weights*(v-desired))
            if track_pose:
                task_error=np.r_[position-ee,orientation_error]+pose_map@(v-x)
                linear+=2*factor*s.T@pose_map.T@pose_weights@task_error
            hessian+=2*selector.T@control_weight@selector
            linear-=2*selector.T@control_weight@support_target
            constraints.append(s);lower.append(low-v);upper.append(high-v)
            tilt_offset=self._coupled_tilt_map@v
            constraints.append(self._coupled_tilt_map@s)
            lower.append(-self._coupled_tilt_limit-tilt_offset);upper.append(self._coupled_tilt_limit-tilt_offset)
            input_map=force_selector@selector
            rates_map=velocity_selector@previous_s;rates_offset=velocity_selector@previous_v
            f_map=self.rolling_force_input@input_map+self.rolling_force_velocity@rates_map
            f_offset=self.rolling_force_velocity@rates_offset+self.rolling_force_bias
            constraints.append(f_map);lower.append(-force_limits-f_offset);upper.append(force_limits-f_offset)
            w_map=self.command_force_map@input_map+self.command_velocity_map@rates_map
            w_offset=self.command_velocity_map@rates_offset+self.command_bias
            constraints.append(w_map);lower.append(-c.wheel_max-w_offset);upper.append(c.wheel_max-w_offset)
        hessian=.5*(hessian+hessian.T)
        variable_low=np.tile([-c.normal_force*6,-c.normal_force*6*c.tree_radius]+[-arm_speed]*6,n)
        if stage in ('navigate','done') or recovering:
            # Stowing before arrival is a task permission, not a soft posture
            # preference: the optimizer may not use arm motion as a base input.
            variable_low.reshape(n,8)[:,2:]=0.
        variable_high = -variable_low
        if cooperative:
            qerror = qref-x[8:14]
            arm_low = np.where(qerror > .008, 0., -arm_speed)
            arm_high = np.where(qerror < -.008, 0., arm_speed)
            at_reference = np.abs(qerror) <= .008
            arm_low[at_reference] = arm_high[at_reference] = 0.
            if self._work_phase in ('base_reposition', 'unreachable_hold') or stage == 'done':
                arm_low[:] = arm_high[:] = 0.
            if self._work_phase == 'folded_shoulder_yaw':
                arm_low[1:] = arm_high[1:] = 0.
            variable_low.reshape(n, 8)[:, 2:] = arm_low
            variable_high.reshape(n, 8)[:, 2:] = arm_high
        constraint_matrix=np.vstack(constraints);lo=np.concatenate(lower);hi=np.concatenate(upper)
        result=self._solver(h=hessian,g=linear,a=constraint_matrix,lba=lo,uba=hi,lbx=variable_low,ubx=variable_high,x0=self._last_solution)
        solution=np.asarray(result['x']).reshape(-1)
        value=constraint_matrix@solution
        violation=max(0.,float(np.max(lo-value)),float(np.max(value-hi)),float(np.max(variable_low-solution)),float(np.max(solution-variable_high)))
        if not self._solver.stats().get('success') or not np.isfinite(solution).all() or violation>5e-5:
            raise RuntimeError(f'QP unsuccessful or violation {violation}')
        self._base_force=solution[:2].copy();self._arm_velocity=solution[2:8].copy()
        self._last_solution=np.r_[solution[8:],solution[-8:]]
        predictions=np.stack([offset+mapping@solution for offset,mapping in zip(offsets,maps)])
        self._solve_count+=1
        self._diagnostics={'solver':'casadi_daqp','solve_success':True,'fallback':False,'horizon':n,'state_dimension':20,
            'decision_dimension_per_step':8,'max_constraint_violation':violation,'solve_ms':(perf_counter()-started)*1000,
            'max_predicted_tilt_norm':float(np.max(np.linalg.norm(predictions[:,[4,6]],axis=1))),
            'predicted_tilt_norm_bound':c.max_tilt,'tilt_constraint_geometry':'inscribed_octagon',
            'passive_tilt_model':'native gravity/inertia/dry friction/damping with local mechanical-stop projection',
            'active_tilt_generalized_force':[0.,0.],'base_generalized_force':self._base_force.tolist(),
            'independent_tilt_control':[0.,0.],
            'nominal_arm_base_mass_coupling_norm':float(np.linalg.norm(terms[0][:4,4:])),
            'reference_height':height,'reference_azimuth':theta,'reference_ee':ee.tolist(),'reference_speed':speed,
            'base_posture_weights':weights[[0,2]].tolist(),
            'planned_joint_reference':qref.tolist(),'base_target':base_target.tolist(),'no_zero_tilt_objective':True,
            'ik_weighted_pose_error':ik_error,'arm_velocity_time_constant':c.arm_velocity_time_constant,
            'solve_count':self._solve_count,'planning_route_found':bool(found)}
        self._diagnostics.update(base_speed_limits=base_speed_limits.tolist(),
            max_predicted_base_speeds=np.max(np.abs(predictions[:,[1,3]]),axis=0).tolist(),
            base_velocity_limits_are_physical_constraints=True)
        self._last_predictions=predictions

    def _base_speed_limits(self,stage):
        limits=np.array([self.config.max_base_speed,self.config.max_azimuth_speed])
        if stage!='navigate':
            limits=np.minimum(limits,[self.config.max_work_base_speed,self.config.max_work_azimuth_speed])
        return limits

    def _operation_target(self,state,reference):
        """A soft measured alignment anchor, with policy base-reference offsets.

        The task pose cost can still move h/theta for a moving target. This
        avoids continuously pulling an aligned base back toward its earlier
        arrival pose through the redundant arm/base nullspace.
        """
        offsets=np.array([reference[0]-float(state['ref_height']),
                          float(wrap_angle(reference[1]-float(state['ref_azimuth'])))])
        return self._operation_base_pose+offsets

    def _sequential(self,state,x,reference,terms,planning,fallback=False):
        height,theta,ee,rotation,speed=reference
        waypoint=planning[2];stage=state['stage'];home=np.asarray(state['arm_home'])
        cooperative = self.config.sensor_work_safety_enabled and stage != 'navigate'
        if cooperative:
            base_target, qref = self._cooperative_work_reference(state, x, reference)
        elif stage=='navigate':base_target=waypoint;qref=home
        elif stage=='extend':base_target=self._arrival_pose.copy();qref=self._arrival_ik.copy()
        elif stage=='operate':
            base_target=self._operation_target(state,reference)
            qref=home if fallback else self._ik(ee,rotation,x,x[8:14])[0]
        elif stage=='realign' and not fallback:
            position,actual,jp,jr=self._kinematics()
            error=np.r_[ee-position,.12*Rotation.from_matrix(rotation@actual.T).as_rotvec()]
            jac=np.vstack((jp[:,:2],.12*jr[:,:2]))
            # Measured-position repositioning; never a roll/pitch torque request.
            base_increment=jac.T@np.linalg.solve(jac@jac.T+.02*np.eye(6),error)
            base_target=x[[0,2]]+np.clip(base_increment,[-.04,-.15],[.04,.15])
            qref,_=self._ik(ee,rotation,x,x[8:14])
        else:base_target=x[[0,2]];qref=home
        if self.planner.recovery_active:
            base_target=waypoint
            qref=home if stage in ('navigate','done') else x[8:14].copy()
        corridor=planning[3]
        base_target=np.clip(base_target,[corridor[0],corridor[2]],[corridor[1],corridor[3]])
        base_speed_limits=self._base_speed_limits(stage)
        desired_velocity=np.clip(1.25*(base_target-x[[0,2]]),-base_speed_limits,base_speed_limits)
        desired_acceleration=np.clip(4*(desired_velocity-x[[1,3]]),[-1.,-2.],[1.,2.])
        mass,bias,_=terms
        self._base_force=mass[:2,:2]@desired_acceleration+bias[:2]
        self._arm_velocity=np.clip(2*(qref-x[8:14]),-self.config.arm_speed_max*speed,self.config.arm_speed_max*speed)
        if stage in ('navigate','done') or self.planner.recovery_active:self._arm_velocity[:]=0.
        if fallback:self._arm_velocity=np.clip(self._arm_velocity,-.15,.15)
        if cooperative and self._work_phase in ('base_reposition', 'unreachable_hold'):
            self._arm_velocity[:] = 0.
        if cooperative and self._work_phase == 'folded_shoulder_yaw':
            self._arm_velocity[1:] = 0.
        self._diagnostics={'solver':'rank_two_measured_feedback','solve_success':False,'fallback':fallback,
            'optimization_applicable':False,'active_tilt_generalized_force':[0.,0.],'no_zero_tilt_objective':True,
            'base_generalized_force':self._base_force.tolist(),'base_target':base_target.tolist(),
            'planned_joint_reference':qref.tolist(),'reference_height':height,'reference_azimuth':theta,
            'reference_ee':ee.tolist(),'reference_speed':speed,'planning_route_found':bool(planning[1])}
        self._diagnostics.update(base_speed_limits=base_speed_limits.tolist(),
            base_velocity_reference=desired_velocity.tolist(),base_velocity_limits_are_physical_constraints=True)

    def command(self,state,reference_action=None,model_action=None):
        x=self._state_vector(state);dt=float(state['dt']);stage=state['stage']
        measured_state=state;measured_x=x.copy();control_step=self._step
        delay_steps=self._delay_steps(state,dt);capture_step=self._step
        action=np.zeros(4) if model_action is None else np.array(model_action,dtype=float,copy=True)
        if action.shape!=(4,) or not np.isfinite(action).all():raise ValueError('model_action must have four finite entries')
        if self._q_command is None:
            self._q_command=x[8:14].copy()
            # Reset's issued position target is public arm_home. The noisy
            # first encoder sample is not the preceding actuator command.
            self._last_arm_command=np.asarray(state['arm_home']).copy() if delay_steps else x[8:14].copy()
        previous_arm_command=self._last_arm_command.copy()
        innovation=np.zeros(4)
        if delay_steps:
            state,x,innovation,capture_step=self._compensate_delay(state,x,delay_steps,dt)
        elif self._previous_predicted is not None:
            innovation=(x[[1,3,5,7]]-self._previous_predicted[[1,3,5,7]])/dt
        if self.mode=='adaptive_mpc' or (self.mode=='residual_mpc' and self.config.residual_use_adaptation):
            limits=np.asarray(self.config.model_residual_scales)
            self._disturbance=np.clip(self._disturbance+self.config.adaptation_rate*np.clip(innovation,-2*limits,2*limits),-limits,limits)
        requested=np.clip(action,-1,1)*np.asarray(self.config.model_residual_scales)*self.config.residual_model_scale if self.mode=='residual_mpc' else np.zeros(4)
        reference=self._refs(state,reference_action,x)
        terms=self._model_terms(x,state)
        if stage=='extend' and self._last_stage!='extend':
            self._arrival_pose=x[[0,2]].copy()
            self._arrival_ik,_=self._ik(reference[2],reference[3],x,x[8:14])
            if self.config.sensor_work_safety_enabled:
                self._initialize_cooperative_work(state, x, reference)
        if self._arrival_pose is None:self._arrival_pose=x[[0,2]].copy()
        if self._arrival_ik is None:self._arrival_ik=x[8:14].copy()
        if stage=='operate' and self._last_stage!='operate':self._operation_base_pose=x[[0,2]].copy()
        if self.config.sensor_work_safety_enabled and stage=='retract' and self._last_stage!='retract':
            self._work_retract_yaw = float(x[8])
            self._work_phase = 'retract_pitch_axes'
        should_solve=self._step%self.config.solve_interval==0 or stage!=self._last_stage
        if should_solve:
            self._learned=requested.copy();self._held_reference=reference
            planning=self._plan(state,x,reference);self._last_planning=planning
            try:
                if not planning[1]:raise RuntimeError('No safe six-wheel terrain route')
                if self.mode=='sequential':self._sequential(state,x,reference,terms,planning)
                else:self._solve(state,x,reference,terms,planning)
            except (RuntimeError,np.linalg.LinAlgError) as error:
                self._fallback_count+=1;self._sequential(state,x,reference,terms,planning,True)
                self._diagnostics.update(solver_status=str(error),fallback_count=self._fallback_count)
        else:planning=self._last_planning
        if self.config.sensor_work_safety_enabled and stage != 'navigate':
            self._arm_velocity = self._cooperative_arm_velocity(self._arm_velocity, state, x, dt)
        # Safety commands also retain exactly the rank-two manifold. Numerical
        # saturation uses a common scalar, never independent wheel clipping.
        raw=self.command_force_map@self._base_force+self.command_velocity_map@x[[1,3,5,7]]+self.command_bias
        speeds=np.linalg.pinv(self.mixer)@raw
        wheel=wheel_commands(speeds,self.config.wheel_max,self.config.tree_radius,self.config.wheel_radius)
        torque=self.torque_force_map@self._base_force
        torque_scale=min(1.,float(np.min(self.wheel_torque_limit/np.maximum(np.abs(torque),1e-12))))
        torque*=torque_scale
        requested_base_force=self._base_force.copy()
        self._base_force*=torque_scale
        if self.planner.recovery_active and not self._recovery_was_active and stage not in ('navigate','done'):
            self._q_command=x[8:14].copy()
        self._q_command=np.clip(self._q_command,x[8:14]-.06,x[8:14]+.06)
        self._q_command=np.clip(self._q_command+dt*self._arm_velocity,self.q_min,self.q_max)
        if stage in ('navigate','done'):self._q_command=np.clip(state['arm_home'],self.q_min,self.q_max).copy()
        arm=np.clip(self._q_command+self._feedforward,self.q_min,self.q_max)
        step_limit=dt*self.config.arm_speed_max*self._held_reference[4]
        cooperative = self.config.sensor_work_safety_enabled and stage != 'navigate'
        if cooperative:
            step_limit = min(step_limit, dt*self.config.sensor_work_arm_speed)
        arm=np.clip(arm,self._last_arm_command-step_limit,self._last_arm_command+step_limit)
        if cooperative:
            issued = (arm-self._last_arm_command)/dt
            change = self.config.sensor_work_arm_acceleration*dt
            issued = np.clip(issued, self._issued_arm_velocity-change, self._issued_arm_velocity+change)
            arm = np.clip(self._last_arm_command+dt*issued, self.q_min, self.q_max)
            self._issued_arm_velocity = (arm-self._last_arm_command)/dt
        else:
            self._issued_arm_velocity = (arm-self._last_arm_command)/dt
        self._last_arm_command=arm.copy()
        self._recovery_was_active=bool(self.planner.recovery_active)
        ad,bd,cd=self._linear_model(x,terms,dt)
        # The plant receives the final position-command ramp, after feedforward,
        # clipping and per-frame slew limits. Its derivative is the input to
        # our nominal local velocity-lag approximation, not a claim that
        # the optimizer's requested velocity was physically executed.
        issued_arm_velocity=(arm-previous_arm_command)/dt
        prediction_input=np.r_[self._base_force,issued_arm_velocity if delay_steps else self._arm_velocity]
        nominal=ad@x+bd@prediction_input+cd
        ad,bd,cd=self._passive_projection(ad,bd,cd,nominal)
        self._previous_predicted=ad@x+bd@prediction_input+cd
        if delay_steps:
            self._command_history.append(dict(step=self._step,dt=dt,model_correction=self._correction().copy(),
                input=prediction_input.copy(),wheel_torque=torque.copy(),applied_base_force=self._base_force.copy(),
                arm_command_start=previous_arm_command,arm_command=arm.copy()))
            while self._command_history and self._command_history[0]['step']<capture_step:
                self._command_history.popleft()
        self._last_stage=stage;self._step+=1
        measured=np.asarray(state.get('cut_pos',self._scratch.site_xpos[self.site_id]))
        diag=dict(self._diagnostics)
        if self.config.sensor_work_safety_enabled:
            diag.update(self._work_plan_diagnostics)
            diag.update(cooperative_work_safety_enabled=True,
                cooperative_work_phase=self._work_phase,
                cooperative_extension_ready=bool(self._work_phase == 'ready'),
                cooperative_arm_command_velocity=self._issued_arm_velocity.tolist(),
                cooperative_arm_command_speed_limit=self.config.sensor_work_arm_speed,
                cooperative_arm_command_acceleration_limit=self.config.sensor_work_arm_acceleration,
                cooperative_arm_reference_policy='folded shoulder yaw, extend; reverse on withdrawal',
                cooperative_work_target_is_measured_robot_model=True)
        diag.update(mode=self.mode,optimized_this_step=bool(should_solve and self.mode!='sequential'),
            estimated_external_acceleration=self._disturbance.tolist(),learned_model_acceleration=self._learned.tolist(),
            combined_external_acceleration=self._correction().tolist(),measurement_acceleration_innovation=innovation.tolist(),
            wheel_manifold_error=float(np.linalg.norm(wheel-self.mixer@np.linalg.lstsq(self.mixer,wheel,rcond=None)[0])),
            wheel_command_base_speeds=(np.linalg.pinv(self.mixer)@wheel).tolist(),
            planner_path=planning[0].tolist(),planner_corridor=planning[3].tolist(),route_found=bool(planning[1]),
            navigation_phase=self.planner.navigation_phase,
            navigation_approach_first=self.config.navigation_approach_first,
            navigation_obstacle=self.planner.navigation_obstacle,
            navigation_pause_verified=bool(self.planner.navigation_pause_verified),
            navigation_pause_state=self.planner.navigation_pause_state,
            navigation_geometry_triggered=True,
            planner_recovery_active=bool(self.planner.recovery_active),
            planner_recovery_reason=self.planner.recovery_reason,
            planner_recovery_target=None if self.planner._escape_goal is None else self.planner._escape_goal.tolist(),
            recovery_tilt_radius=self.config.recovery_tilt_radius if self.planner.recovery_active else None,
            recovery_tilt_reference=x[[4,6]].tolist() if self.planner.recovery_active else None,
            recovery_sensor_noise_sigma=float(state.get('sensor_noise_sigma',0.)),
            recovery_sensor_delay_seconds=float(state.get('sensor_delay_seconds',0.)),
            recovery_keeps_physical_footprint_margins=True,
            arrival_ik_q=self._arrival_ik.tolist(),arrival_base_pose=self._arrival_pose.tolist(),
            realignment_error=float(np.linalg.norm(np.asarray(measured_state.get('cut_pos',measured))-
                np.asarray(measured_state['ref_ee']))),
            adjusted_reference_error=float(np.linalg.norm(measured-self._held_reference[2])),
            base_repositioning=(x[[0,2]]-self._arrival_pose).tolist(),measured_tilt=measured_x[[4,6]].tolist())
        diag['native_passive_stop_active']=self._stop_active.tolist()
        diag['operation_base_anchor']=None if self._operation_base_pose is None else self._operation_base_pose.tolist()
        diag['operation_anchor_is_soft_reference']=True
        diag['operation_base_reference_offset']=([float(self._held_reference[0])-float(state['ref_height']),
            float(wrap_angle(self._held_reference[1]-float(state['ref_azimuth'])))] if stage=='operate' else [0.,0.])
        diag['traction_induced_tilt_torque']=(self._actuation_map@self._base_force+
            self._traction_velocity_force@x[[1,3,5,7]]+self._traction_bias_force)[2:].tolist()
        diag['wheel_nominal_holding_torque']=self._holding_torque.tolist()
        diag['wheel_torque_command']=torque.tolist()
        diag['wheel_torque_common_saturation_scale']=torque_scale
        diag['requested_base_generalized_force']=requested_base_force.tolist()
        diag['applied_base_generalized_force']=self._base_force.tolist()
        diag['base_generalized_force']=self._base_force.tolist()
        diag['traction_inverse']='rank_two_native_motor_torque_allocation_with_no_slip_inertia_projection'
        diag['rolling_reflected_base_mass_matrix']=terms[0][:4,:4].tolist()
        diag['nominal_mass_approximation']='native rolling constraint mass/damping; pose-linearized gravity/bias, fixed nominal scissors pose, no explicit brush state or constraint Tdot in MPC'
        diag.update(sensor_delay_compensated=bool(delay_steps),controller_time=control_step*dt,
            sensor_measurement_time=capture_step*dt,sensor_sample_age=(control_step-capture_step)*dt,
            delay_prediction_is_nominal_approximation=True,
            delay_innovation_uses_issued_input_history=bool(delay_steps),
            delay_raw_measured_state=measured_x.tolist(),delay_compensated_state=x.tolist(),
            predicted_current_tilt=x[[4,6]].tolist(),
            predicted_current_realignment_error=float(np.linalg.norm(measured-np.asarray(state['ref_ee']))),
            adjusted_reference_error_is_delay_compensated=bool(delay_steps),
            delay_arm_input_model='issued clipped position-command ramp derivative with nominal velocity lag',
            delay_issued_arm_reference_velocity=issued_arm_velocity.tolist(),
            raw_delayed_realignment_error=float(np.linalg.norm(np.asarray(measured_state.get('cut_pos',measured_state['ref_ee']))-
                np.asarray(measured_state['ref_ee']))))
        if delay_steps:
            observer=getattr(self,'_observer_last_stiction',None)
            diag['delay_observer_static_axes']=None if observer is None else observer['static_axes'].tolist()
            diag['delay_observer_stop_supported_axes']=None if observer is None else observer['stop_supported_axes'].tolist()
            diag['delay_observer_required_resisting_friction']=None if observer is None else observer['required_resisting_friction'].tolist()
            diag['delay_observer_friction_limits']=None if observer is None else observer['friction_limits'].tolist()
            diag['delay_observer_model']='fresh nominal pose dynamics with issued torque/ramp and input-aware passive stick/slip'
        return {'wheel_command':wheel,'wheel_torque_command':torque,'arm_command':arm,'diagnostics':diag}
