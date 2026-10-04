"""Failure evidence from measured height, wheel slip and motor-load signals."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np


SENSOR_FIELDS = (
    'base_height', 'base_azimuth', 'base_rates', 'base_tilt', 'tilt_rates',
    'wheel_rates', 'wheel_motor_current', 'wheel_shaft_acceleration', 'slip',
    'measured_contact', 'sensor_capture_time', 'capture_time', 'time',
    'sensor_delay_seconds', 'dt',
)
FEATURE_LAYOUT = (
    ('goal_error', 2), ('base_height', 1), ('azimuth_sin_cos', 2),
    ('base_rates', 2), ('base_tilt', 2), ('tilt_rates', 2),
    ('wheel_rates', 6), ('wheel_motor_current', 6),
    ('wheel_shaft_acceleration', 6), ('slip', 6), ('measured_contact', 6),
    ('requested_velocity', 2), ('observed_progress_velocity', 2),
    ('progress_ratio', 1), ('slip_evidence', 1), ('current_deviation', 1),
    ('stall_confidence', 1), ('new_evidence_event', 1), ('sample_age', 1),
    ('previous_action', 2), ('original_azimuth_error', 1),
    ('requested_physical_direction', 2),
)
SCALAR_FEATURE_SIZE = sum(size for _, size in FEATURE_LAYOUT)


def _wrap(value):
    return (np.asarray(value, dtype=float)+np.pi) % (2*np.pi)-np.pi


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return result.copy()


@dataclass(frozen=True)
class OnlineNavigationConfig:
    tree_radius: float = .15
    wheel_radius: float = .032
    height_limits: tuple[float, float] = (.30, 1.90)
    height_action_span: float = .06
    azimuth_action_span: float = .16
    height_reference_margin: float = 1e-5
    command_response_time: float = .32
    max_expected_height_speed: float = .20
    max_expected_azimuth_speed: float = .65
    evidence_window_seconds: float = .32
    minimum_window_seconds: float = .24
    minimum_uncorroborated_window_seconds: float = .32
    uncorroborated_motion_confidence: float = .35
    minimum_requested_travel: float = .004
    maximum_stalled_travel: float = .0015
    stalled_progress_ratio: float = .25
    direction_coherence: float = .70
    slip_threshold: float = .010
    current_deviation_threshold: float = .08
    current_baseline_rate: float = .05
    height_cell_size: float = .02
    azimuth_cell_size: float = .08
    grid_height_cells: int = 7
    grid_azimuth_cells: int = 9
    max_memory_cells: int = 2048

    def __post_init__(self):
        positive = ('tree_radius', 'wheel_radius', 'height_action_span',
                    'azimuth_action_span', 'command_response_time',
                    'height_reference_margin',
                    'max_expected_height_speed', 'max_expected_azimuth_speed',
                    'evidence_window_seconds', 'minimum_window_seconds',
                    'minimum_uncorroborated_window_seconds',
                    'minimum_requested_travel', 'maximum_stalled_travel',
                    'slip_threshold', 'current_deviation_threshold',
                    'height_cell_size', 'azimuth_cell_size')
        if any(not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0 for key in positive):
            raise ValueError('Navigation lengths, times and signal thresholds must be positive')
        limits = _vector(self.height_limits, 2, 'height_limits')
        if limits[0] >= limits[1]:
            raise ValueError('height_limits must increase')
        if 2*self.height_reference_margin >= limits[1]-limits[0]:
            raise ValueError('Height reference margin must leave a nonempty interval')
        if self.minimum_window_seconds > self.evidence_window_seconds:
            raise ValueError('Minimum evidence duration exceeds the history window')
        if not self.minimum_window_seconds <= self.minimum_uncorroborated_window_seconds <= self.evidence_window_seconds:
            raise ValueError('Uncorroborated evidence needs a longer interval within the history window')
        for key in ('direction_coherence', 'stalled_progress_ratio', 'current_baseline_rate',
                    'uncorroborated_motion_confidence'):
            if not 0 < getattr(self, key) <= 1:
                raise ValueError(f'{key} must lie in (0, 1]')
        if self.uncorroborated_motion_confidence >= self.minimum_window_seconds/self.evidence_window_seconds:
            raise ValueError('Uncorroborated evidence must have lower confidence than corroborated evidence')
        if any(not isinstance(getattr(self, key), int) or getattr(self, key) < 1
               for key in ('grid_height_cells', 'grid_azimuth_cells', 'max_memory_cells')):
            raise ValueError('Navigation memory dimensions must be positive integers')
        if self.grid_height_cells % 2 != 1 or self.grid_azimuth_cells % 2 != 1:
            raise ValueError('Ego grids need odd dimensions')


class OnlineNavigationMemory:
    """Sensor-only features and bounded, causal robot-configuration memory.

    ``update`` is called after executing a command. A mapping command carries
    ``issued_time`` plus ``waypoint`` and/or ``velocity_reference``. A bare
    two-vector denotes a velocity reference [m/s, rad/s]. Issued commands are
    queued even while delayed initial sensor samples repeat. Progress is paired
    with inputs issued during the sensor interval, never the newest command.
    Detection is evidence for the actor; it never replaces actor decisions.
    """
    def __init__(self, config: OnlineNavigationConfig | Mapping | None = None):
        self.config = (config if isinstance(config, OnlineNavigationConfig)
                       else OnlineNavigationConfig(**(dict(config) if config else {})))
        self.observation_size = SCALAR_FEATURE_SIZE+2*self.config.grid_height_cells*self.config.grid_azimuth_cells
        self._initialized = False

    @staticmethod
    def sensor_view(state):
        """Explicit allowlist; hidden map/evaluator fields are never accessed."""
        return {key: state[key] for key in SENSOR_FIELDS if key in state}

    def _frame(self, state):
        state = self.sensor_view(state)
        pose = np.array([float(state['base_height']), float(state['base_azimuth'])])
        if not np.isfinite(pose).all():
            raise ValueError('Base odometry must be finite')
        now = float(state.get('time', state.get('sensor_capture_time', state.get('capture_time', 0.))))
        capture = float(state.get('sensor_capture_time', state.get('capture_time',
                              max(0., now-float(state.get('sensor_delay_seconds', 0.))))))
        if not np.isfinite([now, capture]).all() or capture < 0 or capture > now+1e-9:
            raise ValueError('Sensor capture time must be finite and not in the future')
        result = dict(pose=pose, time=now, capture=capture)
        for key, size in [('base_rates', 2), ('base_tilt', 2), ('tilt_rates', 2),
                          ('wheel_rates', 6), ('wheel_motor_current', 6),
                          ('wheel_shaft_acceleration', 6), ('measured_contact', 6)]:
            result[key] = _vector(state.get(key, np.zeros(size)), size, key)
        if 'slip' in state:
            result['slip'] = _vector(state['slip'], 6, 'slip')
        else:
            signs = np.where(np.arange(6) % 2 == 0, 1., -1.)
            # Encoder/odometry approximation used only for detection features.
            # Passive tilt, roller compliance and load are not oracle inputs.
            expected = np.sqrt(.5)*(result['base_rates'][0]
                       + signs*self.config.tree_radius*result['base_rates'][1])
            result['slip'] = self.config.wheel_radius*result['wheel_rates']-expected
        return result

    def reset(self, state, goal):
        self._frame_state = self._frame(state)
        self._goal = _vector(goal, 2, 'goal')
        self._origin_azimuth = float(self._frame_state['pose'][1])
        self._commands = deque()
        self._intervals = deque()
        self._visited = {}
        self._risk = {}
        self._current_baseline = np.abs(self._frame_state['wheel_motor_current']).copy()
        self._last_action = np.zeros(2)
        self._requested_velocity = np.zeros(2)
        self._progress_velocity = np.zeros(2)
        self._physical_direction = np.zeros(2)
        self._progress_ratio = 1.
        self._current_deviation = self._slip_evidence = self._confidence = 0.
        self._evidence_basis = 'none'
        self._new_event = False
        self._samples = 1
        self._duplicate_samples = 0
        self._initialized = True
        self._remember_visited(self._frame_state['pose'], self._frame_state['capture'])
        return self.observation()

    def _cell_key(self, pose):
        c = self.config
        return (int(np.floor(pose[0]/c.height_cell_size)),
                int(np.floor((float(_wrap(pose[1]))+np.pi)/c.azimuth_cell_size)))

    def _remember_visited(self, pose, capture):
        self._visited[self._cell_key(pose)] = (pose.copy(), float(capture))
        self._trim(self._visited)

    def _trim(self, collection):
        while len(collection) > self.config.max_memory_cells:
            del collection[next(iter(collection))]

    def _record_command(self, command, state, dt):
        if command is None:
            return
        if isinstance(command, Mapping):
            issued = float(command.get('issued_time', state['time']-dt))
            velocity = command.get('velocity_reference')
            waypoint = command.get('waypoint')
            action = command.get('action')
        else:
            issued = state['time']-dt
            velocity, waypoint, action = command, None, None
        if not np.isfinite(issued) or issued < -1e-9 or issued > state['time']+1e-9:
            raise ValueError('Command issue time must be finite and no later than the current clock')
        if velocity is None and waypoint is None:
            raise ValueError('Command needs a waypoint or velocity_reference')
        record = dict(time=max(0., issued), velocity=None if velocity is None else _vector(velocity, 2, 'velocity_reference'),
                      waypoint=None if waypoint is None else _vector(waypoint, 2, 'waypoint'))
        if self._commands and issued < self._commands[-1]['time']-1e-9:
            raise ValueError('Issued-command history must be chronological')
        if self._commands and abs(issued-self._commands[-1]['time']) < 1e-9:
            self._commands[-1] = record
        else:
            self._commands.append(record)
        if action is not None:
            self._last_action = np.clip(_vector(action, 2, 'action'), -1., 1.)

    def _velocity(self, command, pose):
        if command is None:
            return np.zeros(2)
        velocity = command['velocity']
        if velocity is None:
            delta = command['waypoint']-pose
            delta[1] = float(_wrap(delta[1]))
            velocity = delta/self.config.command_response_time
        return np.clip(velocity,
                       -np.array([self.config.max_expected_height_speed, self.config.max_expected_azimuth_speed]),
                       np.array([self.config.max_expected_height_speed, self.config.max_expected_azimuth_speed]))

    def _expected_delta(self, start, end, pose):
        records = list(self._commands)
        active = None
        for item in records:
            if item['time'] <= start+1e-9:
                active = item
            else:
                break
        boundaries = [item for item in records if start < item['time'] < end]
        total, previous = np.zeros(2), start
        for item in boundaries:
            total += (item['time']-previous)*self._velocity(active, pose)
            active, previous = item, item['time']
        return total+(end-previous)*self._velocity(active, pose)

    def update(self, state, previous_command=None, dt=None):
        if not self._initialized:
            raise RuntimeError('reset must precede sensor memory updates')
        current = self._frame(state)
        dt = float(dt if dt is not None else state.get('dt', .04))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        self._record_command(previous_command, current, dt)
        self._new_event = False
        old = self._frame_state
        if current['capture'] < old['capture']-1e-9:
            raise ValueError('Sensor sample times must be chronological')
        if abs(current['capture']-old['capture']) < 1e-9:
            self._duplicate_samples += 1
            # Current clock is diagnostic; repeated delayed samples do not
            # create fictitious progress, confidence or memory observations.
            self._frame_state = {**old, 'time': current['time']}
            return self.diagnostics()
        delta = current['pose']-old['pose']
        delta[1] = float(_wrap(delta[1]))
        expected = self._expected_delta(old['capture'], current['capture'], old['pose'])
        current_deviation = float(np.median(np.abs(current['wheel_motor_current'])-self._current_baseline))
        self._intervals.append(dict(start=old['capture'], end=current['capture'], expected=expected,
                                   # A single unloaded/spinning wheel is meaningful:
                                   # the five working wheels must not erase its
                                   # evidence. The subsequent history window
                                   # still filters this per-frame encoder signal.
                                   actual=delta, slip=float(np.max(np.abs(current['slip']))),
                                   current=max(0., current_deviation)))
        cutoff = current['capture']-self.config.evidence_window_seconds
        while self._intervals and self._intervals[0]['end'] <= cutoff+1e-9:
            self._intervals.popleft()
        while len(self._commands) > 1 and self._commands[1]['time'] <= self._intervals[0]['start']:
            self._commands.popleft()
        self._frame_state = current
        self._samples += 1
        self._recompute_evidence()
        if self._confidence == 0.:
            self._remember_visited(current['pose'], current['capture'])
            rate = self.config.current_baseline_rate
            self._current_baseline += rate*(np.abs(current['wheel_motor_current'])-self._current_baseline)
        else:
            self._remember_risk()
        return self.diagnostics()

    def _recompute_evidence(self):
        c = self.config
        duration = self._intervals[-1]['end']-self._intervals[0]['start']
        expected = sum((item['expected'] for item in self._intervals), np.zeros(2))
        actual = sum((item['actual'] for item in self._intervals), np.zeros(2))
        self._requested_velocity = expected/max(duration, 1e-12)
        self._progress_velocity = actual/max(duration, 1e-12)
        scale = np.array([1., c.tree_radius])
        requested = expected*scale
        measured = actual*scale
        travel = float(np.linalg.norm(requested))
        self._physical_direction = requested/max(travel, 1e-12)
        achieved = float(measured@self._physical_direction)
        self._progress_ratio = achieved/max(travel, 1e-12) if travel > 1e-12 else 1.
        path_length = sum(float(np.linalg.norm(item['expected']*scale)) for item in self._intervals)
        coherence = travel/max(path_length, 1e-12)
        self._slip_evidence = float(np.mean([item['slip'] for item in self._intervals]))
        self._current_deviation = float(np.mean([item['current'] for item in self._intervals]))
        corroborated = self._slip_evidence >= c.slip_threshold or self._current_deviation >= c.current_deviation_threshold
        failed_progress = (travel >= c.minimum_requested_travel and coherence >= c.direction_coherence
                           and achieved <= c.maximum_stalled_travel
                           and self._progress_ratio < c.stalled_progress_ratio)
        strong = failed_progress and corroborated and duration+1e-9 >= c.minimum_window_seconds
        weak = failed_progress and duration+1e-9 >= c.minimum_uncorroborated_window_seconds
        # Lack of progress alone does not classify a surface obstruction. It
        # records weaker observed motion failure after a longer command window;
        # wheel/current corroboration permits earlier, stronger evidence.
        self._confidence = (min(1., duration/c.evidence_window_seconds) if strong
                            else c.uncorroborated_motion_confidence if weak else 0.)
        self._evidence_basis = ('progress_with_slip_or_current' if strong
                                else 'sustained_progress_failure' if weak else 'none')

    def _remember_risk(self):
        pose, capture = self._frame_state['pose'], self._frame_state['capture']
        key = self._cell_key(pose)
        previous = self._risk.get(key)
        self._new_event = previous is None
        self._risk[key] = dict(height=float(pose[0]), azimuth=float(_wrap(pose[1])),
            height_radius=self.config.height_cell_size/2, azimuth_radius=self.config.azimuth_cell_size/2,
            confidence=max(self._confidence, previous['confidence'] if previous else 0.),
            last_capture_time=float(capture), observations=(previous['observations'] if previous else 0)+1,
            failed_motion_direction=self._physical_direction.tolist(), kind='observed_motion_failure')
        # A configuration with failed motion is not positively marked as free.
        self._visited.pop(key, None)
        self._trim(self._risk)

    def configuration_cells(self):
        """Return inferred base-pose evidence, never actual tree-patch geometry."""
        return [{**cell, 'failed_motion_direction':list(cell['failed_motion_direction'])}
                for cell in self._risk.values()]

    def _grids(self):
        c, pose = self.config, self._frame_state['pose']
        risk = np.zeros((c.grid_height_cells, c.grid_azimuth_cells), dtype=float)
        visited = np.zeros_like(risk)
        for collection, output in ((self._risk.values(), risk), (self._visited.values(), visited)):
            for cell in collection:
                if output is risk:
                    point = np.array([cell['height'], cell['azimuth']]);value = cell['confidence']
                else:
                    point, _ = cell;value = 1.
                delta = point-pose;delta[1] = float(_wrap(delta[1]))
                row = int(np.rint(delta[0]/c.height_cell_size))+c.grid_height_cells//2
                column = int(np.rint(delta[1]/c.azimuth_cell_size))+c.grid_azimuth_cells//2
                if 0 <= row < risk.shape[0] and 0 <= column < risk.shape[1]:
                    output[row, column] = max(output[row, column], value)
        return risk, visited

    def observation(self, goal=None, previous_action=None):
        if not self._initialized:
            raise RuntimeError('reset must precede navigation observation')
        if goal is not None:
            self._goal = _vector(goal, 2, 'goal')
        action = self._last_action if previous_action is None else _vector(previous_action, 2, 'previous_action')
        s, c = self._frame_state, self.config
        error = self._goal-s['pose'];error[1] = float(_wrap(error[1]))
        rates = np.array([c.max_expected_height_speed, c.max_expected_azimuth_speed])
        values = [error/np.array([c.height_limits[1]-c.height_limits[0], np.pi]),
            [(s['pose'][0]-c.height_limits[0])/(c.height_limits[1]-c.height_limits[0])],
            [np.sin(s['pose'][1]), np.cos(s['pose'][1])],
            s['base_rates']/rates, s['base_tilt']/.20, s['tilt_rates']/2.,
            s['wheel_rates']/30., s['wheel_motor_current'], s['wheel_shaft_acceleration']/300.,
            s['slip']/.15, s['measured_contact']/2., self._requested_velocity/rates,
            self._progress_velocity/rates, [self._progress_ratio/2.],
            [self._slip_evidence/.03], [self._current_deviation/.25], [self._confidence],
            [float(self._new_event)], [(s['time']-s['capture'])/.40], action,
            [float(_wrap(self._origin_azimuth-s['pose'][1]))/np.pi], self._physical_direction]
        risk, visited = self._grids()
        result = np.concatenate([np.asarray(value).ravel() for value in values]+[risk.ravel(), visited.ravel()])
        if result.shape != (self.observation_size,) or not np.isfinite(result).all():
            raise RuntimeError('Navigation observation layout is inconsistent or nonfinite')
        return np.clip(result, -4., 4.).astype(np.float32)

    def diagnostics(self):
        if not self._initialized:
            raise RuntimeError('reset must precede diagnostics')
        return dict(navigation_memory_source='causal_proprioception_only',
            navigation_decision_source='learned_policy_required',
            sensor_capture_time=self._frame_state['capture'], sensor_clock_time=self._frame_state['time'],
            unique_sensor_samples=self._samples, duplicate_sensor_samples=self._duplicate_samples,
            requested_velocity=self._requested_velocity.tolist(), observed_progress_velocity=self._progress_velocity.tolist(),
            motion_progress_ratio=float(self._progress_ratio), observed_slip=float(self._slip_evidence),
            observed_current_deviation=float(self._current_deviation),
            stall_confidence=float(self._confidence), new_contact_evidence_event=bool(self._new_event),
            motion_failure_evidence_basis=self._evidence_basis,
            new_motion_failure_event=bool(self._new_event),
            discovered_configuration_cells=self.configuration_cells(), visited_configuration_cell_count=len(self._visited),
            obstacle_extent_known=False, automatic_detour_action=False)


class BoundedNavigationCommandInterface:
    """Validate the actor's local reference; never invent an obstacle maneuver."""
    def __init__(self, config: OnlineNavigationConfig | None = None):
        self.config = config or OnlineNavigationConfig()

    def command(self, state, action):
        action = np.clip(_vector(action, 2, 'navigation_action'), -1., 1.)
        pose = _vector([state['base_height'], state['base_azimuth']], 2, 'base_pose')
        delta = action*np.array([self.config.height_action_span, self.config.azimuth_action_span])
        waypoint = pose+delta
        lo, hi = self.config.height_limits
        waypoint[0] = np.clip(waypoint[0], lo+self.config.height_reference_margin,
                             hi-self.config.height_reference_margin)
        waypoint[1] = pose[1]+float(_wrap(waypoint[1]-pose[1]))
        # This local configuration box is not certified free: its unobserved
        # interior must be explored by the policy and actual contact feedback.
        corridor = np.array([min(pose[0],waypoint[0]), max(pose[0],waypoint[0]),
                             min(pose[1],waypoint[1]), max(pose[1],waypoint[1])])
        return dict(waypoint=waypoint, action=action, velocity_reference=(waypoint-pose)/self.config.command_response_time,
                    configuration_corridor=corridor, unknown_corridor_certified_safe=False,
                    decision_source='learned_policy', automatic_detour_action=False)
