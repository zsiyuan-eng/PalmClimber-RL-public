"""Single-layer tree climber with passive tilt and simultaneous scissors/nozzle.

MuJoCo integrates all degrees of freedom. Traction, geometric cutting and spray
exposure are simulation models with uncalibrated engineering parameters."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, fields

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from envs.common import (
    ARM_JOINTS, BASE_JOINTS, STAGES, SCHEMA_VERSION, TREE_RADIUS, WHEEL_RADIUS, CONTACT_STIFFNESS,
    load_model, rolling_directions, wheel_commands, wheel_mixer, project_wheel_commands, wrap_angle,
)


@dataclass
class TreeWorkConfig:
    dt: float = .04
    max_episode_steps: int = 750
    history_length: int = 3
    observation_noise: float = .001
    observation_delay_steps: int = 0
    start_height_range: tuple = (.60, .68)
    goal_height_range: tuple = (1.0, 1.15)
    start_azimuth_range: tuple = (-.05, .05)
    goal_azimuth_range: tuple = (.55, .75)
    target_radius_range: tuple = (.57, .63)
    target_height_offset: float = .14
    friction_range: tuple = (.45, .85)
    payload_range: tuple = (0., .15)
    sway_amplitude: float = .005
    sway_frequency: float = .22
    normal_force_per_wheel: float = 45.
    traction_gain: float = 35.
    contact_stiffness: float = CONTACT_STIFFNESS
    wheel_speed_feedback_gain: float = .08
    wheel_speed_integral_gain: float = .12
    nominal_friction: float = .60
    tree_radius: float = TREE_RADIUS
    wheel_radius: float = WHEEL_RADIUS
    wheel_velocity_limit: float = 30.
    base_speed_limits: tuple = (.55, .65)
    work_base_speed_limits: tuple = (.06, .15)
    arm_home: tuple = (0., -1.70, 0., .50, 0., 0.)
    max_base_tilt: float = .205
    min_height: float = .30
    max_height: float = 1.9
    height_tolerance: float = .035
    azimuth_tolerance: float = .06
    arrival_hold_time: float = .20
    extend_min_time: float = .35
    extend_max_time: float = 2.0
    alignment_hold_time: float = .16
    realign_trigger_tolerance: float = .025
    tool_position_tolerance: float = .020
    tool_axis_tolerance: float = .28
    cut_hold_time: float = .12
    scissors_closure_speed: float = 2.5
    required_overlap_time: float = .12
    spray_standoff_range: tuple = (.055, .14)
    spray_reference_distance: float = .095
    spray_cone_half_angle: float = .55
    spray_dose_threshold: float = .08
    spray_required_coverage: float = .80
    spray_max_off_target_fraction: float = .55
    spray_flow_ml_per_second: float = 1.5
    terrain_enabled: bool = True
    sensor_only_navigation: bool = False
    sensor_work_safety_enabled: bool = False
    sensor_work_arm_speed: float = .24
    sensor_work_arm_acceleration: float = .65
    sensor_work_load_margin: float = .10
    sensor_work_azimuth_search: float = 1.0
    sensor_blocked_contact_stiffness: float = 6000.
    sensor_blocked_contact_damping: float = 60.
    sensor_blocked_contact_force_limit: float = 120.
    navigation_approach_first: bool = True
    terrain_footprint_margin: float = .035
    terrain_theta_margin: float = .08
    depression_support_fraction: float = .35
    horizon: int = 8
    prediction_dt: float = .12
    solve_interval: int = 3
    ee_position_weight: float = 1800.
    residual_height_scale: float = .06
    residual_azimuth_scale: float = .15
    residual_ee_scale: float = .025
    model_residual_scales: tuple = (1.5, 3., 8., 8.)
    residual_use_adaptation: bool = True
    residual_model_scale: float = .25
    render_width: int = 640
    render_height: int = 480

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            result = cls()
        elif isinstance(value, cls):
            result = cls(**asdict(value))
        elif isinstance(value, dict):
            unknown = set(value) - {f.name for f in fields(cls)}
            if unknown:
                raise ValueError(f"Unknown config fields: {sorted(unknown)}")
            result = cls(**value)
        else:
            raise TypeError("config must be TreeWorkConfig, dict, or None")
        integers = ("max_episode_steps", "history_length", "observation_delay_steps", "horizon",
                    "solve_interval", "render_width", "render_height")
        for key in integers:
            x = getattr(result, key)
            if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < (0 if key == "observation_delay_steps" else 1):
                raise ValueError(f"{key} must be a valid integer")
        for f in fields(cls):
            x = getattr(result, f.name)
            if isinstance(f.default, float) and (isinstance(x, (bool, np.bool_)) or not isinstance(x, (float, int, np.floating, np.integer))):
                raise ValueError(f"{f.name} must be a finite scalar number")
            if isinstance(x, (float, int, np.floating, np.integer)) and not np.isfinite(x):
                raise ValueError(f"{f.name} must be finite")
        for key in ("dt", "normal_force_per_wheel", "traction_gain", "contact_stiffness", "wheel_speed_feedback_gain", "wheel_speed_integral_gain", "nominal_friction", "wheel_velocity_limit",
                    "tree_radius", "wheel_radius", "height_tolerance", "azimuth_tolerance", "arrival_hold_time",
                    "extend_min_time", "extend_max_time", "alignment_hold_time", "realign_trigger_tolerance",
                    "tool_position_tolerance", "tool_axis_tolerance", "cut_hold_time", "scissors_closure_speed", "required_overlap_time",
                    "spray_reference_distance", "spray_dose_threshold", "spray_flow_ml_per_second", "prediction_dt",
                    "ee_position_weight", "max_base_tilt", "sensor_blocked_contact_stiffness",
                    "sensor_blocked_contact_damping", "sensor_blocked_contact_force_limit",
                    "sensor_work_arm_speed", "sensor_work_arm_acceleration", "sensor_work_load_margin",
                    "sensor_work_azimuth_search"):
            if getattr(result, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in ("observation_noise", "sway_amplitude", "sway_frequency", "terrain_footprint_margin",
                    "terrain_theta_margin", "residual_height_scale", "residual_azimuth_scale", "residual_ee_scale"):
            if getattr(result, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        for key in ("start_height_range", "goal_height_range", "start_azimuth_range", "goal_azimuth_range",
                    "target_radius_range", "friction_range", "payload_range", "spray_standoff_range"):
            x = np.asarray(getattr(result, key), float)
            if x.shape != (2,) or not np.isfinite(x).all() or x[0] > x[1]:
                raise ValueError(f"{key} must be an ordered finite pair")
        if result.friction_range[0] <= 0 or result.payload_range[0] < 0 or result.spray_standoff_range[0] <= 0:
            raise ValueError("friction/standoff must be positive and payload nonnegative")
        for key, n in (("arm_home", 6), ("model_residual_scales", 4), ("base_speed_limits", 2), ("work_base_speed_limits", 2)):
            x = np.asarray(getattr(result, key), float)
            if x.shape != (n,) or not np.isfinite(x).all() or (key != "arm_home" and np.any(x <= 0)):
                raise ValueError(f"{key} must contain {n} finite valid values")
        for key in ("spray_required_coverage", "spray_max_off_target_fraction", "depression_support_fraction", "residual_model_scale"):
            if not 0 <= getattr(result, key) <= 1:
                raise ValueError(f"{key} must be in [0,1]")
        for key in ("terrain_enabled", "residual_use_adaptation", "navigation_approach_first", "sensor_only_navigation", "sensor_work_safety_enabled"):
            if not isinstance(getattr(result, key), bool):
                raise ValueError(f"{key} must be bool")
        if not 0 < result.spray_cone_half_angle < np.pi / 2 or result.min_height >= result.max_height:
            raise ValueError("Invalid cone angle or height limits")
        if result.min_height <= 0 or result.nominal_friction > 2 or result.target_radius_range[0] <= TREE_RADIUS:
            raise ValueError("Invalid physical safety limits/prior/target radius")
        if result.extend_min_time > result.extend_max_time:
            raise ValueError("extend_min_time must not exceed extend_max_time")
        if not np.isclose(result.tree_radius, TREE_RADIUS) or not np.isclose(result.wheel_radius, WHEEL_RADIUS):
            raise ValueError("wheel/tree radii must match the frozen model geometry")
        return result


# Explicit named schema, independent of the numeric slice contract.
OBSERVATION_FIELDS = (
    ("base_state", 8), ("arm_q", 6), ("arm_dq", 6), ("cut_pose", 6), ("nozzle_pose", 6),
    ("target_pose", 6), ("target_velocity", 3), ("wheel_rates", 6), ("measured_contact", 6),
    ("slip", 6), ("reference", 8), ("stage", 6), ("stage_elapsed", 1), ("tool_commands", 2),
    ("cut_event", 1), ("synchronized_cut_event", 1), ("estimated_dose_map", 25), ("estimated_spray_metrics", 2),
    ("previous_reference_action", 6), ("previous_model_action", 4),
    ("estimated_external_acceleration", 4), ("terrain_map", 42),
)
OBSERVATION_LAYOUT = {}
_offset = 0
for _name, _count in OBSERVATION_FIELDS:
    OBSERVATION_LAYOUT[_name] = slice(_offset, _offset + _count)
    _offset += _count
FRAME_OBSERVATION_DIM = _offset


def terrain_support(positions, patches, margin=0., theta_margin=0.):
    """All wheel footprints, periodic tree azimuth; no hidden material inputs."""
    positions = np.asarray(positions, float)
    theta = np.arctan2(positions[:, 1], positions[:, 0])
    height = positions[:, 2]
    support = np.ones(len(positions))
    blocked = np.zeros(len(positions), bool)
    for patch in patches:
        inside = ((np.abs(wrap_angle(theta - patch["theta"])) <= patch["theta_half_width"] + theta_margin)
                  & (np.abs(height - patch["height"]) <= patch["height_half_width"] + margin))
        if patch["kind"] == "blocked":
            blocked |= inside
        else:
            support[inside] = np.minimum(support[inside], patch["support_fraction"])
    return support, blocked


class TreeWorkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, task="combined", controller_mode="mpc", config=None, render_mode=None):
        super().__init__()
        if task not in ("combined", "prune", "spray"):
            raise ValueError("task must be combined, prune, or spray")
        if controller_mode not in ("sequential", "mpc", "adaptive_mpc", "mpc_rl", "pure_rl"):
            raise ValueError("Unknown controller mode")
        if render_mode not in (None, "rgb_array", "human"):
            raise ValueError("Unsupported render_mode")
        self.config = TreeWorkConfig.from_value(config)
        self.task_selection = self.task = task
        self.controller_mode, self.render_mode = controller_mode, render_mode
        self.model, self.nominal_model = load_model(), load_model()
        self.data = mujoco.MjData(self.model)
        self.dt = self.config.dt
        self.frame_skip = int(round(self.dt / self.model.opt.timestep))
        if not np.isclose(self.frame_skip * self.model.opt.timestep, self.dt):
            raise ValueError("dt must be an integer multiple of the XML timestep")
        self._base_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "climber_root")
        self._tool_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "tool")
        self._base_joints = np.array([self._id(mujoco.mjtObj.mjOBJ_JOINT, x) for x in BASE_JOINTS])
        self._base_qadr = self.model.jnt_qposadr[self._base_joints]
        self._base_dadr = self.model.jnt_dofadr[self._base_joints]
        self._moving_root_id = int(self.model.jnt_bodyid[self._base_joints[0]])
        self._arm_joints = np.array([self._id(mujoco.mjtObj.mjOBJ_JOINT, x) for x in ARM_JOINTS])
        self._arm_qadr = self.model.jnt_qposadr[self._arm_joints]
        self._arm_dadr = self.model.jnt_dofadr[self._arm_joints]
        self._wheel_joints = np.array([self._id(mujoco.mjtObj.mjOBJ_JOINT, f"wheel{i}_spin") for i in range(1, 7)])
        self._wheel_dadr = self.model.jnt_dofadr[self._wheel_joints]
        self._wheel_act = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_wheel{i}") for i in range(1, 7)])
        self._arm_act = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm_{i}") for i in range(1, 7)])
        self._jaw_act = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "scissors_servo")
        self._jaw_joint = int(self.model.actuator_trnid[self._jaw_act, 0])
        self._jaw_qadr = int(self.model.jnt_qposadr[self._jaw_joint])
        self._traction_sites = np.array([self._id(mujoco.mjtObj.mjOBJ_SITE, f"traction{i}") for i in range(1, 7)])
        self._cut_site = self._id(mujoco.mjtObj.mjOBJ_SITE, "cut_site")
        self._nozzle_site = self._id(mujoco.mjtObj.mjOBJ_SITE, "nozzle_site")
        self._nominal_masses, self._nominal_inertias = self.model.body_mass.copy(), self.model.body_inertia.copy()
        wheel_bodies = self.model.jnt_bodyid[self._wheel_joints]
        axes = self.model.jnt_axis[self._wheel_joints]
        self._nominal_shaft_inertia = self.nominal_model.dof_armature[self._wheel_dadr] + np.sum(
            axes**2*self.nominal_model.body_inertia[wheel_bodies], axis=1)
        self.nominal_mass = float(self.model.body_subtreemass[self._moving_root_id])
        self._arm_limits = self.model.jnt_range[self._arm_joints].copy()
        if np.any(np.asarray(self.config.arm_home) < self._arm_limits[:, 0]) or np.any(np.asarray(self.config.arm_home) > self._arm_limits[:, 1]):
            raise ValueError("arm_home is outside the joint limits")
        self.controller = None
        if controller_mode != "pure_rl":
            from mpc.controller import TreeWorkController
            controller_config = self._controller_config()
            self.controller = TreeWorkController(self.nominal_model,
                mode="residual_mpc" if controller_mode == "mpc_rl" else controller_mode, config=controller_config)
        self.action_space = spaces.Box(-1., 1., shape=(10 if controller_mode == "pure_rl" else 12,), dtype=np.float32)
        self.frame_observation_dim = FRAME_OBSERVATION_DIM
        self.observation_layout = OBSERVATION_LAYOUT.copy()
        self.observation_space = spaces.Box(-np.inf, np.inf,
            shape=(self.frame_observation_dim * self.config.history_length,), dtype=np.float32)
        self._history = deque(maxlen=self.config.history_length)
        self._delay = deque(maxlen=self.config.observation_delay_steps + 1)
        self._renderer = self._viewer = None
        self._initialized = False
        offsets = np.linspace(-.04, .04, 5)
        u, v = np.meshgrid(offsets, offsets)
        self._patch_uv = np.column_stack((u.ravel(), v.ravel()))
        u, v = np.meshgrid(np.linspace(-.12, .12, 25), np.linspace(-.12, .12, 25))
        self._background_uv = np.column_stack((u.ravel(), v.ravel()))
        self._on_target = (np.abs(u.ravel()) <= .045) & (np.abs(v.ravel()) <= .045)

    def _controller_config(self):
        c = self.config
        return dict(horizon=c.horizon, prediction_dt=c.prediction_dt, solve_interval=c.solve_interval,
                    wheel_max=c.wheel_velocity_limit, nominal_friction=c.nominal_friction,
                    normal_force=c.normal_force_per_wheel, traction_gain=c.traction_gain,
                    contact_stiffness=c.contact_stiffness,
                    wheel_radius=c.wheel_radius, tree_radius=c.tree_radius,
                    max_work_base_speed=c.work_base_speed_limits[0],
                    max_work_azimuth_speed=c.work_base_speed_limits[1],
                    height_residual_scale=c.residual_height_scale, azimuth_residual_scale=c.residual_azimuth_scale,
                    ee_residual_scale=c.residual_ee_scale, model_residual_scales=c.model_residual_scales,
                    residual_use_adaptation=c.residual_use_adaptation, residual_model_scale=c.residual_model_scale,
                    ee_position_weight=c.ee_position_weight, height_tolerance=c.height_tolerance,
                    azimuth_tolerance=c.azimuth_tolerance, min_height=c.min_height, max_height=c.max_height,
                    terrain_footprint_margin=c.terrain_footprint_margin,
                    terrain_theta_margin=c.terrain_theta_margin,
                    navigation_approach_first=c.navigation_approach_first,
                    sensor_work_safety_enabled=c.sensor_work_safety_enabled,
                    sensor_work_arm_speed=c.sensor_work_arm_speed,
                    sensor_work_arm_acceleration=c.sensor_work_arm_acceleration,
                    sensor_work_load_margin=c.sensor_work_load_margin,
                    sensor_work_azimuth_search=c.sensor_work_azimuth_search)

    def _id(self, kind, name):
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"Required model object missing: {name}")
        return result

    @staticmethod
    def _validate_map(patches, max_patches=6):
        if not isinstance(patches, (list, tuple)) or len(patches) > max_patches:
            raise ValueError(f"terrain_map must be a list of at most {max_patches} rectangles")
        output = []
        for patch in patches:
            if set(patch) != {"kind", "theta", "height", "theta_half_width", "height_half_width", "support_fraction"}:
                raise ValueError("Invalid terrain rectangle fields")
            p = dict(patch)
            if p["kind"] not in ("blocked", "depression"):
                raise ValueError("Unknown terrain rectangle kind")
            values = [p[x] for x in p if x != "kind"]
            if not np.isfinite(values).all() or p["theta_half_width"] <= 0 or p["theta_half_width"] >= np.pi or p["height_half_width"] <= 0 or not 0 <= p["support_fraction"] <= 1:
                raise ValueError("Invalid terrain rectangle geometry/support")
            output.append(p)
        return output

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        allowed = {"task", "friction_range", "payload_mass", "start_height", "start_azimuth", "goal_height",
                   "goal_azimuth", "target_radius", "target_pos", "sway_amplitude", "terrain_map", "initial_stage", "initial_tilt"}
        if set(options) - allowed:
            raise ValueError(f"Unknown reset options: {sorted(set(options)-allowed)}")
        c = self.config
        self._rng_observation = np.random.default_rng(int(self.np_random.integers(0, 2**31)))
        self.task = options.get("task", self.task_selection)
        if self.task not in ("combined", "prune", "spray"):
            raise ValueError("Invalid reset task")
        friction_range = np.asarray(options.get("friction_range", c.friction_range), float)
        if friction_range.shape != (2,) or not np.isfinite(friction_range).all() or friction_range[0] <= 0 or friction_range[0] > friction_range[1]:
            raise ValueError("Invalid friction_range")
        self._mu = self.np_random.uniform(*friction_range, size=6)
        self._payload = float(options.get("payload_mass", self.np_random.uniform(*c.payload_range)))
        self.start_height = float(options.get("start_height", self.np_random.uniform(*c.start_height_range)))
        self.start_azimuth = float(options.get("start_azimuth", self.np_random.uniform(*c.start_azimuth_range)))
        self.goal_height = float(options.get("goal_height", self.np_random.uniform(*c.goal_height_range)))
        self.goal_azimuth = float(options.get("goal_azimuth", self.np_random.uniform(*c.goal_azimuth_range)))
        self.target_radius = float(options.get("target_radius", self.np_random.uniform(*c.target_radius_range)))
        self._sway = float(options.get("sway_amplitude", c.sway_amplitude))
        self._phase = float(self.np_random.uniform(0, 2*np.pi))
        if not np.isfinite([self._payload, self.start_height, self.start_azimuth, self.goal_height, self.goal_azimuth, self.target_radius, self._sway]).all() or self._payload < 0 or self._sway < 0:
            raise ValueError("Invalid scenario values")
        if not c.min_height < self.start_height < c.max_height or not c.min_height < self.goal_height < c.max_height or self.target_radius <= c.tree_radius:
            raise ValueError("Scenario lies outside the model workspace")
        self.target_axis = np.array([np.cos(self.goal_azimuth), np.sin(self.goal_azimuth), 0.])
        self.target_center = np.asarray(options.get("target_pos", [self.target_radius*self.target_axis[0],
                self.target_radius*self.target_axis[1], self.goal_height+c.target_height_offset]), float)
        if self.target_center.shape != (3,) or not np.isfinite(self.target_center).all():
            raise ValueError("target_pos must contain three finite values")
        default_map = []
        if c.terrain_enabled:
            default_map = [dict(kind="blocked", theta=self.start_azimuth,
                height=.5*(self.start_height+self.goal_height)+self.np_random.uniform(-.015, .015),
                theta_half_width=float(self.np_random.uniform(.08, .14)),
                height_half_width=float(self.np_random.uniform(.04, .065)), support_fraction=0.),
                dict(kind="depression", theta=self.start_azimuth+np.pi/3+self.np_random.uniform(.28, .40),
                height=.5*(self.start_height+self.goal_height)+self.np_random.uniform(-.02, .06),
                theta_half_width=float(self.np_random.uniform(.08, .13)),
                height_half_width=float(self.np_random.uniform(.035, .055)),
                support_fraction=c.depression_support_fraction)]
        self.terrain_map = self._validate_map(options.get("terrain_map", default_map),
                                              max_patches=32 if c.sensor_only_navigation else 6)
        self.discovered_configuration_cells = []
        self._navigation_waypoint = None
        self.model.body_mass[:] = self._nominal_masses
        self.model.body_inertia[:] = self._nominal_inertias
        self.model.body_mass[self._tool_body] += self._payload
        if self._nominal_masses[self._tool_body] > 0:
            self.model.body_inertia[self._tool_body] *= 1+self._payload/self._nominal_masses[self._tool_body]
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        tilt = np.asarray(options.get("initial_tilt", [0., 0.]), float)
        if (tilt.shape != (2,) or not np.isfinite(tilt).all() or np.linalg.norm(tilt) > c.max_base_tilt
                or np.any(tilt < self.model.jnt_range[self._base_joints[2:], 0])
                or np.any(tilt > self.model.jnt_range[self._base_joints[2:], 1])):
            raise ValueError("Invalid initial_tilt")
        self.data.qpos[self._base_qadr] = [self.start_height, self.start_azimuth, *tilt]
        self.data.qpos[self._arm_qadr] = c.arm_home
        self.data.qpos[self._jaw_qadr] = self.model.actuator_ctrlrange[self._jaw_act, 1]
        self.data.ctrl[self._arm_act] = c.arm_home
        self.data.ctrl[self._jaw_act] = self.model.actuator_ctrlrange[self._jaw_act, 1]
        mujoco.mj_forward(self.model, self.data)
        self.stage = options.get("initial_stage", "navigate")
        if self.stage not in STAGES[:-1]:
            raise ValueError("Invalid initial_stage")
        self._steps = self._stage_steps = 0
        self._stable_hold = self._alignment_hold = self._cut_hold = self._overlap_time = 0.
        self._cut = self._task_success = self._success = self._unsafe = self._blocked_violation = False
        self._synchronized_cut_event = False
        self._cut_active = self._nozzle_active = False
        self._tool_command_stage = self.stage
        self._dose = np.zeros(25)
        self._estimated_dose = np.zeros(25)
        self._dose_background = np.zeros(625)
        self._estimated_dose_background = np.zeros(625)
        self._spray_volume = self._spray_flow = self._energy = self._slip_distance = 0.
        self._slip = np.zeros(6)
        # Clamp contact is initially preloaded under the known nominal robot
        # weight. This is elastic contact strain, never a pose assignment.
        # Added task payload is deliberately absent from the holding current.
        nominal_force=self.nominal_mass*9.81/(6*np.cos(np.pi/4))
        self._contact_deflection=np.full(6,nominal_force/c.contact_stiffness)
        self._traction_force=np.full(6,nominal_force)
        self._holding_torque=np.full(6,nominal_force*c.wheel_radius)
        self.data.ctrl[self._wheel_act]=self._holding_torque
        self._previous_wheel_torque=self._holding_torque.copy()
        self._wheel_speed_integral=np.zeros(6)
        self._measured_contact = np.ones(6)
        self._shaft_acceleration = np.zeros(6)
        self._motor_current_torque = self._holding_torque.copy()
        self._contact_load_estimate = self._holding_torque/c.wheel_radius
        self._obstacle_contact_force = np.zeros(6)
        self._obstacle_load_sensor = np.zeros(6)
        self._previous_sensor_base_rates = None
        self._previous_sensor_capture_step = None
        self._terrain_support_truth = np.ones(6)
        self._blocked_contact_truth = np.zeros(6, bool)
        self._blocked_encounter_truth = False
        self._last_diagnostics = {}
        self._previous_reference_action = np.zeros(6)
        self._previous_model_action = np.zeros(4)
        self._requested_policy_action = np.zeros(self.action_space.shape)
        self._applied_reference_action = np.zeros(6)
        self._applied_tool_request = np.zeros(2)
        self._previous_wheel_command = np.zeros(6)
        self._theta_path = [self.start_azimuth]
        self._height_path = [self.start_height]
        self._simultaneous_tool_steps = self._recovery_count = 0
        self._misalignment_detected = False
        self._first_alignment_error = self._realigned_alignment_error = None
        self._max_tilt = 0.
        self._reward_components = {}
        self._update_target()
        if self.controller is not None:
            self.controller.reset()
        self._history.clear()
        self._delay.clear()
        self._initialized = True
        raw = self._raw_sensor_state()
        for _ in range(c.observation_delay_steps+1):
            self._delay.append(deepcopy(raw))
        self._sensor_state = deepcopy(raw)
        frame = self._observation_frame()
        for _ in range(c.history_length):
            self._history.append(frame.copy())
        self._previous_potential = self._potential()
        return np.concatenate(self._history).astype(np.float32), self._info()

    def _update_target(self):
        phase = 2*np.pi*self.config.sway_frequency*self.data.time+self._phase
        tangent = np.array([-self.target_axis[1], self.target_axis[0], 0.])
        self._target_pos = self.target_center+self._sway*np.sin(phase)*tangent
        self._target_velocity = self._sway*2*np.pi*self.config.sway_frequency*np.cos(phase)*tangent
        tangent = np.array([-self.target_axis[1], self.target_axis[0], 0.])
        for name, position in (("leaf_target", self._target_pos), ("protected_leaf", self._target_pos-.16*tangent)):
            target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if target_body >= 0 and self.model.body_mocapid[target_body] >= 0:
                mid = self.model.body_mocapid[target_body]
                self.data.mocap_pos[mid] = position
                self.data.mocap_quat[mid] = [np.cos(.5*self.goal_azimuth), 0., 0., np.sin(.5*self.goal_azimuth)]
        # Keep sites and visible target consistent with integrated qpos and the
        # current mocap command, including the first frame immediately at reset.
        mujoco.mj_forward(self.model, self.data)

    def _pose(self, site):
        return self.data.site_xpos[site].copy(), self.data.site_xmat[site].reshape(3, 3)[:, 0].copy()

    def _raw_sensor_state(self):
        q, dq = self.data.qpos[self._base_qadr], self.data.qvel[self._base_dadr]
        cut_pos, cut_axis = self._pose(self._cut_site)
        nozzle_pos, nozzle_axis = self._pose(self._nozzle_site)
        state = dict(base_height=float(q[0]), base_azimuth=float(q[1]), base_rates=dq[:2].copy(),
            base_tilt=q[2:].copy(), tilt_rates=dq[2:].copy(), arm_q=self.data.qpos[self._arm_qadr].copy(),
            arm_dq=self.data.qvel[self._arm_dadr].copy(), cut_pos=cut_pos, cut_axis=cut_axis,
            nozzle_pos=nozzle_pos, nozzle_axis=nozzle_axis, target_pos=self._target_pos.copy(),
            target_axis=self.target_axis.copy(), target_velocity=self._target_velocity.copy(),
            wheel_rates=self.data.qvel[self._wheel_dadr].copy(), wheel_positions=self.data.site_xpos[self._traction_sites].copy(),
            slip=self._slip.copy())
        sigma = self.config.observation_noise
        if sigma:
            for key in ("base_height", "base_azimuth", "base_rates", "base_tilt", "tilt_rates", "arm_q", "arm_dq",
                        "cut_pos", "nozzle_pos", "target_pos", "target_velocity", "wheel_rates", "wheel_positions", "slip"):
                state[key] = state[key]+self._rng_observation.normal(0, sigma, np.shape(state[key]))
        if self.config.sensor_only_navigation:
            # Encoder/IMU odometry and known ring geometry, not terrain lookup.
            theta, roll, pitch = state["base_azimuth"], *state["base_tilt"]
            ct, st, cr, sr, cp, sp = np.cos(theta), np.sin(theta), np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
            rz = np.array([[ct, -st, 0.], [st, ct, 0.], [0., 0., 1.]])
            rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
            ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
            rotation = rz @ rx @ ry
            lever = self.model.site_pos[self._traction_sites] @ rotation.T
            state["wheel_positions"] = lever + [0., 0., state["base_height"]]
            omega = (state["base_rates"][1]*np.array([0., 0., 1.])
                     + state["tilt_rates"][0]*rz[:, 0] + state["tilt_rates"][1]*(rz @ rx)[:, 1])
            velocity = np.cross(np.broadcast_to(omega, lever.shape), lever)
            velocity[:, 2] += state["base_rates"][0]
            directions = rolling_directions(theta, rotation)
            state["slip"] = self.config.wheel_radius*state["wheel_rates"] - np.sum(velocity*directions, axis=1)
            motor = self._motor_current_torque.copy()
            accel = self._shaft_acceleration.copy()
            load = self._contact_load_estimate.copy()
            if sigma:
                motor += self._rng_observation.normal(0., sigma, 6)
                accel += self._rng_observation.normal(0., sigma/self.dt, 6)
                load += self._rng_observation.normal(0., sigma/self.config.wheel_radius, 6)
            sensor_accel = np.zeros(2)
            if self._previous_sensor_capture_step is not None and self._steps > self._previous_sensor_capture_step:
                sensor_accel = (state["base_rates"]-self._previous_sensor_base_rates)/((self._steps-self._previous_sensor_capture_step)*self.dt)
            if self._previous_sensor_capture_step != self._steps:
                self._previous_sensor_base_rates = np.asarray(state["base_rates"]).copy()
                self._previous_sensor_capture_step = self._steps
            # A non-localized force residual from current/encoders/odometry.
            # No force truth, terrain query, or calibrated payload is read here.
            # Arm acceleration, unknown load and model errors remain in it.
            external_load = max(0., float(load @ directions[:, 2]
                -self.nominal_mass*(9.81+sensor_accel[0])
                -self.nominal_model.dof_damping[self._base_dadr[0]]*state["base_rates"][0]))
            obstacle_load = np.full(6, external_load/6.)
            nominal_load = self.nominal_mass*9.81/(6*np.cos(np.pi/4))
            state.update(measured_contact=np.clip(np.abs(load)/nominal_load, 0., 1.),
                contact_tangential_load=load, contact_obstacle_load=np.maximum(obstacle_load, 0.),
                estimated_external_vertical_load=external_load, base_acceleration=sensor_accel,
                wheel_motor_current=motor, wheel_motor_torque=motor.copy(),
                wheel_shaft_acceleration=accel, sensor_capture_step=self._steps,
                sensor_capture_time=float(self.data.time))
        else:
            state["measured_contact"] = terrain_support(state["wheel_positions"], self.terrain_map,
                self.config.terrain_footprint_margin, self.config.terrain_theta_margin)[0]
        return state

    def set_navigation_reference(self, height, azimuth):
        """A learned navigation action supplies a waypoint, never a hidden route."""
        point = np.asarray([height, azimuth], float)
        if not np.isfinite(point).all() or not self.config.min_height < point[0] < self.config.max_height:
            raise ValueError("Navigation waypoint must be finite and within height bounds")
        self._navigation_waypoint = point.copy()

    def set_discovered_configuration_cells(self, cells):
        """Accept externally estimated robot-pose risks, not true surface patches."""
        if not isinstance(cells, (list, tuple)):
            raise ValueError("Discovered configuration cells must be a sequence")
        output = []
        for cell in cells:
            if not isinstance(cell, dict) or set(cell) - {"height", "azimuth", "risk", "confidence"} or not {"height", "azimuth", "risk"} <= set(cell):
                raise ValueError("Configuration cells require height, azimuth and estimated risk")
            item = {key: float(value) for key, value in cell.items()}
            if not np.isfinite(list(item.values())).all() or not 0 <= item["risk"] <= 1 or not 0 <= item.get("confidence", 1.) <= 1:
                raise ValueError("Configuration cell estimates must be finite probabilities")
            output.append(item)
        self.discovered_configuration_cells = output[-6:]

    def get_control_state(self):
        if not self._initialized:
            raise RuntimeError("reset must precede control state access")
        state = deepcopy(self._sensor_state)
        state.update(ref_height=self.goal_height, ref_azimuth=self.goal_azimuth,
            ref_ee=state["target_pos"].copy(), ref_axis=state["target_axis"].copy(),
            stage=self.stage, task=self.task, arm_home=np.asarray(self.config.arm_home), dt=self.dt,
            sensor_delay_seconds=self.dt*self.config.observation_delay_steps,
            sensor_noise_sigma=self.config.observation_noise,
            time=float(self.data.time), terrain_map=[] if self.config.sensor_only_navigation else deepcopy(self.terrain_map),
            local_terrain_map=[] if self.config.sensor_only_navigation else deepcopy(self.terrain_map),
            nominal_mass=self.nominal_mass, previous_wheel_command=self._previous_wheel_command.copy(),
            estimated_external_acceleration=np.asarray(self._last_diagnostics.get("estimated_external_acceleration", np.zeros(4))),
            estimated_dose_map=self._estimated_dose.copy(), stage_elapsed_time=self._stage_steps*self.dt)
        if self.config.sensor_only_navigation:
            state.update(sensor_only_navigation=True,
                navigation_waypoint=None if self.stage != "navigate" or self._navigation_waypoint is None else self._navigation_waypoint.copy(),
                discovered_configuration_cells=deepcopy(self.discovered_configuration_cells),
                wheel_torque_command=self._previous_wheel_torque.copy(),
                wheel_motor_current_units="N m torque-equivalent calibrated current proxy",
                measured_contact_semantics="analog shaft-load ratio; not friction or geometric support")
            state["contact_obstacle_load_semantics"] = "equal partition of non-localized current/odometry force residual; not per-wheel contact force"
        return state

    def _external_forces(self):
        """Only contact traction + shaft reaction; native joints handle clamp/tilt."""
        self.data.qfrc_applied[:] = 0.
        sites = self.data.site_xpos[self._traction_sites]
        support, blocked = terrain_support(sites, self.terrain_map,
            self.config.terrain_footprint_margin, self.config.terrain_theta_margin)
        self._terrain_support_truth = support.copy()
        self._blocked_contact_truth = blocked.copy()
        self._blocked_encounter_truth |= bool(blocked.any())
        # A rigid lip adds passive chassis resistance while retaining tree
        # traction. Only a depression reduces the contact's friction bound.
        if not self.config.sensor_only_navigation:
            self._blocked_violation |= bool(blocked.any())
            self._measured_contact = support
        self._obstacle_contact_force[:] = 0.
        theta = float(self.data.qpos[self._base_qadr[1]])
        directions = rolling_directions(theta,self.data.xmat[self._base_id])
        site_jac = np.zeros((3, self.model.nv))
        angular_jac = np.zeros_like(site_jac)
        for i, site in enumerate(self._traction_sites):
            mujoco.mj_jacSite(self.model, self.data, site_jac, angular_jac, int(site))
            velocity = site_jac @ self.data.qvel
            slip = self.config.wheel_radius*self.data.qvel[self._wheel_dadr[i]]-float(velocity @ directions[i])
            self._slip[i] = slip
            limit = self._mu[i]*self.config.normal_force_per_wheel*support[i]
            # Elastic bristles retain static support at zero relative speed.
            # On sliding, the Coulomb bound also resets bristle strain, so a
            # vanished/depressed contact cannot store inaccessible force.
            self._contact_deflection[i]=float(np.clip(self._contact_deflection[i]
                +float(self.model.opt.timestep)*slip,-limit/self.config.contact_stiffness,
                limit/self.config.contact_stiffness)) if limit else 0.
            force = float(np.clip(self.config.contact_stiffness*self._contact_deflection[i]
                                 +self.config.traction_gain*slip, -limit, limit))
            self._traction_force[i]=force
            mujoco.mj_applyFT(self.model, self.data, force*directions[i], np.zeros(3),
                             sites[i], self._base_id, self.data.qfrc_applied)
            self.data.qfrc_applied[self._wheel_dadr[i]] -= force*self.config.wheel_radius
            if self.config.sensor_only_navigation and blocked[i]:
                # An uncalibrated lip/snags proxy: a passive unilateral spring
                # opposes upward penetration. Lateral clearance releases it.
                # Neither robot position nor velocity is assigned or clipped.
                theta_i = float(np.arctan2(sites[i, 1], sites[i, 0]))
                penetration = 0.
                for patch in self.terrain_map:
                    if patch["kind"] != "blocked":
                        continue
                    inside_theta = abs(float(wrap_angle(theta_i-patch["theta"]))) <= patch["theta_half_width"]+self.config.terrain_theta_margin
                    lower = patch["height"]-patch["height_half_width"]-self.config.terrain_footprint_margin
                    upper = patch["height"]+patch["height_half_width"]+self.config.terrain_footprint_margin
                    if inside_theta and lower <= sites[i, 2] <= upper:
                        penetration = max(penetration, float(sites[i, 2]-lower))
                contact = min(self.config.sensor_blocked_contact_force_limit,
                    self.config.sensor_blocked_contact_stiffness*penetration
                    +self.config.sensor_blocked_contact_damping*max(float(velocity[2]), 0.))
                self._obstacle_contact_force[i] = contact
                mujoco.mj_applyFT(self.model, self.data, np.array([0., 0., -contact]), np.zeros(3),
                    sites[i], self._base_id, self.data.qfrc_applied)

    def _wheel_drive_torque(self, wheel_speed_target):
        """Encoder feedback for the direct policy's two speed channels.

        Projecting the measured shaft rates before feedback leaves passive
        tilt wheel motion free. Neither this loop nor its fixed nominal-weight
        holding current has an independently controllable wheel nullspace.
        A common saturation scale preserves the two-input torque manifold.
        """
        target=project_wheel_commands(np.asarray(wheel_speed_target,float))
        measured=project_wheel_commands(self.data.qvel[self._wheel_dadr])
        # This torque loop runs at 2 ms. Its gain is bounded below the shaft
        # inertia's explicit-step stability limit; the old implicit velocity
        # servo's gain 2 cannot be copied into an explicit current controller.
        damping=self.model.dof_damping[self._wheel_dadr]
        error=target-measured
        gain_i=self.config.wheel_speed_integral_gain
        integral=self._wheel_speed_integral+float(self.model.opt.timestep)*error
        integral*=min(1.,.5/max(float(np.max(np.abs(gain_i*integral))),1e-12))
        torque=self._holding_torque+damping*target+self.config.wheel_speed_feedback_gain*error+gain_i*integral
        limits=np.minimum(np.abs(self.model.actuator_ctrlrange[self._wheel_act,0]),
                          self.model.actuator_ctrlrange[self._wheel_act,1])
        scale=min(1.,float(np.min(limits/np.maximum(np.abs(torque),1e-12))))
        # Freeze integration when the shared current limit cannot satisfy an
        # error in that same direction. Reverse errors can unwind the state.
        if scale>=1. or float(error@(torque-scale*torque))<=0.:
            self._wheel_speed_integral=integral
        else:
            torque=self._holding_torque+damping*target+self.config.wheel_speed_feedback_gain*error+gain_i*self._wheel_speed_integral
            scale=min(1.,float(np.min(limits/np.maximum(np.abs(torque),1e-12))))
        return scale*torque

    def _tool_geometry(self, state=None):
        if state is None:
            cut_pos, cut_axis = self._pose(self._cut_site)
            nozzle_pos, nozzle_axis = self._pose(self._nozzle_site)
            target, axis = self._target_pos, self.target_axis
        else:
            cut_pos, cut_axis, nozzle_pos, nozzle_axis = (state[x] for x in ("cut_pos", "cut_axis", "nozzle_pos", "nozzle_axis"))
            target, axis = state["target_pos"], state["target_axis"]
        cut_error = float(np.linalg.norm(cut_pos-target))
        cut_axis_error = float(np.linalg.norm(cut_axis-axis))
        ray = target-nozzle_pos
        distance = float(np.linalg.norm(ray))
        nozzle_axis_error = float(np.linalg.norm(nozzle_axis-ray/max(distance, 1e-12)))
        cut_ok = cut_error <= self.config.tool_position_tolerance and cut_axis_error <= self.config.tool_axis_tolerance
        nozzle_ok = (self.config.spray_standoff_range[0] <= distance <= self.config.spray_standoff_range[1]
                      and nozzle_axis_error <= self.config.tool_axis_tolerance)
        return cut_ok, nozzle_ok, cut_error, cut_axis_error, distance, nozzle_axis_error

    def _plane_points(self, uv, target, axis):
        tangent = np.array([-axis[1], axis[0], 0.])
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        vertical = np.cross(axis, tangent)
        return target+uv[:, :1]*tangent+uv[:, 1:]*vertical

    def _cone_mass_weights(self, uv, nozzle_pos, nozzle_axis, target, target_axis, cell_area):
        """Normalized truncated Gaussian on a plane, with inverse-square spread.

        The continuous kernel integrates to commanded volume on an infinite
        perpendicular plane. Finite cells approximate that integral; no sampled
        grid renormalization artificially puts all spray onto the target.
        """
        points = self._plane_points(uv, target, target_axis)
        vectors = points-nozzle_pos
        depth = vectors @ nozzle_axis
        radius_sq = np.maximum(np.sum(vectors*vectors, axis=1)-depth*depth, 0.)
        radius = np.maximum(depth, 1e-9)*np.tan(self.config.spray_cone_half_angle)
        inside = (depth > 0) & (radius_sq <= radius*radius)
        reference_radius = self.config.spray_reference_distance*np.tan(self.config.spray_cone_half_angle)
        normalizer = .5*np.pi*reference_radius**2*(1-np.exp(-2))
        weights = np.zeros(len(uv))
        projected_area = cell_area*abs(float(np.dot(nozzle_axis, target_axis)))
        weights[inside] = (np.exp(-2*radius_sq[inside]/radius[inside]**2)*projected_area/normalizer
                           *(self.config.spray_reference_distance/depth[inside])**2)
        return weights

    def _integrate_dose(self, estimated=False):
        if self._spray_flow <= 0:
            return
        state = self._sensor_state if estimated else None
        if state is None:
            nozzle_pos, nozzle_axis = self._pose(self._nozzle_site)
            target, axis = self._target_pos, self.target_axis
        else:
            nozzle_pos, nozzle_axis, target, axis = (state[x] for x in ("nozzle_pos", "nozzle_axis", "target_pos", "target_axis"))
        volume = self._spray_flow*self.config.spray_flow_ml_per_second*self.dt
        dose = self._estimated_dose if estimated else self._dose
        background = self._estimated_dose_background if estimated else self._dose_background
        dose += volume*self._cone_mass_weights(self._patch_uv, nozzle_pos, nozzle_axis, target, axis, .02**2)
        background += volume*self._cone_mass_weights(self._background_uv, nozzle_pos, nozzle_axis, target, axis, .01**2)

    def _spray_metrics(self, estimated=False):
        dose = self._estimated_dose if estimated else self._dose
        background = self._estimated_dose_background if estimated else self._dose_background
        coverage = float(np.mean(dose >= self.config.spray_dose_threshold))
        total = float(background.sum())
        off_target = float(background[~self._on_target].sum()/max(total, 1e-12))
        success = coverage >= self.config.spray_required_coverage and off_target <= self.config.spray_max_off_target_fraction
        return coverage, off_target, success

    def _set_stage(self, stage):
        self.stage, self._stage_steps = stage, 0
        self._stable_hold = self._alignment_hold = 0.

    def _advance_stage(self):
        c = self.config
        measured = self.get_control_state()
        cut_ok, nozzle_ok, error, *_ = self._tool_geometry(measured)
        pose_ok = ((cut_ok or self.task == "spray") and (nozzle_ok or self.task == "prune"))
        if self.stage == "navigate":
            arrived = (abs(measured["base_height"]-self.goal_height) <= c.height_tolerance
                       and abs(float(wrap_angle(measured["base_azimuth"]-self.goal_azimuth))) <= c.azimuth_tolerance
                       and np.linalg.norm(measured["base_rates"]) < .12)
            self._stable_hold = self._stable_hold+self.dt if arrived else 0.
            if self._stable_hold >= c.arrival_hold_time:
                self._set_stage("extend")
        elif self.stage == "extend":
            elapsed = self._stage_steps*self.dt
            if c.sensor_work_safety_enabled:
                # Holding a folded arm during a legitimate h/theta relocation
                # does not mean extension has completed. The opt-in controller
                # reports readiness from measured joint error and rates; no
                # timer fabricates a completed extension or alters the plant.
                extension_complete = (bool(self._last_diagnostics.get("cooperative_extension_ready", False))
                                      and np.linalg.norm(measured["arm_dq"]) < .15)
            else:
                extension_complete = (elapsed >= c.extend_max_time or np.linalg.norm(measured["arm_dq"]) < .10)
            if elapsed >= c.extend_min_time and extension_complete:
                self._first_alignment_error = error
                self._misalignment_detected = error > c.realign_trigger_tolerance
                self._set_stage("realign")
        elif self.stage == "realign":
            self._alignment_hold = self._alignment_hold+self.dt if pose_ok else 0.
            if self._alignment_hold >= c.alignment_hold_time:
                self._realigned_alignment_error = error
                self._set_stage("operate")
        elif self.stage == "operate":
            # Both tasks have true geometric scoring; observation remains an estimate.
            spray_success = self._spray_metrics()[2]
            self._task_success = ((self._cut or self.task == "spray") and (spray_success or self.task == "prune")
                                  and (self.task != "combined" or (self._synchronized_cut_event
                                       and self._overlap_time >= c.required_overlap_time)))
            if self._task_success:
                self._set_stage("retract")
            elif not pose_ok:
                self._alignment_hold += self.dt
                if self._alignment_hold > .32:
                    self._recovery_count += 1
                    self._set_stage("realign")
            else:
                self._alignment_hold = 0.
        elif self.stage == "retract":
            withdrawn = (np.linalg.norm(measured["arm_q"]-np.asarray(c.arm_home)) < .16
                         and np.linalg.norm(measured["arm_dq"]) < .15 and np.linalg.norm(measured["base_rates"]) < .12)
            if self._task_success and withdrawn:
                self._success = True
                self._set_stage("done")

    def _potential(self):
        if self.stage == "navigate":
            q = self.data.qpos[self._base_qadr]
            return -abs(q[0]-self.goal_height)-.15*abs(float(wrap_angle(q[1]-self.goal_azimuth)))
        return -float(np.linalg.norm(self.data.site_xpos[self._cut_site]-self._target_pos))

    def step(self, action):
        if not self._initialized:
            raise RuntimeError("reset must precede step")
        action = np.asarray(action, float)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"action must have shape {self.action_space.shape} and finite values")
        action = np.clip(action, -1., 1.)
        self._requested_policy_action = action.copy()
        old_cut, old_success, old_task = self._cut, self._success, self._task_success
        old_soft = float(np.minimum(self._dose/self.config.spray_dose_threshold, 1).mean())
        old_volume = self._spray_volume
        old_stage = self.stage
        self._tool_command_stage = old_stage
        if self.controller_mode == "pure_rl":
            speed_limits=np.asarray(self.config.base_speed_limits)
            if self.stage!='navigate':speed_limits=np.minimum(speed_limits,self.config.work_base_speed_limits)
            wheel = wheel_commands(action[:2]*speed_limits, limit=self.config.wheel_velocity_limit)
            home = np.asarray(self.config.arm_home)
            arm_action = action[2:8]
            arm = home+np.maximum(arm_action, 0)*(self._arm_limits[:, 1]-home)+np.minimum(arm_action, 0)*(home-self._arm_limits[:, 0])
            if self.stage in ("navigate", "done"):
                arm = home.copy()
            tool_action = action[8:10]
            self._applied_reference_action = np.zeros(6)
            self._previous_model_action = np.zeros(4)
            self._last_diagnostics = {}
        else:
            refs, model = action[:6].copy(), action[6:10].copy()
            if self.controller_mode != "mpc_rl":
                refs[:] = 0.
                model[:] = 0.
            if self.stage == "navigate":
                if self.config.residual_height_scale:
                    refs[0] = np.clip(refs[0], -.5*self.config.height_tolerance/self.config.residual_height_scale,
                                     .5*self.config.height_tolerance/self.config.residual_height_scale)
                if self.config.residual_azimuth_scale:
                    refs[1] = np.clip(refs[1], -.5*self.config.azimuth_tolerance/self.config.residual_azimuth_scale,
                                     .5*self.config.azimuth_tolerance/self.config.residual_azimuth_scale)
            command = self.controller.command(self.get_control_state(), refs, model)
            wheel, arm = command["wheel_command"], command["arm_command"]
            self._last_diagnostics = command["diagnostics"]
            self._applied_reference_action = refs.copy()
            self._previous_reference_action, self._previous_model_action = refs.copy(), model.copy()
            tool_action = action[10:12] if self.controller_mode == "mpc_rl" else np.zeros(2)
        wheel, arm = np.asarray(wheel, float), np.asarray(arm, float)
        if wheel.shape != (6,) or arm.shape != (6,) or not np.isfinite(wheel).all() or not np.isfinite(arm).all():
            raise ValueError("Controller returned invalid actuator commands")
        matrix = wheel_mixer()
        if not np.allclose(wheel, matrix @ (np.linalg.pinv(matrix) @ wheel), atol=1e-7):
            raise ValueError("wheel command leaves the rank-two height/azimuth manifold")
        wheel *= min(1., self.config.wheel_velocity_limit/max(np.max(np.abs(wheel)), 1e-12))
        arm = np.clip(arm, self._arm_limits[:, 0], self._arm_limits[:, 1])
        self._previous_wheel_command = wheel.copy()
        previous_arm_command = self.data.ctrl[self._arm_act].copy()
        torque=None if self.controller_mode=="pure_rl" else np.asarray(command['wheel_torque_command'],float)
        if torque is not None:
            if torque.shape!=(6,) or not np.isfinite(torque).all():
                raise ValueError('Controller returned invalid wheel torque')
            if not np.allclose(torque,project_wheel_commands(torque),atol=1e-7):
                raise ValueError('Wheel torque leaves the rank-two drive manifold')
        # Zero residual reproduces baseline full requested pump and scissors.
        requests = np.clip(1.+.5*np.asarray(tool_action), .5, 1.) if self.stage == "operate" else np.zeros(2)
        if self.controller_mode == "pure_rl":
            requests = .5+.5*np.asarray(tool_action) if self.stage == "operate" else np.zeros(2)
        if self.task == "spray":
            requests[0] = 0.
        elif self.task == "prune":
            requests[1] = 0.
        self._applied_tool_request = requests.copy()
        cut_ok, nozzle_ok, *_ = self._tool_geometry()
        jaw_range = self.model.actuator_ctrlrange[self._jaw_act]
        close_permitted = requests[0] > 0 and cut_ok
        if self.task == "combined":
            close_permitted = close_permitted and requests[1] > 0 and nozzle_ok
        if close_permitted:
            self.data.ctrl[self._jaw_act] = np.clip(self.data.qpos[self._jaw_qadr]
                -self.config.scissors_closure_speed*requests[0]*self.dt, jaw_range[0], jaw_range[1])
        else:
            self.data.ctrl[self._jaw_act] = jaw_range[1]
        if self.config.sensor_only_navigation:
            encoder_before = self.data.qvel[self._wheel_dadr].copy()
            motor_samples = np.zeros(6)
            damping_samples = np.zeros(6)
        for substep in range(self.frame_skip):
            # The actual position servos follow a continuous command ramp in
            # every mode, including direct RL. This changes ctrl, never state.
            fraction = (substep+1)/self.frame_skip
            self.data.ctrl[self._arm_act] = previous_arm_command+fraction*(arm-previous_arm_command)
            self.data.ctrl[self._wheel_act]=self._wheel_drive_torque(wheel) if torque is None else torque
            self._previous_wheel_torque=self.data.ctrl[self._wheel_act].copy()
            self._external_forces()
            mujoco.mj_step(self.model, self.data)
            if self.config.sensor_only_navigation:
                motor_samples += self.data.actuator_force[self._wheel_act]
                damping_samples += self.model.dof_damping[self._wheel_dadr]*self.data.qvel[self._wheel_dadr]
        if self.config.sensor_only_navigation:
            self._shaft_acceleration = (self.data.qvel[self._wheel_dadr]-encoder_before)/self.dt
            self._motor_current_torque = motor_samples/self.frame_skip
            self._contact_load_estimate = (self._motor_current_torque-damping_samples/self.frame_skip
                -self._nominal_shaft_inertia*self._shaft_acceleration)/self.config.wheel_radius
            nominal_load = self.nominal_mass*9.81/(6*np.cos(np.pi/4))
            self._measured_contact = np.clip(np.abs(self._contact_load_estimate)/nominal_load, 0., 1.)
        self._update_target()
        self._steps += 1
        self._stage_steps += 1
        self._theta_path.append(float(self.data.qpos[self._base_qadr[1]]))
        self._height_path.append(float(self.data.qpos[self._base_qadr[0]]))
        cut_ok, nozzle_ok, *_ = self._tool_geometry()
        jaw_closed = self.data.qpos[self._jaw_qadr] <= jaw_range[0]+.2*(jaw_range[1]-jaw_range[0])
        self._cut_active = bool(self.stage == "operate" and requests[0] > 0 and cut_ok and jaw_closed)
        self._nozzle_active = bool(self.stage == "operate" and requests[1] > 0 and nozzle_ok)
        self._spray_flow = float(requests[1]) if self._nozzle_active else 0.
        self._cut_hold = self._cut_hold+self.dt if self._cut_active else 0.
        if self._cut_hold >= self.config.cut_hold_time and not self._cut:
            self._cut = True
        if self._cut_active and self._nozzle_active:
            self._overlap_time += self.dt
            self._simultaneous_tool_steps += 1
        if self._cut and not old_cut:
            self._synchronized_cut_event = bool(self._nozzle_active
                and self._overlap_time >= self.config.required_overlap_time)
        self._integrate_dose()
        self._spray_volume += self._spray_flow*self.config.spray_flow_ml_per_second*self.dt
        self._delay.append(self._raw_sensor_state())
        self._sensor_state = deepcopy(self._delay[0])
        self._integrate_dose(estimated=True)
        self._advance_stage()
        tilt = self.data.qpos[self._base_qadr[2:]]
        self._max_tilt = max(self._max_tilt, float(np.linalg.norm(tilt)))
        h = self.data.qpos[self._base_qadr[0]]
        self._unsafe = bool((self._blocked_violation and not self.config.sensor_only_navigation) or not np.isfinite(self.data.qpos).all()
                            or self._max_tilt > self.config.max_base_tilt or h < self.config.min_height or h > self.config.max_height)
        if self._unsafe:
            self._success = False
        self._energy += float(np.sum(np.abs(self.data.actuator_force[self._wheel_act]*self.data.qvel[self._wheel_dadr]))*self.dt)
        self._slip_distance += float(np.mean(np.abs(self._slip))*self.dt)
        potential = self._potential()
        # Stage changes do not produce a reward by changing distance definitions.
        progress = 0. if self.stage != old_stage else 1.5*(potential-self._previous_potential)
        self._previous_potential = potential
        self._reward_components = dict(progress=progress, cut=3.*float(self._cut and not old_cut),
            operation=5.*float(self._task_success and not old_task), completion=15.*float(self._success and not old_success),
            soft_dose=2.*(float(np.minimum(self._dose/self.config.spray_dose_threshold, 1).mean())-old_soft),
            spray_cost=-.04*(self._spray_volume-old_volume), time=-.02,
            unsafe=-8.*float(self._unsafe),
            reference_regularization=-.01*float(np.mean(action[:5]**2)) if self.controller_mode == "mpc_rl" else 0.,
            model_regularization=-.002*float(np.mean(action[6:10]**2)) if self.controller_mode == "mpc_rl" else 0.)
        reward = float(sum(self._reward_components.values()))
        frame = self._observation_frame()
        self._history.append(frame)
        terminated = bool(self._success or self._unsafe)
        truncated = bool(self._steps >= self.config.max_episode_steps and not terminated)
        return np.concatenate(self._history).astype(np.float32), reward, terminated, truncated, self._info()

    def _observation_frame(self):
        s = self.get_control_state()
        values = dict(base_state=np.r_[s["base_height"], s["base_azimuth"], s["base_rates"], s["base_tilt"], s["tilt_rates"]],
            arm_q=s["arm_q"], arm_dq=s["arm_dq"], cut_pose=np.r_[s["cut_pos"], s["cut_axis"]],
            nozzle_pose=np.r_[s["nozzle_pos"], s["nozzle_axis"]], target_pose=np.r_[s["target_pos"], s["target_axis"]],
            target_velocity=s["target_velocity"], wheel_rates=s["wheel_rates"], measured_contact=s["measured_contact"],
            slip=s["slip"], reference=np.r_[s["ref_height"], s["ref_azimuth"], s["ref_ee"], s["ref_axis"]],
            stage=np.eye(len(STAGES))[STAGES.index(self.stage)], stage_elapsed=[self._stage_steps*self.dt/20.],
            tool_commands=[float(self._cut_active), self._spray_flow], cut_event=[float(self._cut)],
            synchronized_cut_event=[float(self._synchronized_cut_event)],
            estimated_dose_map=np.clip(self._estimated_dose/self.config.spray_dose_threshold, 0, 2),
            estimated_spray_metrics=self._spray_metrics(estimated=True)[:2],
            previous_reference_action=self._previous_reference_action, previous_model_action=self._previous_model_action,
            estimated_external_acceleration=s["estimated_external_acceleration"])
        map_array = np.zeros((6, 7))
        if self.config.sensor_only_navigation:
            for i, cell in enumerate(self.discovered_configuration_cells):
                map_array[i] = [cell["risk"], np.sin(cell["azimuth"]), np.cos(cell["azimuth"]),
                                cell["height"], 0., 0., cell.get("confidence", 1.)]
        else:
            for i, p in enumerate(self.terrain_map):
                map_array[i] = [1 if p["kind"] == "blocked" else -1, np.sin(p["theta"]), np.cos(p["theta"]),
                                p["height"], p["theta_half_width"], p["height_half_width"], p["support_fraction"]]
        values["terrain_map"] = map_array.ravel()
        frame = np.empty(self.frame_observation_dim, np.float32)
        for name, count in OBSERVATION_FIELDS:
            value = np.asarray(values[name], float).ravel()
            if value.size != count or not np.isfinite(value).all():
                raise RuntimeError(f"Invalid observation field {name}")
            frame[OBSERVATION_LAYOUT[name]] = value
        return frame

    def _full_info(self):
        coverage, off_target, spray_success = self._spray_metrics()
        geometry = self._tool_geometry()
        q = self.data.qpos[self._base_qadr]
        return dict(schema_version=SCHEMA_VERSION, task=self.task, controller_mode=self.controller_mode,
            stage=self.stage, elapsed_time=float(self.data.time), steps=self._steps,
            success=bool(self._success and not self._unsafe), task_success=bool(self._task_success),
            cut_event=bool(self._cut), spray_success=bool(spray_success), coverage=coverage,
            synchronized_cut_event=self._synchronized_cut_event,
            cut_event_with_nozzle_on=self._synchronized_cut_event,
            off_target_fraction=off_target, spray_volume_ml=float(self._spray_volume),
            cut_active=bool(self._cut_active), nozzle_active=bool(self._nozzle_active), spray_flow=float(self._spray_flow),
            tool_command_stage=self._tool_command_stage, scissors_angle=float(self.data.qpos[self._jaw_qadr]),
            scissors_command=float(self.data.ctrl[self._jaw_act]),
            cut_position_error=geometry[2], cut_axis_error=geometry[3], nozzle_standoff=geometry[4], nozzle_axis_error=geometry[5],
            simultaneous_tool_steps=self._simultaneous_tool_steps, overlap_time=float(self._overlap_time),
            first_alignment_error=self._first_alignment_error, realigned_alignment_error=self._realigned_alignment_error,
            misalignment_detected=self._misalignment_detected, recovery_count=self._recovery_count,
            base_height=float(q[0]), base_azimuth=float(q[1]), base_tilt=q[2:].tolist(), max_tilt=self._max_tilt,
            unsafe=self._unsafe, fallen=bool(q[0] < self.config.min_height), blocked_violation=self._blocked_violation,
            height_path=self._height_path.copy(), theta_path=self._theta_path.copy(),
            measured_contact=self._measured_contact.tolist(), wheel_slip=self._slip.tolist(),
            wheel_rates=self.data.qvel[self._wheel_dadr].tolist(),
            wheel_positions=self.data.site_xpos[self._traction_sites].tolist(),
            wheel_traction_force=self._traction_force.tolist(),
            wheel_contact_deflection=self._contact_deflection.tolist(),
            wheel_contact_elastic_energy=float(.5*self.config.contact_stiffness*np.sum(self._contact_deflection**2)),
            wheel_holding_torque=self._holding_torque.tolist(),
            wheel_motor_torque=self.data.actuator_force[self._wheel_act].tolist(),
            wheel_torque_command=self._previous_wheel_torque.tolist(),
            wheel_speed_integral_torque=(self.config.wheel_speed_integral_gain*self._wheel_speed_integral).tolist(),
            wheel_command_units="rad/s reference/diagnostic; native motor input is wheel_torque_command in N m",
            wheel_command_purpose="two_input_speed_target" if self.controller_mode=="pure_rl" else "nominal_rolling_speed_diagnostic",
            wheel_contact_model="coulomb_limited_elastic_brush_with_viscous_damping",
            arm_q=self.data.qpos[self._arm_qadr].tolist(), arm_command=self.data.ctrl[self._arm_act].tolist(),
            native_base_bias=self.data.qfrc_bias[self._base_dadr].tolist(),
            native_base_applied=self.data.qfrc_applied[self._base_dadr].tolist(),
            native_base_actuator=self.data.qfrc_actuator[self._base_dadr].tolist(),
            native_base_constraint=self.data.qfrc_constraint[self._base_dadr].tolist(),
            base_acceleration=self.data.qacc[self._base_dadr].tolist(),
            energy_proxy_j=self._energy, slip_distance_m=self._slip_distance,
            requested_policy_action=self._requested_policy_action.tolist(), applied_reference_action=self._applied_reference_action.tolist(),
            applied_tool_request=self._applied_tool_request.tolist(), wheel_command=self._previous_wheel_command.tolist(),
            reward_components=self._reward_components.copy(), controller_diagnostics=deepcopy(self._last_diagnostics),
            dose_proxy_units="mL geometric exposure per cell; uncalibrated deposition proxy",
            nominal_mass_kg=self.nominal_mass, actual_mass_kg=float(self.model.body_subtreemass[self._moving_root_id]),
            scenario_parameters=dict(friction=self._mu.tolist(), payload=self._payload, terrain_map=deepcopy(self.terrain_map),
                start_height=self.start_height, start_azimuth=self.start_azimuth, goal_height=self.goal_height,
                goal_azimuth=self.goal_azimuth, target_radius=self.target_radius, target_center=self.target_center.tolist(),
                sway_amplitude=self._sway, sway_phase=self._phase))

    def get_audit_state(self):
        """Privileged simulation logging only; never a controller/policy input."""
        info = self._full_info()
        info.update(terrain_map_truth=deepcopy(self.terrain_map),
                    terrain_support_truth=self._terrain_support_truth.tolist(),
                    blocked_contact_truth=self._blocked_contact_truth.tolist(),
                    blocked_encounter_truth=bool(self._blocked_encounter_truth),
                    obstacle_contact_force_truth=self._obstacle_contact_force.tolist())
        return deepcopy(info)

    def _info(self):
        info = self._full_info()
        if not self.config.sensor_only_navigation:
            return info
        for key in ("scenario_parameters", "actual_mass_kg", "blocked_violation", "wheel_traction_force",
                    "wheel_contact_deflection", "wheel_contact_elastic_energy", "native_base_bias",
                    "native_base_applied", "native_base_actuator", "native_base_constraint", "base_acceleration"):
            info.pop(key, None)
        sensed = self._sensor_state
        info.update(sensor_only_navigation=True, sensor_capture_step=int(sensed["sensor_capture_step"]),
            sensor_capture_time=float(sensed["sensor_capture_time"]),
            base_height=float(sensed["base_height"]), base_azimuth=float(sensed["base_azimuth"]),
            base_tilt=np.asarray(sensed["base_tilt"]).tolist(),
            measured_contact=np.asarray(sensed["measured_contact"]).tolist(),
            measured_contact_semantics="analog shaft-load ratio; not friction or geometric support",
            wheel_slip=np.asarray(sensed["slip"]).tolist(), wheel_rates=np.asarray(sensed["wheel_rates"]).tolist(),
            wheel_positions=np.asarray(sensed["wheel_positions"]).tolist(),
            wheel_motor_torque=np.asarray(sensed["wheel_motor_torque"]).tolist(),
            wheel_motor_current=np.asarray(sensed["wheel_motor_current"]).tolist(),
            wheel_motor_current_units="N m torque-equivalent calibrated current proxy",
            wheel_shaft_acceleration=np.asarray(sensed["wheel_shaft_acceleration"]).tolist(),
            contact_tangential_load=np.asarray(sensed["contact_tangential_load"]).tolist(),
            contact_obstacle_load=np.asarray(sensed["contact_obstacle_load"]).tolist(),
            contact_obstacle_load_semantics="equal partition of non-localized current/odometry force residual; not per-wheel contact force",
            estimated_external_vertical_load=float(sensed["estimated_external_vertical_load"]),
            base_acceleration=np.asarray(sensed["base_acceleration"]).tolist(),
            discovered_configuration_cells=deepcopy(self.discovered_configuration_cells))
        return info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=self.config.render_height, width=self.config.render_width)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        if self.render_mode == "human":
            from mujoco import viewer
            if self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self):
        for obj in (self._renderer, self._viewer):
            if obj is not None:
                obj.close()
        self._renderer = self._viewer = None
