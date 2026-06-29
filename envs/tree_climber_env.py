"""
TreeClimberEnv - MuJoCo Gymnasium environment for the palm tree climbing robot.

Observation space (29-dim):
    [0]   height z (m)
    [1]   vertical velocity dz (m/s)
    [2,3] tilt angles roll, pitch (rad)
    [4,5] tilt rates (rad/s)
    [6]   yaw angle (rad)
    [7]   delta to climb target height (m)
    [8-11] last 4 wheel commands, normalized to [-1, 1]
    [12]  stuck_counter, normalized 0→1  (how long since last height gain)
    [13]  height_gain_rate, rolling mean vertical speed normalized to [-1, 1]
    [14]  azimuth_error to leaf target, normalized -1→1  (yaw_robot - yaw_to_leaf) / π
    [15]  height delta to leaf target (leaf_z - robot_z), clipped to [-2, 2]
    [16-21] current u_MPC, normalized by CONTROL_U_MAX
    [22]  stuck_level, smooth 0→1 authority gate
    [23]  mpc_weight used by the next step()
    [24]  residual_scale / RESIDUAL_SCALE_STUCK
    [25,26] sin/cos of estimated current azimuth around the tree
    [27] azimuth_rate / MAX_AZIMUTH_RATE
    [28] previous magic_lateral_cmd in [-1, 1]

Action space (6-dim, continuous [-1, 1]):
    Residual correction Δu with stuck-gated authority.
    u_final = clip(mpc_weight * u_MPC + residual_scale * action, -CONTROL_U_MAX, CONTROL_U_MAX)

Key behaviors the RL agent learns:
    1. Normal climbing   — uniform positive residual
    2. Stuck recovery    — differential wheel speeds to rotate when stuck_counter rises
    3. Azimuth tracking  — differential wheel speeds to yaw toward the leaf
    The above emerge naturally from the observation; no explicit state machine needed.

Simulation features:
    - Fixed-overfit and randomized local low-friction patch modes. Per-wheel
      friction is kept in info/reward/debug, not in the actor observation.
    - Leaf target placed at random azimuth (0–360°) and random height (1.6–2.4m).
    - Optional domain randomization on base friction and initial pose.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import mujoco
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False
    print("[TreeClimberEnv] mujoco not found – running in stub mode (no physics)")


SCENE_XML = os.path.join(os.path.dirname(__file__), "..", "mujoco_models", "scene.xml")

TARGET_CLIMB_HEIGHT = 2.2   # height the robot needs to reach (m above ground)
START_HEIGHT        = 0.4
TREE_RADIUS         = 0.12  # tree trunk radius (m)
LEAF_RADIAL_OFFSET  = 0.17  # leaf is at this radius from tree centre (just outside bark)

NOMINAL_CLIMB_SPEED = 12.0
CONTROL_U_MAX       = 25.0  # final actuator command limit; MPC uses +/-22 for residual headroom

RESIDUAL_SCALE_NORMAL = 4.0
RESIDUAL_SCALE_STUCK  = 20.0
MPC_WEIGHT_MIN        = 0.2

STUCK_AUTHORITY_TIME     = 2.5   # seconds to ramp stuck authority from 0 to 1
MIN_UPWARD_COMMAND       = 8.0   # rad/s mean |u_mpc| before lack of progress counts as stuck
MIN_HEIGHT_PROGRESS_RATE = 0.01  # m/s, "almost no height gain"
TARGET_HEIGHT_TOL        = 0.05
MAX_HEIGHT_PROGRESS_RATE = 0.25  # m/s normalization for observation

WHEEL_NAMES = [f"wheel{i}" for i in range(1, 7)]
FALLBACK_WHEEL_AZIMUTHS = np.deg2rad([0, 60, 120, 180, 240, 300])
FALLBACK_WHEEL_HEIGHT_OFFSETS = np.array([0.00, 0.22, 0.00, 0.22, 0.00, 0.22])
NOMINAL_PATCH_FRICTION = 0.7

MASS = 3.2
G = 9.81
R_WHEEL = 0.025
NORMAL_FORCE_TOTAL = 50.0
K_SLIP_TRACTION = 60.0
MAX_TRACTION_DV = 0.22

MAX_AZIMUTH_RATE = 1.2  # rad/s
PATCH_ESCAPE_REWARD = 2.0
LATERAL_REASCEND_REWARD = 4.0
FALLEN_PENALTY = -150.0
TOO_LOW_PENALTY = -220.0
EARLY_FAILURE_PENALTY = -80.0
MAX_VERTICAL_SPEED = 0.8
MAX_ANGULAR_SPEED = 4.0

FIXED_OBSTACLE_WHEEL_INDICES = (0, 1)
FIXED_OBSTACLE_ROOT_RISE = 0.35
FIXED_OBSTACLE_HEIGHT_WIDTH = 0.36
FIXED_OBSTACLE_AZIMUTH_WIDTH = np.deg2rad(24.0)
FIXED_OBSTACLE_MU_SCALE = 0.0
MIN_ESCAPE_TURN_RAD = np.deg2rad(12.0)
MIN_POST_ESCAPE_GAIN = 0.15

STUCK_WINDOW        = 30    # steps — rolling window for stuck detection
STUCK_DH_THRESHOLD  = 0.0006 # m/step — below this counts as "not rising" (= 0.04 m/s at 66.7 Hz)
STUCK_U_THRESHOLD   = 8.0   # rad/s — only counts as stuck if wheels are actually commanded
MAX_STUCK_STEPS     = 120   # normalisation denominator for stuck_counter obs


class TreeClimberEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 67}

    def __init__(self, mpc=None, render_mode=None,
                 base_friction_range=(0.65, 1.0),
                 stuck_zone_prob=0.25,
                 obstacle_mode="random",
                 max_tilt_deg=25.0,
                 max_episode_steps=1500):
        super().__init__()
        self.render_mode        = render_mode
        self.mpc                = mpc
        self.base_friction_range = base_friction_range
        self.stuck_zone_prob    = stuck_zone_prob
        if obstacle_mode not in {"none", "random", "fixed"}:
            raise ValueError(f"unsupported obstacle_mode={obstacle_mode!r}")
        self.obstacle_mode      = obstacle_mode
        self.max_tilt_rad       = np.deg2rad(max_tilt_deg)
        self.max_steps          = max_episode_steps

        self._step_count    = 0
        self._height_history = np.zeros(STUCK_WINDOW)  # ring buffer
        self._stuck_counter = 0
        self._stuck_timer   = 0.0
        self._height_gain_rate = 0.0
        self._stuck_level   = 0.0
        self._mpc_weight    = 1.0
        self._residual_scale = RESIDUAL_SCALE_NORMAL
        self._prev_stuck    = False
        self._base_mu       = NOMINAL_PATCH_FRICTION
        self._stuck_patches = []
        self._leaf_pos      = np.zeros(3)  # world position of leaf target
        self._leaf_azimuth  = 0.0
        self._last_ctrl     = np.zeros(6)
        self._cached_u_mpc  = np.zeros(6)
        self._last_delta_u_rl = np.zeros(6)
        self._last_control_debug = {}
        self._recovery_descent_seen = False
        self._lateral_escape_seen = False
        self._wheel_stuck_flags = np.zeros(6, dtype=np.float32)
        self._prev_wheel_stuck_flags = np.zeros(6, dtype=np.float32)
        self._per_wheel_friction = np.full(6, NOMINAL_PATCH_FRICTION, dtype=np.float32)
        self._active_stuck_patch = None
        self._escaped_patch = False
        self._current_azimuth = 0.0
        self._prev_azimuth = 0.0
        self._azimuth_rate = 0.0
        self._magic_lateral_cmd = 0.0
        self._delta_azimuth = 0.0
        self._prev_tilt_mag = 0.0
        self._h_before_step = START_HEIGHT
        self._mpc_pred_dh_prev = 0.0   # prediction from previous solve (for 1-step error)
        self._episode_max_height = START_HEIGHT
        self._episode_net_turn_rad = 0.0
        self._episode_abs_turn_rad = 0.0
        self._episode_max_turn_excursion_rad = 0.0
        self._episode_saw_two_wheel_stuck = False
        self._episode_patch_escape_count = 0
        self._episode_first_escape_height = None
        self._episode_post_escape_max_height = None
        self._fixed_obstacle_entry_yaw = None
        self._fixed_obstacle_cleared_by_turn = False

        # ── observation / action spaces ──────────────────────────────────
        obs_low  = np.array([-5, -5,
                              -np.pi, -np.pi, -20, -20, -np.pi,
                              -3,
                              -1, -1, -1, -1,
                              0.0, -1.0,
                              -1.0, -1.0,
                              -1, -1, -1, -1, -1, -1,
                              0.0, 0.0, 0.0,
                              -1, -1, -1, -1], dtype=np.float32)
        obs_high = np.array([ 5,  5,
                               np.pi,  np.pi,  20,  20,  np.pi,
                               3,
                               1,  1,  1,  1,
                               1.0,  1.0,
                               1.0,  1.0,
                               1, 1, 1, 1, 1, 1,
                               1.0, 1.0, 1.0,
                               1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(-np.ones(6, dtype=np.float32),
                                             np.ones(6, dtype=np.float32))

        self._model  = None
        self._data   = None
        self._viewer = None
        self._renderer = None
        self._trunk_geom_ids = []  # geom ids for friction update
        self._wheel_geom_ids = []

        if _HAS_MUJOCO:
            self._load_model()

    # ─────────────────────────── model loading ──────────────────────────
    def _load_model(self):
        self._model = mujoco.MjModel.from_xml_path(os.path.abspath(SCENE_XML))
        self._data  = mujoco.MjData(self._model)
        # collect trunk geom indices for friction patching
        self._trunk_geom_ids = []
        self._wheel_geom_ids = []
        wheel_geom_names = {f"wheel{wheel}_g" for wheel in range(1, 7)}
        for i in range(self._model.ngeom):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if "trunk" in name:
                self._trunk_geom_ids.append(i)
            if name in wheel_geom_names:
                self._wheel_geom_ids.append(i)
        for gid in self._trunk_geom_ids + self._wheel_geom_ids:
            self._model.geom_condim[gid] = 1

    # ──────────────────────── episode randomisation ─────────────────────
    def _wrap_angle(self, angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _generate_stuck_patches(self):
        """Generate local height/azimuth patches instead of full-ring bands."""
        self._base_mu = float(np.random.uniform(*self.base_friction_range))
        if self.obstacle_mode == "none" or self.stuck_zone_prob <= 0.0:
            return []

        if self.obstacle_mode == "fixed":
            tree_xy = self._get_tree_center_xy()
            wheel_pos = self._get_wheel_world_positions(self._current_azimuth)
            patches = []
            for patch_id, wheel_idx in enumerate(FIXED_OBSTACLE_WHEEL_INDICES):
                wp = wheel_pos[wheel_idx]
                patches.append({
                    "id": patch_id,
                    "group_id": 0,
                    "wheel_index": wheel_idx,
                    "root_height_center": START_HEIGHT + FIXED_OBSTACLE_ROOT_RISE,
                    "root_yaw_center": 0.0,
                    "height_center": float(wp[2] + FIXED_OBSTACLE_ROOT_RISE),
                    "height_width": FIXED_OBSTACLE_HEIGHT_WIDTH,
                    "azimuth_center": float(np.arctan2(wp[1] - tree_xy[1], wp[0] - tree_xy[0])),
                    "azimuth_width": float(FIXED_OBSTACLE_AZIMUTH_WIDTH),
                    "mu_scale": FIXED_OBSTACLE_MU_SCALE,
                    "severity": 1.0,
                })
            return patches

        n_patches = max(1, int(round(3 + 8 * self.stuck_zone_prob)))
        patches = []
        for idx in range(n_patches):
            severity = float(np.random.uniform(0.0, 0.18))
            patches.append({
                "id": idx,
                "height_center": float(np.random.uniform(0.55, TARGET_CLIMB_HEIGHT - 0.1)),
                "height_width": float(np.random.uniform(0.14, 0.24)),
                "azimuth_center": float(np.random.uniform(-np.pi, np.pi)),
                "azimuth_width": float(np.deg2rad(np.random.uniform(30.0, 70.0))),
                "mu_scale": severity,
                "severity": float(1.0 - severity),
            })
        return patches

    def _get_friction_at_height(self, h: float) -> float:
        if len(self._per_wheel_friction) == 0:
            return self._base_mu
        return float(np.mean(self._per_wheel_friction))

    def _apply_friction(self, h: float):
        if not _HAS_MUJOCO:
            return
        # Contact keeps the wheels radially outside the trunk. Tangential grip
        # is supplied exclusively by the per-wheel traction model below, so a
        # local zero-friction patch cannot be bypassed by global trunk friction.
        mu = 0.0
        for gid in self._trunk_geom_ids + self._wheel_geom_ids:
            self._model.geom_friction[gid, :] = [mu, 0.0, 0.0]

    def _generate_leaf_target(self):
        """
        Place the leaf at a random azimuth around the tree at a random height
        in the upper half of the climb range.
        """
        if self.obstacle_mode == "fixed":
            self._leaf_azimuth = 0.0
            leaf_z = TARGET_CLIMB_HEIGHT
        else:
            self._leaf_azimuth = np.random.uniform(0, 2 * np.pi)
            leaf_z = np.random.uniform(1.6, 2.4)
        self._leaf_pos     = np.array([
            LEAF_RADIAL_OFFSET * np.cos(self._leaf_azimuth),
            LEAF_RADIAL_OFFSET * np.sin(self._leaf_azimuth),
            leaf_z,
        ])

    # ──────────────────────────── observation ───────────────────────────
    def _get_robot_pose(self):
        """Return (pos, roll, pitch, yaw, linvel, angvel) from free joint."""
        if not _HAS_MUJOCO:
            return np.zeros(3), 0., 0., 0., np.zeros(3), np.zeros(3)
        d = self._data
        pos  = d.qpos[0:3].copy()
        quat = d.qpos[3:7]          # w x y z
        vel  = d.qvel[0:3].copy()
        angv = d.qvel[3:6].copy()
        w, x, y, z = quat
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return pos, float(roll), float(pitch), float(yaw), vel, angv

    def _compute_azimuth_error(self, yaw: float, robot_xy: np.ndarray) -> float:
        """
        Angular error between where the robot is facing (yaw)
        and the direction from robot to leaf target (projected on XY plane).
        Returns value in [-π, π].
        """
        dx = self._leaf_pos[0] - robot_xy[0]
        dy = self._leaf_pos[1] - robot_xy[1]
        target_yaw = np.arctan2(dy, dx)
        err = target_yaw - yaw
        # wrap to [-π, π]
        err = (err + np.pi) % (2 * np.pi) - np.pi
        return float(err)

    def _quat_from_rpy(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
        cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
        cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], dtype=np.float64)

    def _get_tree_center_xy(self) -> np.ndarray:
        if not _HAS_MUJOCO:
            return np.zeros(2)
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "tree")
        if body_id >= 0:
            return self._data.xpos[body_id, :2].copy()
        for gid in self._trunk_geom_ids:
            return self._data.geom_xpos[gid, :2].copy()
        return np.zeros(2)

    def _get_wheel_world_positions(self, yaw: float) -> np.ndarray:
        if _HAS_MUJOCO:
            positions = []
            for name in WHEEL_NAMES:
                body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id < 0:
                    positions = []
                    break
                positions.append(self._data.xpos[body_id].copy())
            if len(positions) == 6:
                return np.asarray(positions)

        tree_xy = self._get_tree_center_xy()
        root_z = START_HEIGHT
        if _HAS_MUJOCO and self._data is not None:
            root_z = float(self._data.qpos[2])
        radius = TREE_RADIUS + 0.025
        positions = []
        for az, dz in zip(FALLBACK_WHEEL_AZIMUTHS, FALLBACK_WHEEL_HEIGHT_OFFSETS):
            a = yaw + az
            positions.append([
                tree_xy[0] + radius * np.cos(a),
                tree_xy[1] + radius * np.sin(a),
                root_z + dz,
            ])
        return np.asarray(positions, dtype=np.float64)

    def _compute_wheel_patch_state(self):
        pos, roll, pitch, yaw, vel, angvel = self._get_robot_pose()
        tree_xy = self._get_tree_center_xy()
        wheel_pos = self._get_wheel_world_positions(yaw)

        frictions = np.full(6, self._base_mu, dtype=np.float32)
        flags = np.zeros(6, dtype=np.float32)
        active_patch = None

        if self.obstacle_mode == "fixed":
            if self._fixed_obstacle_cleared_by_turn:
                return frictions, flags, active_patch, yaw
            for patch in self._stuck_patches:
                wheel_idx = int(patch["wheel_index"])
                in_height = (
                    abs(float(pos[2]) - patch["root_height_center"])
                    <= patch["height_width"] / 2.0
                )
                yaw_center = (
                    self._fixed_obstacle_entry_yaw
                    if self._fixed_obstacle_entry_yaw is not None
                    else patch["root_yaw_center"]
                )
                in_az = abs(self._wrap_angle(yaw - yaw_center)) <= patch["azimuth_width"] / 2.0
                if in_height and in_az:
                    frictions[wheel_idx] = self._base_mu * patch["mu_scale"]
                    flags[wheel_idx] = 1.0
                    active_patch = patch
            return frictions, flags, active_patch, yaw

        for i, wp in enumerate(wheel_pos):
            wheel_az = np.arctan2(wp[1] - tree_xy[1], wp[0] - tree_xy[0])
            wheel_h = float(wp[2])
            for patch in self._stuck_patches:
                in_height = abs(wheel_h - patch["height_center"]) <= patch["height_width"] / 2.0
                in_az = abs(self._wrap_angle(wheel_az - patch["azimuth_center"])) <= patch["azimuth_width"] / 2.0
                if in_height and in_az:
                    frictions[i] = min(frictions[i], self._base_mu * patch["mu_scale"])
                    flags[i] = 1.0
                    active_patch = patch

        return frictions, flags, active_patch, yaw

    def _update_wheel_patch_state(self):
        self._prev_wheel_stuck_flags = self._wheel_stuck_flags.copy()
        if self.obstacle_mode == "fixed" and self._fixed_obstacle_entry_yaw is None:
            pos, _roll, _pitch, yaw, _vel, _angvel = self._get_robot_pose()
            root_center = START_HEIGHT + FIXED_OBSTACLE_ROOT_RISE
            if abs(float(pos[2]) - root_center) <= FIXED_OBSTACLE_HEIGHT_WIDTH / 2.0:
                self._fixed_obstacle_entry_yaw = yaw
                tree_xy = self._get_tree_center_xy()
                wheel_pos = self._get_wheel_world_positions(yaw)
                for patch in self._stuck_patches:
                    wheel_idx = int(patch["wheel_index"])
                    wp = wheel_pos[wheel_idx]
                    patch["azimuth_center"] = float(
                        np.arctan2(wp[1] - tree_xy[1], wp[0] - tree_xy[0])
                    )
                    patch["root_yaw_center"] = yaw
        frictions, flags, active_patch, yaw = self._compute_wheel_patch_state()
        self._per_wheel_friction = frictions
        self._wheel_stuck_flags = flags
        self._active_stuck_patch = active_patch
        self._prev_azimuth = self._current_azimuth
        self._current_azimuth = self._wrap_angle(yaw)
        if self.obstacle_mode == "fixed" and self._fixed_obstacle_entry_yaw is not None:
            self._fixed_obstacle_cleared_by_turn = bool(
                self._fixed_obstacle_cleared_by_turn
                or abs(self._wrap_angle(yaw - self._fixed_obstacle_entry_yaw))
                > FIXED_OBSTACLE_AZIMUTH_WIDTH / 2.0
            )
        self._escaped_patch = bool(
            np.sum(self._prev_wheel_stuck_flags) > np.sum(flags)
            and (self.obstacle_mode != "fixed" or self._fixed_obstacle_cleared_by_turn)
        )
        if self._escaped_patch:
            self._lateral_escape_seen = True

    def _control_dt(self) -> float:
        if _HAS_MUJOCO and self._model is not None:
            return float(self._model.opt.timestep * 10)
        return 0.02

    def _update_stuck_metrics(self, height: float, u_mean: float, dt: float):
        """
        Maintain height progress metrics. This is intentionally called only
        once per env.step() after physics advances.
        """
        self._height_history = np.roll(self._height_history, -1)
        self._height_history[-1] = height

        gain_per_step = float(np.mean(np.diff(self._height_history)))
        self._height_gain_rate = gain_per_step / max(dt, 1e-6)

        if gain_per_step < STUCK_DH_THRESHOLD and u_mean > STUCK_U_THRESHOLD:
            self._stuck_counter = min(self._stuck_counter + 1, MAX_STUCK_STEPS)
        else:
            self._stuck_counter = max(self._stuck_counter - 2, 0)  # decay faster than accumulate

    def _update_stuck_authority(self, height, target_height, height_gain_rate, u_mpc, dt):
        commanding_up = np.mean(np.abs(u_mpc)) > MIN_UPWARD_COMMAND
        not_near_target = height < target_height - TARGET_HEIGHT_TOL
        low_progress = height_gain_rate < MIN_HEIGHT_PROGRESS_RATE

        if commanding_up and not_near_target and low_progress:
            self._stuck_timer += dt
        else:
            self._stuck_timer = max(0.0, self._stuck_timer - 2.0 * dt)

        stuck_raw = np.clip(self._stuck_timer / STUCK_AUTHORITY_TIME, 0.0, 1.0)
        stuck_level = stuck_raw * stuck_raw * (3.0 - 2.0 * stuck_raw)

        mpc_weight = 1.0 - (1.0 - MPC_WEIGHT_MIN) * stuck_level
        residual_scale = (
            RESIDUAL_SCALE_NORMAL
            + (RESIDUAL_SCALE_STUCK - RESIDUAL_SCALE_NORMAL) * stuck_level
        )

        self._stuck_level = float(stuck_level)
        self._mpc_weight = float(mpc_weight)
        self._residual_scale = float(residual_scale)

    def _update_episode_metrics(self, height: float) -> None:
        self._episode_max_height = max(self._episode_max_height, height)
        actual_delta_azimuth = self._wrap_angle(self._current_azimuth - self._prev_azimuth)
        self._episode_net_turn_rad += actual_delta_azimuth
        self._episode_abs_turn_rad += abs(actual_delta_azimuth)
        self._episode_max_turn_excursion_rad = max(
            self._episode_max_turn_excursion_rad,
            abs(self._episode_net_turn_rad),
        )

        previous_count = int(np.sum(self._prev_wheel_stuck_flags))
        current_count = int(np.sum(self._wheel_stuck_flags))
        if current_count >= 2:
            self._episode_saw_two_wheel_stuck = True

        cleared_now = (
            previous_count > 0
            and current_count == 0
            and (self.obstacle_mode != "fixed" or self._fixed_obstacle_cleared_by_turn)
        )
        if cleared_now:
            self._episode_patch_escape_count += 1
            if self._episode_first_escape_height is None:
                self._episode_first_escape_height = height
                self._episode_post_escape_max_height = height

        if self._episode_first_escape_height is not None:
            self._episode_post_escape_max_height = max(
                self._episode_post_escape_max_height,
                height,
            )

    def _episode_metrics(self, fallen: bool = False) -> dict:
        post_escape_gain = 0.0
        if self._episode_first_escape_height is not None:
            post_escape_gain = (
                self._episode_post_escape_max_height - self._episode_first_escape_height
            )
        escape_success = bool(
            self._episode_saw_two_wheel_stuck
            and self._episode_patch_escape_count > 0
            and self._episode_max_turn_excursion_rad >= MIN_ESCAPE_TURN_RAD
            and post_escape_gain >= MIN_POST_ESCAPE_GAIN
            and not fallen
        )
        return {
            "episode_max_height": float(self._episode_max_height),
            "episode_net_turn_deg": float(np.rad2deg(self._episode_net_turn_rad)),
            "episode_abs_turn_deg": float(np.rad2deg(self._episode_abs_turn_rad)),
            "episode_max_turn_excursion_deg": float(np.rad2deg(self._episode_max_turn_excursion_rad)),
            "episode_patch_escape_count": int(self._episode_patch_escape_count),
            "episode_post_escape_height_gain": float(post_escape_gain),
            "episode_escape_success": escape_success,
        }

    def _build_obs(self):
        """Build the 29-dim policy observation without mutating env state."""
        if not _HAS_MUJOCO:
            return np.zeros(29, dtype=np.float32)

        pos, roll, pitch, yaw, vel, angvel = self._get_robot_pose()
        height = float(pos[2])
        dz     = float(vel[2])
        dh     = TARGET_CLIMB_HEIGHT - height

        az_err  = self._compute_azimuth_error(yaw, pos[:2])
        leaf_dz = float(np.clip(self._leaf_pos[2] - height, -2.0, 2.0))
        u_mpc_norm = np.clip(self._cached_u_mpc / CONTROL_U_MAX, -1.0, 1.0)
        stuck_norm = self._stuck_counter / MAX_STUCK_STEPS
        gain_rate_norm = np.clip(
            self._height_gain_rate / MAX_HEIGHT_PROGRESS_RATE,
            -1.0,
            1.0,
        )

        obs = np.array([
            height, dz,
            roll, pitch, float(angvel[0]), float(angvel[1]),
            yaw,
            float(np.clip(dh, -3, 3)),
            *(self._last_ctrl[:4] / CONTROL_U_MAX),
            stuck_norm,
            gain_rate_norm,
            float(np.clip(az_err / np.pi, -1.0, 1.0)),
            leaf_dz / 2.0,
            *u_mpc_norm,
            self._stuck_level,
            self._mpc_weight,
            self._residual_scale / RESIDUAL_SCALE_STUCK,
            np.sin(self._current_azimuth),
            np.cos(self._current_azimuth),
            np.clip(self._azimuth_rate / MAX_AZIMUTH_RATE, -1.0, 1.0),
            self._magic_lateral_cmd,
        ], dtype=np.float32)
        return obs

    # ───────────────────────── MPC + action ─────────────────────────────
    def _get_mpc_action(self, obs) -> np.ndarray:
        if self.mpc is None:
            return np.full(6, NOMINAL_CLIMB_SPEED)
        state = {
            "height":      float(obs[0]),
            "velocity":    float(obs[1]),
            "tilt_x":      float(obs[2]),
            "tilt_y":      float(obs[3]),
            "tilt_rate_x": float(obs[4]),
            "tilt_rate_y": float(obs[5]),
            "yaw":         float(obs[6]),
        }
        return self.mpc.solve(state, target_height=TARGET_CLIMB_HEIGHT)

    def _apply_pseudo_traction(self, u_final: np.ndarray, dt: float) -> dict:
        if not _HAS_MUJOCO:
            return {"traction_force_up": 0.0, "traction_dv": 0.0, "effective_wheel_friction": self._per_wheel_friction.copy()}

        dz = float(self._data.qvel[2])
        v_wheel = R_WHEEL * np.asarray(u_final, dtype=np.float64)
        slip = v_wheel - dz
        f_raw = K_SLIP_TRACTION * slip
        f_max = np.maximum(self._per_wheel_friction, 0.0) * (NORMAL_FORCE_TOTAL / 6.0)
        f_i = f_max * np.tanh(f_raw / np.maximum(f_max, 1e-6))
        f_up = float(np.sum(f_i))
        # MuJoCo integrates gravity and joint damping during mj_step(). This
        # correction injects only wheel traction, avoiding double gravity.
        traction_dv = float(np.clip((f_up / MASS) * dt, -MAX_TRACTION_DV, MAX_TRACTION_DV))
        self._data.qvel[2] += traction_dv
        return {
            "traction_force_up": f_up,
            "traction_dv": traction_dv,
            "effective_wheel_friction": self._per_wheel_friction.copy(),
        }

    def _compute_magic_lateral_cmd(self, u_final: np.ndarray) -> tuple[float, float, float]:
        u = np.asarray(u_final, dtype=np.float64)
        alternating = np.mean(np.array([u[0], -u[1], u[2], -u[3], u[4], -u[5]]))
        differential = np.mean(u[[0, 2, 4]]) - np.mean(u[[1, 3, 5]])
        alternating_cmd = float(np.clip(alternating / CONTROL_U_MAX, -1.0, 1.0))
        differential_cmd = float(np.clip(differential / CONTROL_U_MAX, -1.0, 1.0))
        magic_cmd = float(np.clip(0.6 * alternating_cmd + 0.4 * differential_cmd, -1.0, 1.0))
        return magic_cmd, alternating_cmd, differential_cmd

    def _apply_magic_lateral(self, u_final: np.ndarray, dt: float) -> dict:
        magic_cmd, alternating_cmd, differential_cmd = self._compute_magic_lateral_cmd(u_final)
        lateral_gate = 0.3 + 0.7 * self._stuck_level
        azimuth_rate = MAX_AZIMUTH_RATE * magic_cmd * lateral_gate
        delta_azimuth = float(np.clip(azimuth_rate * dt, -0.08, 0.08))

        if _HAS_MUJOCO and abs(delta_azimuth) > 1e-8:
            pos, roll, pitch, yaw, vel, angvel = self._get_robot_pose()
            tree_xy = self._get_tree_center_xy()

            rel = self._data.qpos[0:2].copy() - tree_xy
            radius = float(np.linalg.norm(rel))
            if radius > 0.02:
                old_angle = np.arctan2(rel[1], rel[0])
                new_angle = old_angle + delta_azimuth
                self._data.qpos[0] = tree_xy[0] + radius * np.cos(new_angle)
                self._data.qpos[1] = tree_xy[1] + radius * np.sin(new_angle)

            new_yaw = self._wrap_angle(yaw + delta_azimuth)
            self._data.qpos[3:7] = self._quat_from_rpy(roll, pitch, new_yaw)
            self._data.qvel[5] = azimuth_rate
            mujoco.mj_forward(self._model, self._data)

        self._magic_lateral_cmd = magic_cmd
        self._delta_azimuth = delta_azimuth
        self._azimuth_rate = float(azimuth_rate)
        self._current_azimuth = self._wrap_angle(self._current_azimuth + delta_azimuth)
        return {
            "magic_lateral_cmd": magic_cmd,
            "alternating_cmd": alternating_cmd,
            "differential_cmd": differential_cmd,
            "delta_azimuth": delta_azimuth,
            "azimuth_rate": float(azimuth_rate),
        }

    def _apply_action(self, action, u_mpc) -> tuple[np.ndarray, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        mpc_weight = self._mpc_weight
        residual_scale = self._residual_scale
        delta_u_rl = residual_scale * action
        u_final = np.clip(
            mpc_weight * u_mpc + delta_u_rl,
            -CONTROL_U_MAX,
            CONTROL_U_MAX,
        )
        if _HAS_MUJOCO:
            # In scene.xml the positive hinge direction moves each inner wheel
            # surface upward, which pushes the chassis downward. Keep the
            # controller convention positive=climb by flipping at the actuator.
            self._data.ctrl[:6] = -u_final
        self._last_ctrl = u_final.copy()
        self._last_delta_u_rl = delta_u_rl.copy()

        debug = {
            "stuck_level": self._stuck_level,
            "mpc_weight": mpc_weight,
            "residual_scale": residual_scale,
            "u_mpc": u_mpc.copy(),
            "delta_u_rl": delta_u_rl.copy(),
            "u_final": u_final.copy(),
        }
        self._last_control_debug = debug
        return u_final, debug

    # ───────────────────────────── reward ───────────────────────────────
    def _compute_reward(self, obs, u_final, was_stuck: bool,
                        fallen: bool = False, too_low: bool = False) -> tuple[float, dict]:
        height      = float(obs[0])
        dz          = float(obs[1])
        roll        = float(obs[2])
        pitch       = float(obs[3])
        az_err_norm = float(obs[14])   # in [-1, 1]
        az_err_rad  = az_err_norm * np.pi
        stuck_level = float(obs[22])

        dh_to_climb = TARGET_CLIMB_HEIGHT - height
        dh_to_leaf  = abs(float(obs[15]) * 2.0)   # unnormalize

        # ── base climbing ──
        upward_scale = 1.0
        height_scale = 1.0 - 0.35 * stuck_level
        stable_upright = abs(roll) < self.max_tilt_rad * 0.85 and abs(pitch) < self.max_tilt_rad * 0.85
        r_climb  =  2.0 * max(dz, 0) * upward_scale if stable_upright else 0.0
        r_pot    = -0.4 * dh_to_climb**2 * height_scale      # potential toward target
        r_tilt   = -6.0 * (roll**2 + pitch**2)               # stability
        r_energy = -0.002 * float(np.sum(u_final**2))

        # ── stuck-specific shaping ──
        low_progress = self._height_gain_rate < MIN_HEIGHT_PROGRESS_RATE
        r_stuck_penalty = -0.20 * stuck_level if low_progress else 0.0

        controlled_descent = was_stuck and (-0.15 < dz < -0.005)
        r_descent_escape = 0.45 * stuck_level if controlled_descent else 0.0
        if controlled_descent:
            self._recovery_descent_seen = True

        r_drop_penalty = -2.0 * stuck_level * abs(dz) if dz < -0.30 else 0.0
        tilt_mag = float(np.hypot(roll, pitch))
        tilt_not_worse = tilt_mag <= self._prev_tilt_mag + 0.05
        no_big_drop = dz > -0.20
        stuck_count_reduced = np.sum(self._wheel_stuck_flags) < np.sum(self._prev_wheel_stuck_flags)
        lateral_moved = abs(self._delta_azimuth) > np.deg2rad(1.0) or abs(self._magic_lateral_cmd) > 0.25

        r_lateral_escape = 0.0
        if was_stuck and lateral_moved and tilt_not_worse and no_big_drop and (stuck_count_reduced or self._escaped_patch):
            r_lateral_escape = 0.8 * max(stuck_level, 0.3)

        r_patch_escape = PATCH_ESCAPE_REWARD if (was_stuck and self._escaped_patch and no_big_drop) else 0.0
        if r_patch_escape > 0:
            self._lateral_escape_seen = True

        # ── stuck recovery bonus ──
        # if we were stuck last step but now gaining height → reward escaping the groove
        now_gaining = self._height_gain_rate > MIN_HEIGHT_PROGRESS_RATE * 2.0
        r_recovery  = 3.0 if (was_stuck and now_gaining) else 0.0
        if self._recovery_descent_seen and now_gaining:
            r_recovery += 2.0
            self._recovery_descent_seen = False
        if self._lateral_escape_seen and now_gaining:
            r_recovery += LATERAL_REASCEND_REWARD
            self._lateral_escape_seen = False

        # ── azimuth alignment reward (only when near leaf height) ──
        # once the robot is within 0.4m height of the leaf, it should rotate to face it
        near_leaf_height = dh_to_leaf < 0.4
        if near_leaf_height:
            r_align = -1.5 * az_err_rad**2   # penalise misalignment when close
            r_align_bonus = 1.5 if abs(az_err_rad) < np.deg2rad(15) else 0.0
        else:
            r_align = 0.0
            r_align_bonus = 0.0

        # ── goal bonuses ──
        r_reach_height = 5.0 if dh_to_climb < 0.05 else 0.0
        r_reach_aligned = 8.0 if (dh_to_climb < 0.05 and abs(az_err_rad) < np.deg2rad(20)) else 0.0
        early_failure = self._step_count < 60 and (fallen or too_low)
        r_safety_stop = 0.0
        if fallen:
            r_safety_stop += FALLEN_PENALTY
        if too_low:
            r_safety_stop += TOO_LOW_PENALTY
        if early_failure:
            r_safety_stop += EARLY_FAILURE_PENALTY

        components = {
            "reward_upward": float(r_climb),
            "reward_height": float(r_pot),
            "reward_tilt": float(r_tilt),
            "reward_energy": float(r_energy),
            "reward_stuck_penalty": float(r_stuck_penalty),
            "reward_descent_escape": float(r_descent_escape + r_drop_penalty),
            "reward_lateral_escape": float(r_lateral_escape),
            "reward_patch_escape": float(r_patch_escape),
            "reward_recovery": float(r_recovery),
            "reward_alignment": float(r_align + r_align_bonus),
            "reward_success": float(r_reach_height + r_reach_aligned),
            "reward_safety_stop": float(r_safety_stop),
        }
        self._prev_tilt_mag = tilt_mag
        total = float(sum(components.values()))
        return total, components

    # ──────────────────────── gymnasium interface ────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count     = 0
        self._stuck_counter  = 0
        self._stuck_timer    = 0.0
        self._height_gain_rate = 0.0
        self._stuck_level    = 0.0
        self._mpc_weight     = 1.0
        self._residual_scale = RESIDUAL_SCALE_NORMAL
        self._prev_stuck     = False
        self._height_history = np.full(STUCK_WINDOW, START_HEIGHT, dtype=np.float64)
        self._last_ctrl      = np.zeros(6)
        self._cached_u_mpc   = np.zeros(6)
        self._last_delta_u_rl = np.zeros(6)
        self._last_control_debug = {}
        self._recovery_descent_seen = False
        self._lateral_escape_seen = False
        self._wheel_stuck_flags = np.zeros(6, dtype=np.float32)
        self._prev_wheel_stuck_flags = np.zeros(6, dtype=np.float32)
        self._per_wheel_friction = np.full(6, NOMINAL_PATCH_FRICTION, dtype=np.float32)
        self._active_stuck_patch = None
        self._escaped_patch = False
        self._current_azimuth = 0.0
        self._prev_azimuth = 0.0
        self._azimuth_rate = 0.0
        self._magic_lateral_cmd = 0.0
        self._delta_azimuth = 0.0
        self._prev_tilt_mag = 0.0
        self._h_before_step = START_HEIGHT
        self._mpc_pred_dh_prev = 0.0
        self._episode_max_height = START_HEIGHT
        self._episode_net_turn_rad = 0.0
        self._episode_abs_turn_rad = 0.0
        self._episode_max_turn_excursion_rad = 0.0
        self._episode_saw_two_wheel_stuck = False
        self._episode_patch_escape_count = 0
        self._episode_first_escape_height = None
        self._episode_post_escape_max_height = None
        self._fixed_obstacle_entry_yaw = None
        self._fixed_obstacle_cleared_by_turn = False

        self._stuck_patches = []
        self._generate_leaf_target()

        if _HAS_MUJOCO:
            mujoco.mj_resetData(self._model, self._data)
            # slight random initial pose; keep quaternion normalized to avoid
            # injecting artificial angular energy at reset.
            if self.obstacle_mode == "fixed":
                self._data.qpos[0:3] = [0.0, 0.0, START_HEIGHT]
                self._data.qpos[3:7] = self._quat_from_rpy(0.0, 0.0, 0.0)
            else:
                self._data.qpos[0] = np.random.uniform(-0.01, 0.01)
                self._data.qpos[1] = np.random.uniform(-0.01, 0.01)
                self._data.qpos[2] = START_HEIGHT + np.random.uniform(-0.03, 0.03)
                self._data.qpos[3:7] = self._quat_from_rpy(
                    np.random.uniform(-0.015, 0.015),
                    np.random.uniform(-0.015, 0.015),
                    np.random.uniform(-0.015, 0.015),
                )
            mujoco.mj_forward(self._model, self._data)
            self._stuck_patches = self._generate_stuck_patches()
            self._apply_friction(START_HEIGHT)
            mujoco.mj_forward(self._model, self._data)
            self._height_history.fill(float(self._data.qpos[2]))
            self._update_wheel_patch_state()
        else:
            self._stuck_patches = self._generate_stuck_patches()

        self._episode_max_height = float(self._data.qpos[2]) if _HAS_MUJOCO else START_HEIGHT

        obs_for_mpc = self._build_obs()
        self._cached_u_mpc = self._get_mpc_action(obs_for_mpc)
        obs  = self._build_obs()
        self._h_before_step = float(obs[0])
        initial_mpc_diag = self.mpc.get_diagnostics() if self.mpc is not None else {}
        self._mpc_pred_dh_prev = initial_mpc_diag.get("mpc_pred_dh_1step", 0.0)
        info = {
            "stuck_patches": self._stuck_patches,
            "leaf_pos":      self._leaf_pos.tolist(),
            "leaf_azimuth":  float(np.rad2deg(self._leaf_azimuth)),
            "u_mpc":         self._cached_u_mpc.copy(),
            "stuck_level":   self._stuck_level,
            "mpc_weight":    self._mpc_weight,
            "residual_scale": self._residual_scale,
            "wheel_stuck_flags": self._wheel_stuck_flags.copy(),
            "per_wheel_friction": self._per_wheel_friction.copy(),
            "current_azimuth": self._current_azimuth,
            "obstacle_mode": self.obstacle_mode,
            **self._episode_metrics(),
        }
        return obs, info

    def step(self, action):
        was_stuck = self._stuck_level > 0.25 or self._stuck_counter > 20
        dt = self._control_dt()
        h_before_step = float(self._data.qpos[2]) if _HAS_MUJOCO else self._h_before_step

        u_mpc = self._cached_u_mpc.copy()
        if u_mpc.shape != (6,):
            u_mpc = self._get_mpc_action(self._build_obs())
        u_final, control_debug = self._apply_action(action, u_mpc)

        if _HAS_MUJOCO:
            height = float(self._data.qpos[2])
            self._apply_friction(height)   # update friction for current height zone
            lateral_debug = self._apply_magic_lateral(u_final, dt)
            for _ in range(10):
                mujoco.mj_step(self._model, self._data)
            # Apply the traction impulse after MuJoCo integrates gravity. The
            # corrected velocity is then integrated naturally next control step.
            traction_debug = self._apply_pseudo_traction(u_final, dt)
        else:
            traction_debug = {"traction_force_up": 0.0, "traction_dv": 0.0, "effective_wheel_friction": self._per_wheel_friction.copy()}
            lateral_debug = {"magic_lateral_cmd": 0.0, "alternating_cmd": 0.0, "differential_cmd": 0.0, "delta_azimuth": 0.0, "azimuth_rate": 0.0}

        if _HAS_MUJOCO:
            height = float(self._data.qpos[2])
            self._update_wheel_patch_state()
        else:
            height = 0.0

        self._update_episode_metrics(height)

        self._update_stuck_metrics(height, float(np.mean(np.abs(u_final))), dt)
        self._update_stuck_authority(
            height,
            TARGET_CLIMB_HEIGHT,
            self._height_gain_rate,
            u_mpc,
            dt,
        )

        obs_for_next_mpc = self._build_obs()
        self._cached_u_mpc = self._get_mpc_action(obs_for_next_mpc)
        obs = self._build_obs()

        actual_dh_1step = float(obs[0]) - h_before_step
        mpc_pred_dh_used_1step = float(self._mpc_pred_dh_prev)
        mpc_diag = self.mpc.get_diagnostics() if self.mpc is not None else {}
        mpc_dh_error_1step = actual_dh_1step - mpc_pred_dh_used_1step
        self._mpc_pred_dh_prev = mpc_diag.get("mpc_pred_dh_1step", 0.0)
        self._h_before_step = float(obs[0])

        height   = float(obs[0])
        roll     = float(obs[2])
        pitch    = float(obs[3])
        az_err   = float(obs[14]) * np.pi

        fallen   = abs(roll) > self.max_tilt_rad or abs(pitch) > self.max_tilt_rad
        too_low  = height < 0.1
        reached  = (height >= TARGET_CLIMB_HEIGHT - 0.03
                    and abs(az_err) < np.deg2rad(25))
        reward, reward_components = self._compute_reward(
            obs,
            u_final,
            was_stuck,
            fallen=fallen,
            too_low=too_low,
        )
        self._step_count += 1

        if _HAS_MUJOCO:
            self._data.qvel[0:3] = np.clip(self._data.qvel[0:3], -MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED)
            self._data.qvel[3:6] = np.clip(self._data.qvel[3:6], -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

        terminated = fallen or too_low
        truncated  = self._step_count >= self.max_steps or reached

        info = {
            "height":           height,
            "stuck_patches":    self._stuck_patches,
            "stuck_counter":    self._stuck_counter,
            "azimuth_error_deg": float(np.rad2deg(abs(az_err))),
            "leaf_pos":         self._leaf_pos.tolist(),
            "friction_now":     self._get_friction_at_height(height),
            "per_wheel_friction": self._per_wheel_friction.copy(),
            "wheel_stuck_flags": self._wheel_stuck_flags.copy(),
            "active_stuck_patch": self._active_stuck_patch,
            "escaped_patch": self._escaped_patch,
            "current_azimuth": self._current_azimuth,
            "height_gain_rate":  self._height_gain_rate,
            **control_debug,
            **traction_debug,
            **lateral_debug,
            **reward_components,
            **mpc_diag,
            "actual_dh_1step":        actual_dh_1step,
            "mpc_pred_dh_used_1step": mpc_pred_dh_used_1step,
            "mpc_dh_error_1step":     mpc_dh_error_1step,
            "fallen":           fallen,
            "reached_target":   reached,
            "obstacle_mode":    self.obstacle_mode,
            **self._episode_metrics(fallen=fallen),
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if not _HAS_MUJOCO:
            return None
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._model, height=480, width=640)
            self._renderer.update_scene(self._data)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
