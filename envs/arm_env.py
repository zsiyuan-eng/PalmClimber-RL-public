"""
ArmReachEnv - SO-ARM101 manipulation after the climber reaches target height and aligns.

The climber body is fixed at z=2.2m, already rotated to face the leaf target.
The arm must extend its end-effector to the leaf position.

Leaf position is passed in at construction (or reset) so this env can be driven
either standalone or chained after TreeClimberEnv in a sequential task.

Observation (15-dim):
    [0-4]  joint angles q1..q5
    [5-9]  joint velocities dq1..dq5
    [10-12] ee position relative to leaf target (x, y, z)
    [13]   distance to target (scalar, useful for curriculum)
    [14]   gripper opening (normalised 0-1)

Action (6-dim):
    [0-4]  normalised torque for joints 1-5
    [5]    gripper command (+1 open, -1 close)

The gripper auto-closes when ee is within GRASP_DIST of target.
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

SCENE_XML = os.path.join(os.path.dirname(__file__), "..", "mujoco_models", "scene.xml")

# default leaf position if none provided -- just a reasonable demo target
DEFAULT_LEAF_POS = np.array([0.13, 0.0, 2.5])

ARM_BASE_Z        = 2.2 + 0.243   # climber z + mount plate offset
GRASP_DIST        = 0.04          # metres -- auto-close threshold
SUCCESS_DIST      = 0.035         # metres -- task success


class ArmReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, leaf_pos=None, render_mode=None, max_episode_steps=600):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps   = max_episode_steps
        self._step_count = 0

        # leaf_pos can be updated between episodes via reset(options={"leaf_pos": ...})
        self._leaf_pos   = np.array(leaf_pos) if leaf_pos is not None else DEFAULT_LEAF_POS.copy()

        # obs: 5 q + 5 dq + 3 rel_pos + 1 dist + 1 gripper
        obs_high = np.concatenate([
            np.full(5, np.pi),      # joint angles
            np.full(5, 10.0),       # joint velocities
            np.full(3, 1.5),        # relative ee position
            np.array([1.5]),        # distance
            np.array([1.0]),        # gripper
        ]).astype(np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(-np.ones(6, dtype=np.float32),
                                             np.ones(6, dtype=np.float32))

        self._model  = None
        self._data   = None
        self._viewer = None

        # indices filled in _load_model
        self._arm_qposadr    = []
        self._arm_ctrl_start = 6   # first 6 ctrl slots are wheel actuators
        self._ee_id          = -1
        self._gripper_jnt    = -1
        self._free_qposadr   = -1
        self._free_dofadr    = -1

        if _HAS_MUJOCO:
            self._load_model()

    def _load_model(self):
        self._model = mujoco.MjModel.from_xml_path(os.path.abspath(SCENE_XML))
        self._data  = mujoco.MjData(self._model)

        def jnt_qadr(name):
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self._model.jnt_qposadr[jid]

        arm_joints = [
            "joint1_shoulder_pan", "joint2_shoulder_lift",
            "joint3_elbow", "joint4_wrist_flex", "joint5_wrist_roll",
        ]
        self._arm_qposadr = [jnt_qadr(n) for n in arm_joints]
        self._gripper_jnt = jnt_qadr("joint6_gripper")

        self._ee_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        free_jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "climber_free")
        self._free_qposadr = self._model.jnt_qposadr[free_jid]
        self._free_dofadr  = self._model.jnt_dofadr[free_jid]

    def _pin_climber(self, yaw=0.0):
        """Lock climber at target height, facing the correct yaw toward the leaf."""
        adr = self._free_qposadr
        self._data.qpos[adr:adr+3]   = [0.0, 0.0, 2.2]
        # build quaternion from yaw only: q = [cos(yaw/2), 0, 0, sin(yaw/2)]
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        self._data.qpos[adr+3:adr+7] = [cy, 0.0, 0.0, sy]
        dadr = self._free_dofadr
        self._data.qvel[dadr:dadr+6] = 0.0

    def _compute_leaf_yaw(self) -> float:
        """Yaw the climber body needs to point its arm toward the leaf."""
        dx = self._leaf_pos[0]
        dy = self._leaf_pos[1]
        return float(np.arctan2(dy, dx))

    def _get_obs(self) -> np.ndarray:
        if not _HAS_MUJOCO:
            return np.zeros(15, dtype=np.float32)

        q    = np.array([self._data.qpos[a] for a in self._arm_qposadr])
        dq   = np.array([self._data.qvel[a] for a in self._arm_qposadr])

        ee_world = self._data.site_xpos[self._ee_id].copy()
        rel_pos  = ee_world - self._leaf_pos
        dist     = float(np.linalg.norm(rel_pos))

        gripper_open = float(self._data.qpos[self._gripper_jnt]) / 0.025   # normalise 0-1

        return np.concatenate([q, dq, rel_pos,
                                [dist], [gripper_open]]).astype(np.float32)

    def _compute_reward(self, obs, action) -> tuple[float, bool]:
        rel_pos = obs[10:13]
        dist    = float(obs[13])

        r_dist   = -dist                               # dense: closer is better
        r_smooth = -0.005 * float(np.sum(action**2))  # small smoothness penalty
        r_close  = 2.0  if dist < 0.06 else 0.0       # encouragement bonus
        r_grasp  = 10.0 if dist < SUCCESS_DIST else 0.0

        success  = dist < SUCCESS_DIST
        return float(r_dist + r_smooth + r_close + r_grasp), success

    # ─────────────────────────── gym interface ───────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        # allow caller to override leaf position (e.g. when chaining with TreeClimberEnv)
        if options is not None and "leaf_pos" in options:
            self._leaf_pos = np.array(options["leaf_pos"])

        leaf_yaw = self._compute_leaf_yaw()

        if _HAS_MUJOCO:
            mujoco.mj_resetData(self._model, self._data)
            self._pin_climber(yaw=leaf_yaw)
            # random initial arm pose near home
            for adr in self._arm_qposadr:
                self._data.qpos[adr] = np.random.uniform(-0.15, 0.15)
            # open gripper
            self._data.qpos[self._gripper_jnt] = 0.02
            mujoco.mj_forward(self._model, self._data)

        obs  = self._get_obs()
        info = {"leaf_pos": self._leaf_pos.tolist(), "leaf_yaw_deg": float(np.rad2deg(leaf_yaw))}
        return obs, info

    def step(self, action):
        if _HAS_MUJOCO:
            leaf_yaw = self._compute_leaf_yaw()
            self._pin_climber(yaw=leaf_yaw)

            # joint torques for arm
            TORQUE_SCALE = [3.5, 4.0, 3.0, 1.8, 1.2]
            for i, scale in enumerate(TORQUE_SCALE):
                self._data.ctrl[self._arm_ctrl_start + i] = float(action[i]) * scale

            # gripper: action[5] > 0 → open, < 0 → close
            # if near target → auto-close regardless of action
            dist = float(np.linalg.norm(self._data.site_xpos[self._ee_id] - self._leaf_pos))
            if dist < GRASP_DIST:
                self._data.ctrl[self._arm_ctrl_start + 5] = -1.0
            else:
                self._data.ctrl[self._arm_ctrl_start + 5] = float(action[5])

            for _ in range(5):
                mujoco.mj_step(self._model, self._data)
                self._pin_climber(yaw=leaf_yaw)

        obs    = self._get_obs()
        reward, success = self._compute_reward(obs, action)
        self._step_count += 1

        terminated = False
        truncated  = self._step_count >= self.max_steps or success

        info = {
            "dist_to_leaf":  float(obs[13]),
            "success":       success,
            "leaf_pos":      self._leaf_pos.tolist(),
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
            renderer = mujoco.Renderer(self._model, height=480, width=640)
            renderer.update_scene(self._data)
            return renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
