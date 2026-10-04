"""Feedback-only navigation on native tire-contact worlds.

The actor receives 313 sensor and memory values and chooses among 21 waypoint
requests. Obstacle geometry is used only by the simulator world constructor."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.contact_env import TreeWorkContactEnv
from envs.navigation_base import (
    ACTION_NAMES, ACTION_TABLE, ACTION_VECTORS, YAW_LEVELS,
    NavigationBaseConfig, NavigationBaseEnv,
)
from envs.surface_worlds import construct_visible_navigation_world
from mpc.memory import PersistentAttemptMemory


@dataclass
class NavigationConfig(NavigationBaseConfig):
    obstacle_count_max: int = 6
    obstacle_lip_depth: float = .010
    obstacle_facet_max_width: float = .04
    obstacle_front_azimuth: float = -np.pi / 4.

    def __post_init__(self):
        super().__post_init__()
        if self.obstacle_count_max > 6:
            raise ValueError('Visible worlds retain at most six genuine patches')
        if not .002 <= self.obstacle_lip_depth <= .02:
            raise ValueError('lip depth must be between 2 and 20 mm')
        if not .01 <= self.obstacle_facet_max_width <= .08:
            raise ValueError('facet angular width must be between .01 and .08 rad')
        if not np.isfinite(self.obstacle_front_azimuth):
            raise ValueError('Visible-face construction angle must be finite')


class NavigationEnv(NavigationBaseEnv):
    """Sensor navigation using native tire contacts."""

    def __init__(self, config=None, *, complete_task=False,
                 native_step_callback=None, render_mode=None,
                 render_width=960, render_height=720):
        gym.Env.__init__(self)
        self.config = config if isinstance(config, NavigationConfig) else NavigationConfig(**(config or {}))
        self.complete_task = bool(complete_task)
        self.native_step_callback = native_step_callback
        c = self.config
        self.base_env = TreeWorkContactEnv(
            task='combined', controller_mode='adaptive_mpc', config={
                'sensor_only_navigation': True, 'navigation_approach_first': False,
                'sensor_work_safety_enabled': c.work_safety,
                'observation_noise': c.observation_noise,
                'observation_delay_steps': c.observation_delay_steps,
                'horizon': c.horizon, 'solve_interval': c.solve_interval,
                'max_episode_steps': max(1800, c.max_macro_steps*c.native_steps_per_action+1500),
                'friction_range': (.55, .75), 'payload_range': (0., .10),
                'sway_amplitude': 0., 'render_width': render_width, 'render_height': render_height,
            }, contact_config={
                'lip_depth': c.obstacle_lip_depth,
                'angular_facet_max_width': c.obstacle_facet_max_width,
                'max_patches': 32,
            }, render_mode=render_mode)
        self.memory = PersistentAttemptMemory(ACTION_TABLE)
        self.action_names = ACTION_NAMES
        self.action_space = spaces.Discrete(len(ACTION_TABLE))
        self.observation_space = spaces.Box(-np.inf, np.inf,
            shape=(self.memory.observation_size,), dtype=np.float32)
        self._zero_action = np.zeros(12)
        self._previous_action = np.zeros(2)
        self.last_transition = self.last_observation = None
        self.native_lifetime_steps = 0
        self._last_world_certificate = []

    def _sample_world(self):
        c = self.config
        world, certificate = construct_visible_navigation_world(
            self.np_random, difficulty=c.difficulty,
            empty_probability=c.empty_world_probability,
            encounter_probability=c.encounter_world_probability,
            minimum_obstacles=c.obstacle_count_min,
            maximum_obstacles=c.obstacle_count_max,
            front_azimuth=c.obstacle_front_azimuth)
        self._last_world_certificate = certificate
        return world

    def set_curriculum(self, difficulty, empty_probability=None, encounter_probability=None):
        updates = dict(difficulty=difficulty)
        if empty_probability is not None:
            updates['empty_world_probability'] = float(empty_probability)
        if encounter_probability is not None:
            updates['encounter_world_probability'] = float(encounter_probability)
        self.config = NavigationConfig(**{**asdict(self.config), **updates})

    def contract(self):
        result = super().contract()
        result.update(schema='sensor_navigation_v8_native_tire_contacts',
            physical_obstruction='native MuJoCo tire/roller versus static lip facets; no point-spring obstacle proxy',
            obstacle_geometry_reaches_actor=False,
            contact_geometry_reaches_MPC=False,
            obstacle_lip_depth=self.config.obstacle_lip_depth,
            obstacle_facet_max_width=self.config.obstacle_facet_max_width,
            visible_face_world_construction=True)
        return result
