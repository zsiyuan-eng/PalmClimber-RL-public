"""Learned waypoint navigation using measured progress and failure memory."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
from typing import Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.common import wrap_angle
from envs.tree_work_env import TreeWorkEnv
from mpc.memory import PersistentAttemptMemory


YAW_LEVELS = (.07, .20, .42)
ACTION_NAMES = ['hold', 'climb_to_task_height', 'retreat']
ACTION_TABLE = [[0., 0.], [1., 0.], [-1., 0.]]
for amplitude, label in zip(YAW_LEVELS, ('small', 'medium', 'large')):
    for sign, direction in ((-1., 'negative'), (1., 'positive')):
        for height, vertical in ((0., 'turn'), (1., 'climb'), (-1., 'retreat')):
            ACTION_NAMES.append(f'{vertical}_{direction}_{label}')
            ACTION_TABLE.append([height, sign * amplitude])
ACTION_NAMES = tuple(ACTION_NAMES)
ACTION_TABLE = np.asarray(ACTION_TABLE, dtype=np.float64)
# Normalized command requests; yaw waypoints remain in radians.
ACTION_VECTORS = ACTION_TABLE / np.array([1., max(YAW_LEVELS)])


@dataclass
class NavigationBaseConfig:
    native_steps_per_action: int = 8
    height_step: float = .08
    max_macro_steps: int = 220
    difficulty: int = 3
    empty_world_probability: float = .10
    encounter_world_probability: float = .85
    observation_noise: float = .0003
    observation_delay_steps: int = 0
    horizon: int = 4
    solve_interval: int = 3
    obstacle_count_min: int = 3
    obstacle_count_max: int = 7
    work_safety: bool = True
    time_penalty_per_second: float = .25
    repeated_failed_action_penalty: float = .30
    no_progress_penalty: float = .012
    direction_reversal_penalty: float = .05
    yaw_travel_penalty: float = .025
    arrival_reward: float = 25.
    unsafe_penalty: float = 20.

    def __post_init__(self):
        for key in ('native_steps_per_action', 'max_macro_steps', 'horizon', 'solve_interval',
                    'obstacle_count_min', 'obstacle_count_max'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.horizon < 2 or not 0 < self.height_step <= .10:
            raise ValueError('requires a predictive horizon and bounded height requests')
        if not isinstance(self.difficulty, int) or not 0 <= self.difficulty <= 3:
            raise ValueError('Curriculum difficulty must be an integer from zero to three')
        if not 1 <= self.obstacle_count_min <= self.obstacle_count_max <= 9:
            raise ValueError('obstacle count must be between one and nine')
        for key in ('empty_world_probability', 'encounter_world_probability'):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f'{key} must be a probability')
        if self.observation_noise < 0 or self.observation_delay_steps < 0:
            raise ValueError('Sensor uncertainty cannot be negative')
        for key in ('time_penalty_per_second', 'repeated_failed_action_penalty',
                    'no_progress_penalty', 'direction_reversal_penalty',
                    'yaw_travel_penalty', 'arrival_reward', 'unsafe_penalty'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError('Reward weights must be finite and nonnegative')


def construct_navigation_world(rng, *, difficulty=3, empty_probability=.10,
                               encounter_probability=.85, minimum_obstacles=3,
                               maximum_obstacles=7):
    """Generate feasible multi-row worlds for the six-contact footprint.

    Feasibility is construction-only. Angles alias every 60 degrees because
    all six wheels must clear a patch. Patches in each row form a cluster in
    this periodic configuration space, leaving a wide angular channel. Clear
    height bands permit changing channels between rows. The certificate is
    returned separately and is never passed to the actor or native state.
    """
    start = float(rng.uniform(.48, .54))
    goal = float(rng.uniform(1.45, 1.60))
    theta = float(rng.uniform(-.22, .22))
    target_theta = theta + float(rng.uniform(-.035, .035))
    period = np.pi / 3.
    terrain, certificate = [], []
    if difficulty and rng.random() >= empty_probability:
        if difficulty == 1:
            count, rows = 1, 1
        elif difficulty == 2:
            count, rows = int(rng.integers(2, 5)), 2
        else:
            count = int(rng.integers(minimum_obstacles, maximum_obstacles + 1))
            rows = min(3, count)
        allocations = np.full(rows, count // rows, dtype=int)
        allocations[:count % rows] += 1
        heights = np.linspace(start + .22, goal - .22, rows) if rows > 1 else [float(rng.uniform(start + .24, goal - .24))]
        encounter = bool(rng.random() < encounter_probability)
        sign = float(rng.choice([-1., 1.]))
        base_phase = theta if encounter else float(rng.uniform(-period/2, period/2))
        phases = [base_phase, base_phase + sign*.29, base_phase - sign*.18]
        for row, (height, n) in enumerate(zip(heights, allocations)):
            phase = phases[row] + float(rng.uniform(-.025, .025))
            offsets = np.linspace(-.17, .17, n) if n > 1 else [0.]
            max_extent = 0.
            row_indices = []
            for offset in offsets:
                width = float(rng.uniform(.045, .065))
                hwidth = float(rng.uniform(.025, .035))
                # Real patches occupy different physical sides of the tree;
                # their aliases share one blocked configuration cluster.
                wheel_index = int(rng.integers(0, 6))
                terrain.append(dict(kind='blocked', theta=float(wrap_angle(phase + offset + wheel_index*period)),
                                    height=float(height), theta_half_width=width,
                                    height_half_width=hwidth, support_fraction=0.))
                max_extent = max(max_extent, abs(float(offset)) + width + .08 + .02)
                row_indices.append(len(terrain)-1)
            certificate.append(dict(row=row, height=float(height),
                                    safe_configuration_azimuth=float(wrap_angle(phase + period/2)),
                                    safe_channel_half_width=float(period/2-max_extent),
                                    vertical_half_extent=.035 + .035 + .15*np.sin(.205),
                                    patch_indices=row_indices))
    world = dict(start_height=start, start_azimuth=theta, goal_height=goal, goal_azimuth=target_theta,
                 terrain_map=terrain, payload_mass=float(rng.uniform(0., .055)), target_radius=.60,
                 friction_range=(.60, .70), sway_amplitude=0.)
    return world, certificate


class NavigationBaseEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 25}

    def __init__(self, config=None, *, complete_task=False, native_step_callback: Callable | None = None,
                 render_mode=None, render_width=960, render_height=720):
        super().__init__()
        self.config = config if isinstance(config, NavigationBaseConfig) else NavigationBaseConfig(**(config or {}))
        self.complete_task = bool(complete_task)
        self.native_step_callback = native_step_callback
        c = self.config
        self.base_env = TreeWorkEnv(task='combined', controller_mode='adaptive_mpc', config={
            'sensor_only_navigation': True, 'navigation_approach_first': False,
            'sensor_work_safety_enabled': c.work_safety,
            'observation_noise': c.observation_noise, 'observation_delay_steps': c.observation_delay_steps,
            'horizon': c.horizon, 'solve_interval': c.solve_interval,
            'max_episode_steps': max(1800, c.max_macro_steps*c.native_steps_per_action+1500),
            'friction_range': (.55, .75), 'payload_range': (0., .10), 'sway_amplitude': 0.,
            'render_width': render_width, 'render_height': render_height,
        }, render_mode=render_mode)
        self.memory = PersistentAttemptMemory(ACTION_TABLE)
        self.action_names = ACTION_NAMES
        self.action_space = spaces.Discrete(len(ACTION_TABLE))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.memory.observation_size,), dtype=np.float32)
        self._zero_action = np.zeros(12)
        self._previous_action = np.zeros(2)
        self.last_transition = self.last_observation = None
        self.native_lifetime_steps = 0
        self._last_world_certificate = []

    def _sample_world(self):
        world, certificate = construct_navigation_world(self.np_random, difficulty=self.config.difficulty,
            empty_probability=self.config.empty_world_probability,
            encounter_probability=self.config.encounter_world_probability,
            minimum_obstacles=self.config.obstacle_count_min, maximum_obstacles=self.config.obstacle_count_max)
        self._last_world_certificate = certificate
        return world

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._last_world_certificate = []
        # Draw the plant seed before construction. Replaying the same wrapper
        # seed with saved reset options must retain friction/noise exactly.
        plant_seed = int(self.np_random.integers(0, 2**31))
        world = self._sample_world() if options is None else dict(options)
        self._last_reset_options = deepcopy(world)  # privileged external logging only
        _, info = self.base_env.reset(seed=plant_seed, options=world)
        self.goal = np.array([self.base_env.goal_height, self.base_env.goal_azimuth])
        state = self.base_env.get_control_state()
        self.memory.reset(state, self.goal)
        self._previous_action = np.zeros(2)
        self._macro_steps = 0
        self._navigation_arrived = False
        self.last_transition = None
        self._reward_components = {}
        self.last_observation = self.memory.observation(self.goal, self._previous_action)
        return self.last_observation.copy(), self._public_info(info)

    def _waypoint(self, state, action_row):
        current = np.array([state['base_height'], state['base_azimuth']], dtype=float)
        height_mode, yaw = np.asarray(action_row, dtype=float)
        error = self.goal - current
        error[1] = float(wrap_angle(error[1]))
        waypoint = current + [height_mode*self.config.height_step, yaw]
        if height_mode > 0:
            waypoint[0] = current[0] + np.clip(error[0], -self.config.height_step, self.config.height_step)
        elif height_mode == 0 and yaw == 0 and np.all(np.abs(error) <= [self.config.height_step, .42]):
            waypoint = current + error
        if abs(error[1]) > self.base_env.config.azimuth_tolerance and error[1]*yaw > 0 and abs(yaw) > abs(error[1]):
            waypoint[1] = current[1] + error[1]
        waypoint[0] = np.clip(waypoint[0], self.base_env.config.min_height+.04, self.base_env.config.max_height-.04)
        return waypoint

    def _public_info(self, native_info):
        return dict(success=bool(self._navigation_arrived), navigation_success=bool(self._navigation_arrived),
                    full_task_success=bool(native_info.get('success', False)),
                    unsafe=bool(native_info.get('unsafe', False)), fallen=bool(native_info.get('fallen', False)),
                    stage=native_info['stage'], native_steps=native_info['steps'], elapsed_time=native_info['elapsed_time'],
                    navigation_diagnostics=self.memory.diagnostics(),
                    control_diagnostics=native_info.get('controller_diagnostics', {}),
                    reward_components=dict(self._reward_components))

    def step(self, action):
        if self.last_observation is None:
            raise RuntimeError('reset must precede navigation step')
        value = np.asarray(action)
        if value.size != 1 or not np.isfinite(value).all() or float(value.item()) != int(value.item()):
            raise ValueError('navigation action must be one integer')
        index = int(value.item())
        if not self.action_space.contains(index):
            raise ValueError('navigation action is outside the learned action space')
        before = self.base_env.get_control_state()
        active = self.base_env.stage == 'navigate'
        observation_pre = self.last_observation.copy()
        requested = ACTION_TABLE[index].copy()
        waypoint = self._waypoint(before, requested)
        delta = waypoint - [before['base_height'], before['base_azimuth']]
        delta[1] = float(wrap_angle(delta[1]))
        vector = np.clip(delta / [self.config.height_step, .42], -1., 1.)
        known_failure = float(self.memory.action_profiles(before)[index, 0])
        if active:
            self.base_env.set_navigation_reference(*waypoint)
        else:
            vector[:] = delta[:] = 0.
            waypoint = np.array([before['base_height'], before['base_azimuth']])
        command = dict(issued_time=float(before['time']), waypoint=waypoint, action=vector)
        self._macro_steps += 1
        term = trunc = arrived = False
        for substep in range(self.config.native_steps_per_action):
            _, _, term, trunc, info = self.base_env.step(self._zero_action)
            self.native_lifetime_steps += 1
            sensed = self.base_env.get_control_state()
            self.memory.update(sensed, command)
            if not self._navigation_arrived and self.base_env.stage != 'navigate':
                self._navigation_arrived = arrived = True
            if active and (substep == self.config.native_steps_per_action-1 or term or trunc
                           or (self._navigation_arrived and not self.complete_task)):
                self.memory.record_attempt(before, sensed, index, delta)
            record = dict(macro_step=self._macro_steps, native_substep=substep, native_step=self.base_env._steps,
                observation_pre=observation_pre.copy(), actor_active_at_issue=active,
                action_discrete=index, requested_action_vector=ACTION_VECTORS[index].copy(),
                requested_yaw_delta_rad=float(requested[1]),
                yaw_amplitude_label=('zero' if requested[1] == 0 else ('small', 'medium', 'large')[YAW_LEVELS.index(abs(float(requested[1])))]),
                native_steps_requested=self.config.native_steps_per_action,
                repeat_failed_action_confidence=known_failure,
                action_vector=vector.copy(), waypoint=waypoint.copy(), sensor_state_post=sensed,
                navigation_diagnostics=self.memory.diagnostics(), info=self._public_info(info))
            self.last_transition = record
            if self.native_step_callback is not None:
                self.native_step_callback(self.base_env, record)
            if term or trunc or (self._navigation_arrived and not self.complete_task):
                break
        state = self.base_env.get_control_state()
        diagnostics = self.memory.attempt_diagnostics()
        elapsed = max(0., float(state['time']) - float(before['time']))
        no_progress = diagnostics['best_height_gain'] <= .0015 and diagnostics['best_task_distance_gain'] <= .002
        measured_yaw_travel = abs(float(wrap_angle(state['base_azimuth']-before['base_azimuth'])))
        slip = float(np.mean(np.abs(np.asarray(state.get('slip', np.zeros(6))))))
        load = float(np.max(np.abs(np.asarray(state.get('contact_obstacle_load', np.zeros(6))))))
        c = self.config
        self._reward_components = dict(
            frontier_height=16.*diagnostics['best_height_gain']*float(active),
            frontier_task_distance=8.*diagnostics['best_task_distance_gain']*float(active),
            elapsed_time=-c.time_penalty_per_second*elapsed,
            known_failed_request=-c.repeated_failed_action_penalty*known_failure*float(no_progress),
            no_progress=-c.no_progress_penalty*min(diagnostics['consecutive_no_progress_attempts'], 10)*float(no_progress),
            direction_reversal=-c.direction_reversal_penalty*float(diagnostics['direction_reversal'] and no_progress),
            measured_yaw_travel=-c.yaw_travel_penalty*measured_yaw_travel/.20,
            measured_slip=-.015*min(slip/.05, 4.), measured_load=-.03*min(load/25., 4.),
            arrival=c.arrival_reward*float(arrived), unsafe=-c.unsafe_penalty*float(info.get('unsafe', False)))
        reward = float(sum(self._reward_components.values()))
        self._previous_action = vector.copy()
        self.last_observation = self.memory.observation(self.goal, self._previous_action)
        terminated = bool(term or (self._navigation_arrived and not self.complete_task))
        truncated = bool(trunc or (not terminated and not self._navigation_arrived
                                  and self._macro_steps >= c.max_macro_steps))
        return self.last_observation.copy(), reward, terminated, truncated, self._public_info(info)

    def set_curriculum(self, difficulty, empty_probability=None, encounter_probability=None):
        updates = dict(difficulty=difficulty)
        if empty_probability is not None:
            updates['empty_world_probability'] = float(empty_probability)
        if encounter_probability is not None:
            updates['encounter_world_probability'] = float(encounter_probability)
        replacement = NavigationBaseConfig(**{**asdict(self.config), **updates})
        self.config = replacement

    def set_difficulty(self, difficulty):
        self.set_curriculum(difficulty)

    def contract(self):
        return dict(schema='sensor_navigation_v7_persistent_attempt_memory', navigation_config=asdict(self.config),
                    observation_size=self.observation_space.shape[0], action_names=list(ACTION_NAMES),
                    action_vectors=ACTION_VECTORS.tolist(), action_table_height_mode_yaw_radians=ACTION_TABLE.tolist(),
                    yaw_amplitudes_radians=list(YAW_LEVELS), native_steps_per_action=self.config.native_steps_per_action,
                    executor='native adaptive MPC local waypoint', actor_map_access=False, hidden_geometry_rewards=False,
                    automatic_action_override=False, action_mask_applied=False,
                    memory='persistent measured action outcomes, public monotonic progress, causal proprioception',
                    reward='monotonic task frontier, measured time/repetition/no-progress/reversal/slip/load, arrival/safety',
                    goal_action_semantics={'positive_height': 'bounded correction toward public task height including overshoot',
                        'retreat': 'policy-selected bounded downward motion, never prohibited',
                        'yaw': 'policy-selected sign and small/medium/large relative waypoint',
                        'hold': 'settle public task pose inside one maximum waypoint span, otherwise hold sensed pose',
                        'actual_motion': 'native rate limits and contact dynamics can reduce requested amplitude',
                        'arrival': 'unchanged native position, rate and sustained hold gates'},
                    work_controller='adaptive MPC; navigation actor stops choosing after arrival')

    def close(self):
        self.base_env.close()
