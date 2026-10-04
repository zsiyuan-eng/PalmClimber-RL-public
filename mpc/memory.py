"""Persistent measured action outcomes and causal navigation history."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from mpc.sensor_memory import OnlineNavigationConfig, OnlineNavigationMemory


def _wrap(value):
    return (np.asarray(value, dtype=float) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class AttemptMemoryConfig:
    context_height_radius: float = .04
    context_azimuth_radius: float = .12
    maximum_attempts: int = 2048
    recent_failure_slots: int = 8
    minimum_attempt_seconds: float = .16
    minimum_height_request: float = .018
    minimum_azimuth_request: float = .045
    failed_height_travel: float = .003
    failed_azimuth_travel: float = .015
    failure_progress_fraction: float = .20

    def __post_init__(self):
        if any(not np.isfinite(getattr(self, k)) or getattr(self, k) <= 0
               for k in ('context_height_radius', 'context_azimuth_radius',
                         'minimum_attempt_seconds', 'minimum_height_request',
                         'minimum_azimuth_request', 'failed_height_travel',
                         'failed_azimuth_travel', 'failure_progress_fraction')):
            raise ValueError('Attempt memory lengths, times and thresholds must be positive')
        if self.maximum_attempts < 1 or self.recent_failure_slots < 1:
            raise ValueError('Attempt memory needs positive capacities')


class PersistentAttemptMemory(OnlineNavigationMemory):
    """Sensor evidence, prospective action outcomes and episode history."""

    def __init__(self, action_table, *, sensor_config=None, attempt_config=None):
        table = np.asarray(action_table, dtype=float)
        if table.ndim != 2 or table.shape[1] != 2 or not np.isfinite(table).all():
            raise ValueError('Action table must have finite height/yaw rows')
        self.action_table = table.copy()
        self.attempt_config = (attempt_config if isinstance(attempt_config, AttemptMemoryConfig)
                               else AttemptMemoryConfig(**(attempt_config or {})))
        super().__init__(sensor_config or OnlineNavigationConfig(
            height_action_span=.08, azimuth_action_span=.42,
            height_cell_size=.03, azimuth_cell_size=.10))
        self.sensor_observation_size = self.observation_size
        self.episode_feature_size = 12
        self.observation_size += (3 * len(table) + 7 * self.attempt_config.recent_failure_slots
                                  + self.episode_feature_size)

    def reset(self, state, goal):
        self._attempts = []
        self._recent_failures = []
        self._attempt_count = self._failed_count = self._repeated_failed_count = 0
        self._no_progress_count = 0
        self._last_failure = self._last_progress_fraction = self._last_repeat_confidence = 0.
        self._last_failure_basis = 'none'
        self._last_direction_reversal = False
        self._last_attempt_delta = np.zeros(2)
        self._last_attempt_end_capture = None
        self._last_nonzero_yaw_sign = 0.
        capture = float(state.get('sensor_capture_time', state.get('capture_time', state.get('time', 0.))))
        self._episode_start_capture = self._best_progress_capture = capture
        self._best_height = min(float(state['base_height']), float(np.asarray(goal)[0]))
        self._best_distance = abs(float(state['base_height']) - float(np.asarray(goal)[0])) + .12 * abs(
            float(_wrap(float(state['base_azimuth']) - float(np.asarray(goal)[1]))))
        self._last_best_height_gain = self._last_best_distance_gain = 0.
        super().reset(state, goal)
        return self.observation()

    def _nearby(self, pose, action_index=None):
        c = self.attempt_config
        for item in self._attempts:
            delta = np.asarray(item['start_pose']) - pose
            delta[1] = float(_wrap(delta[1]))
            if (abs(delta[0]) <= c.context_height_radius and abs(delta[1]) <= c.context_azimuth_radius
                    and (action_index is None or item['action_index'] == action_index)):
                yield item

    def action_profiles(self, state=None):
        """Failure, number of attempts and last progress for each actor option."""
        pose = (self._frame_state['pose'] if state is None
                else self._frame(state)['pose'])
        result = np.zeros((len(self.action_table), 3), dtype=float)
        for index in range(len(result)):
            nearby = list(self._nearby(pose, index))
            if nearby:
                # A genuinely successful later try at the same configuration
                # can update an earlier failure instead of permanently banning it.
                latest = nearby[-1]
                failure = max((item['failure_confidence'] for item in nearby), default=0.)
                if latest['progress_fraction'] >= .50 and not latest['failed']:
                    failure *= .25
                result[index] = [failure, min(len(nearby), 8) / 4.,
                                 np.clip(latest['progress_fraction'], -2., 2.) / 2.]
        return result

    def record_attempt(self, before, after, action_index, effective_delta):
        """Store only measured macro outcomes; sensor delay is respected.

        A diagonal request can move sideways while failing to gain height. The
        vertical and horizontal components are evaluated separately so sideways
        motion cannot erase the evidence of a blocked climb.
        """
        initial_profiles = self.action_profiles(before)
        self._last_best_height_gain = self._last_best_distance_gain = 0.
        before, after = self._frame(before), self._frame(after)
        delta = np.asarray(effective_delta, dtype=float).copy()
        if delta.shape != (2,) or not np.isfinite(delta).all():
            raise ValueError('Effective attempt delta must be finite height/yaw')
        index = int(action_index)
        if index != action_index or not 0 <= index < len(self.action_table):
            raise ValueError('Attempt action index is outside the actor contract')
        duration = after['capture'] - before['capture']
        if duration < self.attempt_config.minimum_attempt_seconds - 1e-9:
            return self.attempt_diagnostics()
        if self._last_attempt_end_capture is not None and after['capture'] <= self._last_attempt_end_capture + 1e-9:
            return self.attempt_diagnostics()
        movement = after['pose'] - before['pose']
        movement[1] = float(_wrap(movement[1]))
        requested_h = abs(delta[0]) >= self.attempt_config.minimum_height_request
        requested_yaw = abs(delta[1]) >= self.attempt_config.minimum_azimuth_request
        expected = np.minimum(np.abs(delta), np.array([.20, .65]) * duration)
        achieved = movement * np.sign(delta)
        component_fraction = achieved / np.maximum(expected, 1e-9)
        components = []
        if requested_h:
            components.append(component_fraction[0])
        if requested_yaw:
            components.append(component_fraction[1])
        fraction = float(min(components)) if components else 1.
        failed_h = (requested_h and achieved[0] <= self.attempt_config.failed_height_travel
                    and component_fraction[0] < self.attempt_config.failure_progress_fraction)
        failed_yaw = (requested_yaw and achieved[1] <= self.attempt_config.failed_azimuth_travel
                      and component_fraction[1] < self.attempt_config.failure_progress_fraction)
        corroboration = max(self._confidence, min(1., float(np.max(np.abs(after['slip']))) / .025))
        weak_confidence = self.config.uncorroborated_motion_confidence
        persistent_motion_failure = self._confidence >= weak_confidence-1e-9
        # A blocked climb can still make horizontal progress. The original
        # combined-vector detector then correctly sees some motion, but that
        # must not erase a full-window failed vertical request. Constant macro
        # requests make the measured component interval causally coherent.
        sustained_component_failure = (duration+1e-9 >= self.config.minimum_uncorroborated_window_seconds
                                       and (failed_h or failed_yaw))
        strong_failure = (failed_h or failed_yaw) and corroboration >= .40
        failed = bool((failed_h or failed_yaw)
                      and (strong_failure or persistent_motion_failure or sustained_component_failure))
        confidence = float(corroboration if strong_failure else weak_confidence if failed else 0.)
        failure_basis = ('sensor_corroborated_axis_failure' if strong_failure
                         else self._evidence_basis if failed and persistent_motion_failure
                         else 'sustained_height_component_failure' if failed and failed_h
                         else 'sustained_yaw_component_failure' if failed and failed_yaw else 'none')
        repeat = float(initial_profiles[index, 0])
        yaw_sign = float(np.sign(delta[1])) if requested_yaw else 0.
        reversal = bool(yaw_sign and self._last_nonzero_yaw_sign
                        and yaw_sign != self._last_nonzero_yaw_sign)
        item = dict(start_pose=before['pose'].tolist(), end_pose=after['pose'].tolist(),
                    start_capture_time=float(before['capture']), end_capture_time=float(after['capture']),
                    action_index=index, effective_delta=delta.tolist(), measured_delta=movement.tolist(),
                    progress_fraction=fraction, failure_confidence=confidence, failed=failed,
                    failure_evidence_basis=failure_basis,
                    height_component_failed=bool(failed_h), yaw_component_failed=bool(failed_yaw))
        self._attempts.append(item)
        del self._attempts[:-self.attempt_config.maximum_attempts]
        if failed:
            self._recent_failures.append(item)
            del self._recent_failures[:-self.attempt_config.recent_failure_slots]
        self._attempt_count += 1
        self._failed_count += int(failed)
        self._repeated_failed_count += int(failed and repeat >= weak_confidence-1e-9)
        self._last_repeat_confidence = repeat
        self._last_failure, self._last_progress_fraction = confidence, fraction
        self._last_failure_basis = failure_basis
        self._last_direction_reversal = reversal
        self._last_attempt_delta = delta
        self._last_attempt_end_capture = after['capture']
        if yaw_sign:
            self._last_nonzero_yaw_sign = yaw_sign
        height = min(float(after['pose'][0]), float(self._goal[0]))
        distance = abs(float(after['pose'][0]) - float(self._goal[0])) + .12 * abs(
            float(_wrap(after['pose'][1] - self._goal[1])))
        self._last_best_height_gain = max(0., height - self._best_height)
        self._last_best_distance_gain = max(0., self._best_distance - distance)
        self._best_height = max(self._best_height, height)
        self._best_distance = min(self._best_distance, distance)
        if self._last_best_height_gain > .0015 or self._last_best_distance_gain > .002:
            self._best_progress_capture = after['capture']
            self._no_progress_count = 0
        else:
            self._no_progress_count += 1
        return self.attempt_diagnostics()

    def attempt_diagnostics(self):
        return dict(attempt_memory_source='measured_proprioceptive_macro_outcomes',
                    attempted_action_count=int(self._attempt_count),
                    failed_attempt_count=int(self._failed_count),
                    repeated_failed_attempts=int(self._repeated_failed_count),
                    repeat_failed_action_confidence=float(self._last_repeat_confidence),
                    last_attempt_failure_confidence=float(self._last_failure),
                    last_attempt_failure_evidence_basis=self._last_failure_basis,
                    last_attempt_progress_fraction=float(self._last_progress_fraction),
                    consecutive_no_progress_attempts=int(self._no_progress_count),
                    direction_reversal=bool(self._last_direction_reversal),
                    best_height=float(self._best_height), best_task_distance=float(self._best_distance),
                    best_height_gain=float(self._last_best_height_gain),
                    best_task_distance_gain=float(self._last_best_distance_gain),
                    elapsed_since_best_progress=max(0., float(self._frame_state['capture'] - self._best_progress_capture)),
                    last_attempt_delta=self._last_attempt_delta.tolist(),
                    automatic_action_override=False, action_mask_applied=False)

    def diagnostics(self):
        return {**super().diagnostics(), **self.attempt_diagnostics()}

    def observation(self, goal=None, previous_action=None):
        # The parent verifies its own 182-value layout, so temporarily expose
        # that size only while assembling its original sensor projection.
        full_size = self.observation_size
        self.observation_size = self.sensor_observation_size
        try:
            sensors = super().observation(goal, previous_action)
        finally:
            self.observation_size = full_size
        pose = self._frame_state['pose']
        recent = np.zeros((self.attempt_config.recent_failure_slots, 7))
        for slot, item in enumerate(reversed(self._recent_failures)):
            relative = np.asarray(item['start_pose']) - pose
            relative[1] = float(_wrap(relative[1]))
            delta = np.asarray(item['effective_delta'])
            recent[slot] = [relative[0] / .20, relative[1] / .60,
                            delta[0] / .08, delta[1] / .42,
                            item['failure_confidence'], np.clip(item['progress_fraction'], -2., 2.) / 2.,
                            min(1., (self._frame_state['capture'] - item['end_capture_time']) / 20.)]
        episode = np.array([
            (self._frame_state['capture'] - self._episode_start_capture) / 60.,
            (self._frame_state['capture'] - self._best_progress_capture) / 8.,
            (self._goal[0] - self._best_height) / 1.6, self._best_distance / .8,
            self._no_progress_count / 8., self._repeated_failed_count / 8.,
            self._attempt_count / 220., self._last_failure,
            self._last_progress_fraction / 2., float(self._last_direction_reversal),
            self._last_attempt_delta[0] / .08, self._last_attempt_delta[1] / .42])
        result = np.concatenate([sensors, self.action_profiles().ravel(), recent.ravel(), episode])
        if result.shape != (full_size,) or not np.isfinite(result).all():
            raise RuntimeError('attempt-memory observation is inconsistent')
        return np.clip(result, -4., 4.).astype(np.float32)
