"""
Collect heuristic expert transitions for local low-friction escape.

The expert may use simulator debug info such as wheel_stuck_flags and
active_stuck_patch, but the saved observations are the 29-dim actor
observations only.

Usage:
    MUJOCO_GL=egl python -m rl.collect_escape_expert --out-dir /tmp/palm_escape_expert
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from envs.tree_climber_env import TreeClimberEnv
from mpc.climbing_mpc import ClimbingMPC


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def install_curriculum_patch(env: TreeClimberEnv, rng: np.random.Generator):
    """Install a small random local patch ahead of one wheel to ensure escapes appear."""
    if not hasattr(env, "_get_wheel_world_positions"):
        return

    tree_xy = env._get_tree_center_xy()
    wheel_pos = env._get_wheel_world_positions(env._current_azimuth)
    wheel_idx = int(rng.integers(0, 6))
    wp = wheel_pos[wheel_idx]
    az = float(np.arctan2(wp[1] - tree_xy[1], wp[0] - tree_xy[0]))
    height = float(wp[2] + rng.uniform(0.05, 0.45))

    forced_patch = {
        "id": 10_000,
        "height_center": height,
        "height_width": float(rng.uniform(0.14, 0.22)),
        "azimuth_center": wrap_angle(az + rng.uniform(-0.15, 0.15)),
        "azimuth_width": float(np.deg2rad(rng.uniform(24.0, 42.0))),
        "mu_scale": float(rng.uniform(0.0, 0.10)),
        "severity": 1.0,
    }
    env._stuck_patches = [forced_patch] + list(env._stuck_patches)


def expert_action(
    obs: np.ndarray,
    info: dict,
    step: int,
    rng: np.random.Generator,
    fixed_direction: float | None = None,
) -> np.ndarray:
    if fixed_direction is not None and info.get("episode_patch_escape_count", 0) > 0:
        return np.zeros(6, dtype=np.float32)

    flags = np.asarray(info.get("wheel_stuck_flags", np.zeros(6)), dtype=np.float32)
    flag_level = float(np.clip(np.sum(flags) / 2.0, 0.0, 1.0))
    stuck_level = float(info.get("stuck_level", obs[22]))
    counter_level = float(np.clip(info.get("stuck_counter", 0) / 40.0, 0.0, 1.0))
    stuck_signal = max(stuck_level, flag_level, counter_level)

    if stuck_signal < 0.08:
        return np.zeros(6, dtype=np.float32)

    current_azimuth = float(info.get("current_azimuth", np.arctan2(obs[25], obs[26])))
    active_patch = info.get("active_stuck_patch")
    if fixed_direction is not None:
        direction = float(np.sign(fixed_direction) or 1.0)
    elif active_patch is not None:
        err = wrap_angle(current_azimuth - float(active_patch["azimuth_center"]))
        direction = 1.0 if err >= 0.0 else -1.0
        if abs(err) < 0.08:
            direction = 1.0 if (step // 12) % 2 == 0 else -1.0
    else:
        direction = 1.0 if (step // 16) % 2 == 0 else -1.0
        if rng.random() < 0.02:
            direction *= -1.0

    rotate_pattern = direction * np.array([1, -1, 1, -1, 1, -1], dtype=np.float32)
    loosen = -0.25 * stuck_signal * np.ones(6, dtype=np.float32)
    rotate = (0.35 + 0.65 * stuck_signal) * rotate_pattern
    action = rotate + loosen

    if bool(info.get("escaped_patch", False)):
        action *= 0.35

    return np.clip(action, -1.0, 1.0).astype(np.float32)


def collect(args):
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    observations = []
    actions = []
    rewards = []
    dones = []
    stuck_levels = []
    wheel_stuck_counts = []
    episode_ids = []
    heights = []
    azimuths = []
    delta_azimuths = []
    escaped_patches = []
    expert_escape_successes = []
    episode_summaries = []

    mpc = None if args.no_mpc else ClimbingMPC(require_casadi=args.require_casadi)
    env = TreeClimberEnv(
        mpc=mpc,
        base_friction_range=(args.mu_min, args.mu_max),
        stuck_zone_prob=args.stuck_zone_prob,
        obstacle_mode=args.obstacle_mode,
        max_episode_steps=args.max_steps,
    )

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        if args.obstacle_mode == "random":
            install_curriculum_patch(env, rng)

        ep_saved = 0
        ep_transition_indices = []
        for step in range(args.max_steps):
            fixed_direction = -1.0 if args.obstacle_mode == "fixed" else None
            action = expert_action(obs, info, step, rng, fixed_direction=fixed_direction)
            stuckish = (
                info.get("stuck_level", 0.0) > 0.05
                or info.get("stuck_counter", 0) > 0
                or np.sum(info.get("wheel_stuck_flags", np.zeros(6))) > 0
                or np.any(np.abs(action) > 1e-4)
            )
            keep = stuckish or (rng.random() < args.normal_keep_prob)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            if keep:
                observations.append(obs.astype(np.float32))
                actions.append(action.astype(np.float32))
                rewards.append(float(reward))
                dones.append(bool(terminated or truncated))
                stuck_levels.append(float(info.get("stuck_level", 0.0)))
                wheel_stuck_counts.append(float(np.sum(info.get("wheel_stuck_flags", np.zeros(6)))))
                episode_ids.append(ep)
                heights.append(float(info.get("height", obs[0])))
                azimuths.append(float(info.get("current_azimuth", 0.0)))
                delta_azimuths.append(float(info.get("delta_azimuth", 0.0)))
                escaped_patches.append(bool(info.get("escaped_patch", False)))
                expert_escape_successes.append(False)
                ep_transition_indices.append(len(observations) - 1)
                ep_saved += 1

            obs, info = next_obs, next_info
            if terminated or truncated:
                break

        success = bool(info.get("episode_escape_success", False))
        for idx in ep_transition_indices:
            expert_escape_successes[idx] = success
        episode_summaries.append({
            "episode": ep,
            "saved_transitions": ep_saved,
            "max_height": float(info.get("episode_max_height", obs[0])),
            "net_turn_deg": float(info.get("episode_net_turn_deg", 0.0)),
            "abs_turn_deg": float(info.get("episode_abs_turn_deg", 0.0)),
            "max_turn_excursion_deg": float(info.get("episode_max_turn_excursion_deg", 0.0)),
            "patch_escape_count": int(info.get("episode_patch_escape_count", 0)),
            "post_escape_height_gain": float(info.get("episode_post_escape_height_gain", 0.0)),
            "escape_success": success,
        })

        if (ep + 1) % max(args.log_every, 1) == 0:
            print(f"[collect] episode {ep + 1}/{args.episodes}, saved {ep_saved} transitions")

    env.close()

    if not observations:
        raise RuntimeError("no expert transitions collected")

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    path = out_dir / "expert_transitions.npz"
    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        stuck_levels=np.asarray(stuck_levels, dtype=np.float32),
        wheel_stuck_counts=np.asarray(wheel_stuck_counts, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
        heights=np.asarray(heights, dtype=np.float32),
        azimuths=np.asarray(azimuths, dtype=np.float32),
        delta_azimuths=np.asarray(delta_azimuths, dtype=np.float32),
        escaped_patches=np.asarray(escaped_patches, dtype=np.bool_),
        expert_escape_successes=np.asarray(expert_escape_successes, dtype=np.bool_),
    )
    successful_episodes = int(sum(row["escape_success"] for row in episode_summaries))
    summary = {
        "obstacle_mode": args.obstacle_mode,
        "episodes": len(episode_summaries),
        "successful_escape_episodes": successful_episodes,
        "escape_success_rate": successful_episodes / max(len(episode_summaries), 1),
        "saved_transitions": len(observations),
        "episodes_detail": episode_summaries,
    }
    summary_path = out_dir / "expert_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[collect] saved {len(observations)} transitions to {path}")
    print(f"[collect] obs shape={observations.shape}, action shape={actions.shape}")
    print(f"[collect] successful escape episodes={successful_episodes}/{len(episode_summaries)}")
    print(f"[collect] summary={summary_path}")
    if args.obstacle_mode == "fixed" and successful_episodes == 0:
        raise RuntimeError("fixed-obstacle expert produced no complete escape examples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stuck-zone-prob", type=float, default=0.8)
    parser.add_argument("--obstacle-mode", choices=["none", "random", "fixed"], default="random")
    parser.add_argument("--mu-min", type=float, default=0.60)
    parser.add_argument("--mu-max", type=float, default=1.10)
    parser.add_argument("--normal-keep-prob", type=float, default=0.08)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-mpc", action="store_true")
    parser.add_argument("--require-casadi", action="store_true",
                        help="Fail instead of using fallback MPC if CasADi is unavailable")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
