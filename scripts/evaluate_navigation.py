"""Evaluate a trained policy on fresh obstacle worlds."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from envs.navigation_env import NavigationEnv
from rl.policy import load_policy, write_json


def evaluate(args):
    model, norm, metadata = load_policy(args.policy, device=args.device)
    env = NavigationEnv(metadata["manifest"]["config"], complete_task=args.complete_task)
    rows = []
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            obs, _ = env.reset(seed=seed)
            total, decisions, stages = 0., 0, Counter()
            while True:
                action, _ = model.predict(norm.normalize_obs(obs[None, :]), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
                total += reward
                decisions += 1
                stages[info["stage"]] += 1
                if terminated or truncated:
                    break
            success = info["full_task_success"] if args.complete_task else info["navigation_success"]
            rows.append({"seed": seed, "success": bool(success and not info["unsafe"]),
                         "reward": total, "decisions": decisions,
                         "elapsed_time": info["elapsed_time"], "stages": dict(stages)})
            print(json.dumps(rows[-1]), flush=True)
    finally:
        env.close()
        norm.close()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "evaluation.json", {"episodes": rows,
               "success_count": sum(row["success"] for row in rows),
               "mean_reward": float(np.mean([row["reward"] for row in rows]))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--complete-task", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("Episodes must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
