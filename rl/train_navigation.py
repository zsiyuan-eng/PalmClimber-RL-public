"""Train PPO navigation with measured failure memory and native MPC execution."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from envs.navigation_env import NavigationEnv
from rl.policy import build_vector, jsonable, load_policy, save_policy, write_json


def evaluate(model, norm, config, first_seed, episodes):
    env = NavigationEnv(config)
    rows = []
    try:
        for seed in range(first_seed, first_seed + episodes):
            obs, _ = env.reset(seed=seed)
            total, decisions = 0., 0
            while True:
                action, _ = model.predict(norm.normalize_obs(obs[None, :]), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
                total += reward
                decisions += 1
                if terminated or truncated:
                    break
            diagnostic = env.memory.attempt_diagnostics()
            repeated = int(diagnostic.get("repeated_failed_attempts", 0))
            success = bool(info["navigation_success"] and not info["unsafe"])
            rows.append({"seed": seed, "reward": total, "success": success,
                         "decisions": decisions, "repeated_failed_attempts": repeated,
                         "effort": decisions + 8 * repeated + (0 if success else 280)
                                   + (1000 if info["unsafe"] else 0)})
    finally:
        env.close()
    return {"episodes": rows, "success_count": sum(row["success"] for row in rows),
            "mean_reward": float(np.mean([row["reward"] for row in rows])),
            "mean_effort": float(np.mean([row["effort"] for row in rows]))}


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize

    torch.set_num_threads(1)
    config = json.loads(args.config.read_text()) if args.config else {}
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    vector = build_vector(config, args.seed, args.n_envs)
    norm = VecNormalize(vector, norm_obs=True, norm_reward=True, clip_obs=10.,
                        clip_reward=10., gamma=.995)
    reference = NavigationEnv(config)
    manifest = {"seed": args.seed, "config": config, "contract": reference.contract(),
                "network": [256, 256], "activation": "Tanh", "gamma": .995,
                "timesteps": args.timesteps, "rollout_length": args.rollout_steps,
                "n_envs": args.n_envs, "validation_seed": args.validation_seed,
                "validation_episodes": args.validation_episodes}
    reference.close()
    write_json(out / "manifest.json", manifest)

    model = PPO("MlpPolicy", norm, device=args.device, seed=args.seed,
                n_steps=args.rollout_steps, batch_size=min(512, args.rollout_steps * args.n_envs),
                n_epochs=8, learning_rate=lambda remaining: 3e-5 + 7e-5 * remaining,
                gamma=.995, gae_lambda=.95, ent_coef=.020, clip_range=.20,
                policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
                verbose=1)
    if args.warm_start:
        imported, imported_norm, metadata = load_policy(args.warm_start, device=args.device)
        try:
            if imported.observation_space != model.observation_space or imported.action_space != model.action_space:
                raise ValueError("Transferred observation/action spaces differ")
            model.policy.load_state_dict(imported.policy.state_dict())
            norm.obs_rms = imported_norm.obs_rms
            norm.ret_rms = imported_norm.ret_rms
            manifest["warm_start_sha256"] = metadata["model_sha256"]
        finally:
            imported_norm.close()
        write_json(out / "manifest.json", manifest)

    class Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best = (-1, -float("inf"), -float("inf"))
            self.last_validation = 0
            self.last_phase = None
            self.evaluations = []

        def set_phase(self):
            fraction = self.num_timesteps / args.timesteps
            difficulty = 1 if fraction < .25 else 2 if fraction < .5 else 3
            if difficulty != self.last_phase:
                self.training_env.env_method("set_difficulty", difficulty)
                self.last_phase = difficulty

        def _on_training_start(self):
            self.set_phase()

        def _on_step(self):
            self.set_phase()
            self.model.ent_coef = .020 - .014 * min(self.num_timesteps / args.timesteps, 1.)
            for info in self.locals["infos"]:
                if "episode" in info:
                    row = {"timesteps": self.num_timesteps, "episode": info["episode"],
                           "success": info.get("navigation_success", False),
                           "difficulty": self.last_phase}
                    with (out / "episodes.jsonl").open("a") as stream:
                        stream.write(json.dumps(jsonable(row)) + "\n")
            if self.num_timesteps - self.last_validation >= args.eval_frequency:
                self.validate()
            return True

        def validate(self):
            self.last_validation = self.num_timesteps
            result = evaluate(self.model, self.training_env, config,
                              args.validation_seed, args.validation_episodes)
            result["timesteps"] = self.num_timesteps
            self.evaluations.append(result)
            write_json(out / "validation.json", self.evaluations)
            with (out / "rewards.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["timesteps", "mean_reward", "success_count"])
                writer.writeheader()
                writer.writerows({key: row[key] for key in writer.fieldnames} for row in self.evaluations)
            score = (result["success_count"], -result["mean_effort"], result["mean_reward"])
            if score > self.best:
                self.best = score
                save_policy(self.model, self.training_env, out / "checkpoints/best_model.zip", manifest)
            print(json.dumps({key: value for key, value in result.items() if key != "episodes"}), flush=True)

    callback = Callback()
    try:
        callback.init_callback(model)
        callback.validate()
        model.learn(total_timesteps=args.timesteps, callback=callback)
        callback.validate()
        save_policy(model, norm, out / "checkpoints/final_model.zip", manifest)
    finally:
        norm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/train"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--timesteps", type=int, default=524288)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--eval-frequency", type=int, default=16384)
    parser.add_argument("--validation-seed", type=int, default=1000)
    parser.add_argument("--validation-episodes", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.timesteps, args.n_envs, args.eval_frequency, args.validation_episodes) < 1 or args.rollout_steps < 2:
        parser.error("Counts must be positive and rollout length must be at least two")
    train(args)


if __name__ == "__main__":
    main()
