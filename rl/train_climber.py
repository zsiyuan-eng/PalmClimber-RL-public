"""
Train the residual RL agent for tree climbing.

Architecture:
    u_final = mpc_weight * u_MPC + residual_scale(stuck_level) * pi_theta(obs)

Training loop uses PPO from stable-baselines3.
Friction is randomized every episode (domain randomization) to make the
residual policy robust to real-world uncertainty.

Usage:
    python rl/train_climber.py
    python rl/train_climber.py --no-mpc        # train pure RL (slower to converge)
    python rl/train_climber.py --timesteps 1000000
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from envs.tree_climber_env import TreeClimberEnv  # obs=29dim, act=6dim
from rl.residual_agent import RESIDUAL_POLICY_KWARGS
from mpc.climbing_mpc import ClimbingMPC

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


SAVE_DIR    = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class HeightTrackingCallback(BaseCallback):
    """Log mean height reached per episode."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_heights = []
        self._ep_max_heights = []

    def _on_training_start(self):
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._ep_max_heights = [-np.inf for _ in range(n_envs)]

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))
        for idx, info in enumerate(infos):
            if "height" in info:
                self._ep_max_heights[idx] = max(self._ep_max_heights[idx], info["height"])
            if idx < len(dones) and dones[idx]:
                if np.isfinite(self._ep_max_heights[idx]):
                    self.episode_heights.append(self._ep_max_heights[idx])
                self._ep_max_heights[idx] = -np.inf
        return True


def make_env(rank, use_mpc=True, seed=0):
    def _init():
        mpc = ClimbingMPC() if use_mpc else None
        env = TreeClimberEnv(mpc=mpc, base_friction_range=(0.65, 1.1))
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train(args):
    n_envs   = args.n_envs
    use_mpc  = not args.no_mpc
    total_ts = args.timesteps

    print(f"Setting up {n_envs} parallel environments (MPC={'on' if use_mpc else 'off'})")
    env_fns  = [make_env(i, use_mpc=use_mpc, seed=42) for i in range(n_envs)]
    vec_env  = SubprocVecEnv(env_fns)
    vec_env  = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([lambda: Monitor(TreeClimberEnv(
        mpc=ClimbingMPC() if use_mpc else None,
        base_friction_range=(0.7, 0.9),
    ))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_cb  = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=MODEL_DIR,
        name_prefix="climber_ppo",
    )
    height_cb = HeightTrackingCallback()

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=RESIDUAL_POLICY_KWARGS,
        verbose=1,
        tensorboard_log=os.path.join(SAVE_DIR, "tb_logs"),
        device="auto",
    )

    print(f"Starting training for {total_ts:,} timesteps...")
    model.learn(
        total_timesteps=total_ts,
        callback=[eval_cb, ckpt_cb, height_cb],
        progress_bar=True,
    )

    # save final model + normalizer
    model.save(os.path.join(MODEL_DIR, "climber_ppo_final"))
    vec_env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    print(f"Model saved to {MODEL_DIR}/climber_ppo_final")

    _plot_results(model, height_cb)


def _plot_results(model, height_cb):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # training reward curve from SB3 logger
    if hasattr(model, "ep_info_buffer") and len(model.ep_info_buffer) > 0:
        rewards = [ep["r"] for ep in model.ep_info_buffer]
        axes[0].plot(rewards, alpha=0.4, color="steelblue", label="episode reward")
        # rolling mean
        window = min(50, len(rewards))
        if len(rewards) >= window:
            roll = np.convolve(rewards, np.ones(window)/window, mode="valid")
            axes[0].plot(range(window-1, len(rewards)), roll, color="steelblue",
                         linewidth=2, label=f"rolling mean ({window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Training Reward (Residual MPC-RL)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # height reached over time
    if height_cb.episode_heights:
        h = height_cb.episode_heights
        axes[1].plot(h, alpha=0.4, color="coral", label="max height / ep")
        window = min(30, len(h))
        if len(h) >= window:
            roll = np.convolve(h, np.ones(window)/window, mode="valid")
            axes[1].plot(range(window-1, len(h)), roll, color="coral",
                         linewidth=2, label=f"rolling mean")
        axes[1].axhline(y=2.2, color="green", linestyle="--", label="target height")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Height reached (m)")
        axes[1].set_title("Climbing Progress")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "training_curve_climber.png")
    plt.savefig(out, dpi=150)
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs",   type=int, default=4)
    parser.add_argument("--no-mpc",   action="store_true",
                        help="Train pure RL without MPC (baseline comparison)")
    args = parser.parse_args()
    train(args)
