"""
Train the arm reach policy using SAC (off-policy, sample efficient).
The climber is fixed at target height; arm learns to reach the coconut.

Usage:
    python rl/train_arm.py
    python rl/train_arm.py --timesteps 500000
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from envs.arm_env import ArmReachEnv
from rl.residual_agent import ARM_POLICY_KWARGS

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


SAVE_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def train(args):
    # leaf_pos is randomized inside ArmReachEnv.reset() when options={"leaf_pos": ...} is passed.
    # For standalone arm training we use the default position; in the full pipeline
    # TreeClimberEnv passes its leaf_pos to ArmReachEnv via options at episode end.
    env      = Monitor(ArmReachEnv())
    eval_env = Monitor(ArmReachEnv())

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        eval_freq=5_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=MODEL_DIR,
        name_prefix="arm_sac",
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=ARM_POLICY_KWARGS,
        verbose=1,
        tensorboard_log=os.path.join(SAVE_DIR, "tb_logs"),
        device="auto",
    )

    print(f"Training arm policy for {args.timesteps:,} steps with SAC...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_cb, ckpt_cb],
        progress_bar=True,
    )

    model.save(os.path.join(MODEL_DIR, "arm_sac_final"))
    print(f"Arm model saved to {MODEL_DIR}/arm_sac_final")
    _plot_results(model)


def _plot_results(model):
    fig, ax = plt.subplots(figsize=(8, 4))
    if hasattr(model, "ep_info_buffer") and len(model.ep_info_buffer) > 0:
        rewards = [ep["r"] for ep in model.ep_info_buffer]
        ax.plot(rewards, alpha=0.4, color="darkorange", label="episode reward")
        window = min(40, len(rewards))
        if len(rewards) >= window:
            roll = np.convolve(rewards, np.ones(window)/window, mode="valid")
            ax.plot(range(window-1, len(rewards)), roll, color="darkorange",
                    linewidth=2, label=f"rolling mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Arm Reach Training (SAC) -- SO-ARM101 → Coconut")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "training_curve_arm.png")
    plt.savefig(out, dpi=150)
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    args = parser.parse_args()
    train(args)
