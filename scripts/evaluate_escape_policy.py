"""
Evaluate and record the trained escape policy on random local friction patches.

Outputs:
    metrics.json / metrics.csv
    summary.png
    policy_episode_*.mp4

"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from envs.tree_climber_env import TARGET_CLIMB_HEIGHT, TreeClimberEnv
from mpc.climbing_mpc import ClimbingMPC


def load_policy(run_dir: Path, model_name: str):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model_path = run_dir / "checkpoints" / model_name
    norm_path = run_dir / "checkpoints" / "vec_normalize.pkl"
    if model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")
    if model_path.stem == "best_model":
        best_norm = run_dir / "checkpoints" / "best_model_vec_normalize.pkl"
        if best_norm.exists():
            norm_path = best_norm
    elif model_path.stem.startswith("escape_ppo_") and model_path.stem.endswith("_steps"):
        step = model_path.stem.removeprefix("escape_ppo_").removesuffix("_steps")
        checkpoint_norm = run_dir / "checkpoints" / f"escape_ppo_vecnormalize_{step}_steps.pkl"
        if checkpoint_norm.exists():
            norm_path = checkpoint_norm
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not norm_path.exists():
        raise FileNotFoundError(norm_path)

    dummy_env = DummyVecEnv([lambda: TreeClimberEnv(mpc=ClimbingMPC())])
    norm_env = VecNormalize.load(str(norm_path), dummy_env)
    norm_env.training = False
    norm_env.norm_reward = False
    model = PPO.load(str(model_path), env=norm_env, device="cpu")
    return model, norm_env


def draw_overlay(frame: np.ndarray, info: dict, action: np.ndarray, reward: float, mode: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    height = float(info.get("height", 0.0))
    stuck_level = float(info.get("stuck_level", 0.0))
    flags = np.asarray(info.get("wheel_stuck_flags", np.zeros(6)))
    fric = np.asarray(info.get("per_wheel_friction", np.ones(6)))
    magic = float(info.get("magic_lateral_cmd", 0.0))
    escaped = bool(info.get("escaped_patch", False))

    text = (
        f"{mode} | h {height:.2f}/{TARGET_CLIMB_HEIGHT:.1f}m | "
        f"stuck {stuck_level:.2f} | wheels stuck {int(flags.sum())} | "
        f"lat {magic:+.2f} | r {reward:+.1f}"
    )
    if escaped:
        text += " | ESCAPED"

    pad = 8
    box = draw.textbbox((12, 12), text)
    draw.rounded_rectangle(
        (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
        radius=5,
        fill=(0, 0, 0, 150),
    )
    draw.text((12, 12), text, fill=(255, 255, 255, 255))

    # Right-side diagnostic panel: local patch map in height/azimuth space.
    panel_w = 150
    px0 = w - panel_w - 12
    py0 = 58
    px1 = w - 12
    py1 = h - 18
    draw.rounded_rectangle((px0, py0, px1, py1), radius=5, fill=(0, 0, 0, 105))
    draw.text((px0 + 10, py0 + 8), "random patches", fill=(255, 255, 255, 255))

    patches = info.get("stuck_patches", [])
    for patch in patches[:8]:
        az = float(patch["azimuth_center"])
        z = float(patch["height_center"])
        az_norm = (az + np.pi) / (2 * np.pi)
        z_norm = np.clip((z - 0.4) / max(TARGET_CLIMB_HEIGHT - 0.4, 1e-6), 0.0, 1.0)
        cx = px0 + 14 + az_norm * (panel_w - 28)
        cy = py1 - 20 - z_norm * (py1 - py0 - 48)
        aw = max(4.0, float(patch["azimuth_width"]) / (2 * np.pi) * (panel_w - 28))
        hh = max(4.0, float(patch["height_width"]) / max(TARGET_CLIMB_HEIGHT - 0.4, 1e-6) * (py1 - py0 - 48))
        draw.rectangle((cx - aw / 2, cy - hh / 2, cx + aw / 2, cy + hh / 2), fill=(255, 0, 0, 170))

    cur_az = float(info.get("current_azimuth", 0.0))
    cur_x = px0 + 14 + ((cur_az + np.pi) / (2 * np.pi)) * (panel_w - 28)
    cur_y = py1 - 20 - np.clip((height - 0.4) / max(TARGET_CLIMB_HEIGHT - 0.4, 1e-6), 0.0, 1.0) * (py1 - py0 - 48)
    draw.ellipse((cur_x - 5, cur_y - 5, cur_x + 5, cur_y + 5), fill=(0, 255, 120, 230))

    # Per-wheel friction strip.
    strip_y = py1 - 12
    for i, mu in enumerate(fric[:6]):
        color = (255, 40, 40, 230) if flags[i] > 0 else (60, 210, 120, 230)
        x = px0 + 12 + i * 20
        draw.rectangle((x, strip_y - 10, x + 14, strip_y), fill=color)

    return np.asarray(img)


def run_episode(env, model, norm_env, seed, max_steps, mode, video_path=None, fps=25, render_every=2):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    max_height = float(obs[0])
    max_stuck_level = 0.0
    stuck_steps = 0
    escaped_events = 0
    lateral_steps = 0
    reached = False
    fallen = False
    frames = []

    for step in range(max_steps):
        if mode == "policy":
            policy_obs = norm_env.normalize_obs(obs.reshape(1, -1))
            action, _ = model.predict(policy_obs, deterministic=True)
            action = action.flatten()
        else:
            action = np.zeros(6, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        max_height = max(max_height, float(info.get("height", obs[0])))
        stuck_level = float(info.get("stuck_level", 0.0))
        max_stuck_level = max(max_stuck_level, stuck_level)
        stuck_steps += int(stuck_level > 0.25 or np.sum(info.get("wheel_stuck_flags", np.zeros(6))) > 0)
        escaped_events += int(bool(info.get("escaped_patch", False)))
        lateral_steps += int(abs(float(info.get("magic_lateral_cmd", 0.0))) > 0.20)
        reached = bool(info.get("reached_target", False))
        fallen = bool(info.get("fallen", False))

        if video_path is not None and step % render_every == 0:
            frame = env.render()
            if frame is not None:
                frames.append(draw_overlay(frame, info, action, reward, mode))

        if terminated or truncated:
            break

    if video_path is not None and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=fps, macro_block_size=1)

    return {
        "mode": mode,
        "seed": seed,
        "steps": step + 1,
        "total_reward": total_reward,
        "max_height": max_height,
        "final_height": float(info.get("height", obs[0])),
        "reached_target": reached,
        "fallen": fallen,
        "max_stuck_level": max_stuck_level,
        "stuck_steps": stuck_steps,
        "escaped_events": escaped_events,
        "lateral_steps": lateral_steps,
    }


def summarize(rows):
    out = {}
    for mode in sorted(set(r["mode"] for r in rows)):
        part = [r for r in rows if r["mode"] == mode]
        out[mode] = {
            "episodes": len(part),
            "success_rate": float(np.mean([r["reached_target"] for r in part])),
            "fall_rate": float(np.mean([r["fallen"] for r in part])),
            "mean_max_height": float(np.mean([r["max_height"] for r in part])),
            "mean_final_height": float(np.mean([r["final_height"] for r in part])),
            "mean_reward": float(np.mean([r["total_reward"] for r in part])),
            "mean_escaped_events": float(np.mean([r["escaped_events"] for r in part])),
            "mean_lateral_steps": float(np.mean([r["lateral_steps"] for r in part])),
            "mean_stuck_steps": float(np.mean([r["stuck_steps"] for r in part])),
        }
    return out


def save_summary_plot(rows, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = ["mpc", "policy"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    metrics = [
        ("max_height", "Max height (m)"),
        ("escaped_events", "Patch escapes / ep"),
        ("lateral_steps", "Lateral-command steps / ep"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        data = [[r[key] for r in rows if r["mode"] == mode] for mode in modes]
        ax.boxplot(data, labels=["MPC only", "MPC+RL"])
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="escape_ppo_final")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--video-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--stuck-zone-prob", type=float, default=1.0)
    parser.add_argument("--obstacle-mode", choices=["none", "random", "fixed"], default="fixed")
    parser.add_argument("--mu-min", type=float, default=0.75)
    parser.add_argument("--mu-max", type=float, default=0.75)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.run_dir / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, norm_env = load_policy(args.run_dir, args.model_name)
    rows = []
    for mode in ["mpc", "policy"]:
        env = TreeClimberEnv(
            mpc=ClimbingMPC(),
            render_mode="rgb_array",
            base_friction_range=(args.mu_min, args.mu_max),
            stuck_zone_prob=args.stuck_zone_prob,
            obstacle_mode=args.obstacle_mode,
            max_episode_steps=args.max_steps,
        )
        for ep in range(args.episodes):
            seed = args.seed + ep
            video_path = None
            if mode == "policy" and ep < args.video_episodes:
                video_path = out_dir / f"policy_episode_{ep:02d}.mp4"
            row = run_episode(
                env,
                model,
                norm_env,
                seed,
                args.max_steps,
                mode,
                video_path=video_path,
            )
            rows.append(row)
            print(
                f"[{mode}] ep={ep:02d} seed={seed} "
                f"max_h={row['max_height']:.3f} final_h={row['final_height']:.3f} "
                f"escapes={row['escaped_events']} lateral_steps={row['lateral_steps']} "
                f"reached={row['reached_target']} fallen={row['fallen']}"
            )
        env.close()

    summary = summarize(rows)
    metrics_json = out_dir / "metrics.json"
    metrics_csv = out_dir / "metrics.csv"
    with metrics_json.open("w") as f:
        json.dump({"summary": summary, "episodes": rows}, f, indent=2)
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_summary_plot(rows, out_dir / "summary.png")

    print("[summary]")
    print(json.dumps(summary, indent=2))
    print(f"[saved] {metrics_json}")
    print(f"[saved] {metrics_csv}")
    print(f"[saved] {out_dir / 'summary.png'}")


if __name__ == "__main__":
    main()
