"""
Train the climbing residual policy for local stuck/escape behavior.

Pipeline:
    1. Load heuristic expert transitions collected by rl.collect_escape_expert
    2. Behavior-clone the PPO policy on 29-dim actor observations
    3. Continue PPO in the selected fixed or randomized patch environment

"""

import argparse
import csv
import os
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


EPISODE_INFO_KEYS = (
    "episode_max_height",
    "episode_net_turn_deg",
    "episode_abs_turn_deg",
    "episode_max_turn_excursion_deg",
    "episode_patch_escape_count",
    "episode_post_escape_height_gain",
    "episode_escape_success",
)


class EpisodeMetricsCallback(BaseCallback):
    """Write fixed-obstacle episode outcomes to TensorBoard and CSV."""

    def __init__(self, csv_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self._stream = None
        self._writer = None
        self._windows = defaultdict(lambda: deque(maxlen=100))

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.csv_path.open("w", newline="")
        self._writer = csv.DictWriter(
            self._stream,
            fieldnames=["timesteps", *EPISODE_INFO_KEYS],
        )
        self._writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if not done or not all(key in info for key in EPISODE_INFO_KEYS):
                continue
            row = {"timesteps": self.num_timesteps}
            for key in EPISODE_INFO_KEYS:
                value = float(info[key])
                row[key] = value
                self._windows[key].append(value)
            self._writer.writerow(row)
            self._stream.flush()

        tensorboard_names = {
            "episode_max_height": "rollout/max_height_mean",
            "episode_net_turn_deg": "rollout/net_turn_deg_mean",
            "episode_abs_turn_deg": "rollout/abs_turn_deg_mean",
            "episode_max_turn_excursion_deg": "rollout/max_turn_excursion_deg_mean",
            "episode_patch_escape_count": "rollout/patch_escape_count_mean",
            "episode_post_escape_height_gain": "rollout/post_escape_height_gain_mean",
            "episode_escape_success": "rollout/escape_success_rate",
        }
        for key, tag in tensorboard_names.items():
            if self._windows[key]:
                self.logger.record(tag, float(np.mean(self._windows[key])))
        return True

    def _on_training_end(self) -> None:
        if self._stream is not None:
            self._stream.close()


class SaveVecNormalizeOnBestCallback(BaseCallback):
    def __init__(self, save_path: Path):
        super().__init__(verbose=0)
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            vec_normalize.save(str(self.save_path))
        return True


def make_env(
    rank,
    seed,
    use_mpc,
    mu_min,
    mu_max,
    stuck_zone_prob,
    obstacle_mode,
    max_episode_steps,
    require_casadi,
):
    def _init():
        from stable_baselines3.common.monitor import Monitor

        from envs.tree_climber_env import TreeClimberEnv
        from mpc.climbing_mpc import ClimbingMPC

        mpc = ClimbingMPC(require_casadi=require_casadi) if use_mpc else None
        env = TreeClimberEnv(
            mpc=mpc,
            base_friction_range=(mu_min, mu_max),
            stuck_zone_prob=stuck_zone_prob,
            obstacle_mode=obstacle_mode,
            max_episode_steps=max_episode_steps,
        )
        env = Monitor(env, info_keywords=EPISODE_INFO_KEYS)
        env.reset(seed=seed + rank)
        return env

    return _init


def load_expert_arrays(expert_path):
    data = np.load(expert_path)
    observations = data["observations"].astype(np.float32)
    actions = np.clip(data["actions"].astype(np.float32), -1.0, 1.0)
    if observations.ndim != 2 or observations.shape[1] != 29:
        raise ValueError(f"expert observations must be (N,29), got {observations.shape}")
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(f"expert actions must be (N,6), got {actions.shape}")
    return observations, actions


def fit_vecnormalize_to_observations(vec_env, observations):
    if hasattr(vec_env, "obs_rms"):
        vec_env.obs_rms.update(observations)
        print(f"[norm] updated VecNormalize obs stats from {len(observations)} expert observations")


def maybe_normalize_observations(vec_env, observations):
    if hasattr(vec_env, "normalize_obs"):
        return vec_env.normalize_obs(observations.copy())
    return observations


def behavior_clone_policy(model, vec_env, observations, actions, run_dir, epochs, batch_size, seed):
    if epochs <= 0:
        return

    import torch

    observations = maybe_normalize_observations(vec_env, observations).astype(np.float32)

    rng = np.random.default_rng(seed)
    device = model.policy.device
    model.policy.set_training_mode(True)

    n = observations.shape[0]
    print(f"[bc] warm-starting on {n} expert transitions for {epochs} epochs")
    for epoch in range(epochs):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            obs_t = torch.as_tensor(observations[idx], dtype=torch.float32, device=device)
            act_t = torch.as_tensor(actions[idx], dtype=torch.float32, device=device)

            dist = model.policy.get_distribution(obs_t)
            log_prob = dist.log_prob(act_t)
            entropy = dist.entropy()
            loss = -log_prob.mean() - 0.001 * entropy.mean()

            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        print(f"[bc] epoch {epoch + 1:03d}/{epochs}, loss={np.mean(losses):.4f}")

    bc_path = Path(run_dir) / "escape_ppo_bc_pretrained"
    model.save(str(bc_path))
    print(f"[bc] saved warm-start checkpoint to {bc_path}.zip")


def build_vec_env(args, use_mpc, stuck_zone_prob, mu_min, mu_max, obstacle_mode, seed_offset=0):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    env_fns = [
        make_env(
            i,
            args.seed + seed_offset,
            use_mpc,
            mu_min,
            mu_max,
            stuck_zone_prob,
            obstacle_mode,
            args.max_episode_steps,
            require_casadi=args.require_casadi,
        )
        for i in range(args.n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)
    if not args.no_vec_normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def build_eval_env(args, use_mpc, stuck_zone_prob, mu_min, mu_max, obstacle_mode):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from envs.tree_climber_env import TreeClimberEnv
    from mpc.climbing_mpc import ClimbingMPC

    eval_env = DummyVecEnv([lambda: Monitor(TreeClimberEnv(
        mpc=ClimbingMPC(require_casadi=args.require_casadi) if use_mpc else None,
        base_friction_range=(mu_min, mu_max),
        stuck_zone_prob=stuck_zone_prob,
        obstacle_mode=obstacle_mode,
        max_episode_steps=args.max_episode_steps,
    ), info_keywords=EPISODE_INFO_KEYS)])
    if not args.no_vec_normalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.training = False
    return eval_env


def copy_vecnormalize_stats(src, dst):
    if hasattr(src, "obs_rms") and hasattr(dst, "obs_rms"):
        dst.obs_rms = src.obs_rms
    if hasattr(src, "ret_rms") and hasattr(dst, "ret_rms"):
        dst.ret_rms = src.ret_rms


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.utils import set_random_seed

    from rl.residual_agent import RESIDUAL_POLICY_KWARGS

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "tb_logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    use_mpc = not args.no_mpc

    expert_obs, expert_actions = (None, None)
    if args.expert_transitions:
        expert_obs, expert_actions = load_expert_arrays(args.expert_transitions)

    vec_env = build_vec_env(
        args,
        use_mpc,
        args.normal_stuck_zone_prob,
        args.normal_mu_min,
        args.normal_mu_max,
        "none",
        seed_offset=0,
    )
    if expert_obs is not None and not args.no_vec_normalize:
        fit_vecnormalize_to_observations(vec_env, expert_obs)

    eval_env = build_eval_env(
        args,
        use_mpc,
        args.normal_stuck_zone_prob,
        args.normal_mu_min,
        args.normal_mu_max,
        "none",
    )
    copy_vecnormalize_stats(vec_env, eval_env)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.ppo_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=args.target_kl,
        policy_kwargs=RESIDUAL_POLICY_KWARGS,
        verbose=1,
        tensorboard_log=str(log_dir),
        device=args.device,
    )

    if args.normal_pretrain_timesteps > 0:
        print(
            f"[normal] pretraining stable climb for {args.normal_pretrain_timesteps:,} steps "
            f"(stuck_zone_prob={args.normal_stuck_zone_prob})"
        )
        normal_ckpt = ckpt_dir / "normal_pretrain_ppo"
        normal_metrics_cb = EpisodeMetricsCallback(run_dir / "normal_episode_metrics.csv")
        model.learn(
            total_timesteps=args.normal_pretrain_timesteps,
            callback=normal_metrics_cb,
            progress_bar=True,
        )
        model.save(str(normal_ckpt))
        if not args.no_vec_normalize:
            vec_env.save(str(ckpt_dir / "normal_pretrain_vec_normalize.pkl"))
        print(f"[normal] saved normal pretrain checkpoint to {normal_ckpt}.zip")

    if expert_obs is not None:
        behavior_clone_policy(
            model,
            vec_env,
            expert_obs,
            expert_actions,
            run_dir,
            args.bc_epochs,
            args.bc_batch_size,
            args.seed,
        )
        if not args.no_vec_normalize:
            vec_env.save(str(ckpt_dir / "bc_vec_normalize.pkl"))

    escape_vec_env = build_vec_env(
        args,
        use_mpc,
        args.stuck_zone_prob,
        args.mu_min,
        args.mu_max,
        args.obstacle_mode,
        seed_offset=10_000,
    )
    copy_vecnormalize_stats(vec_env, escape_vec_env)
    model.set_env(escape_vec_env)
    vec_env = escape_vec_env

    eval_env = build_eval_env(
        args,
        use_mpc,
        args.stuck_zone_prob,
        args.mu_min,
        args.mu_max,
        args.obstacle_mode,
    )
    copy_vecnormalize_stats(vec_env, eval_env)

    best_norm_cb = SaveVecNormalizeOnBestCallback(ckpt_dir / "best_model_vec_normalize.pkl")
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir),
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        callback_on_new_best=best_norm_cb,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(args.save_freq // max(args.n_envs, 1), 1),
        save_path=str(ckpt_dir),
        name_prefix="escape_ppo",
        save_vecnormalize=not args.no_vec_normalize,
    )
    escape_metrics_cb = EpisodeMetricsCallback(run_dir / "escape_episode_metrics.csv")

    print(f"[train] run_dir={run_dir}")
    print(f"[train] escape_timesteps={args.timesteps:,}, n_envs={args.n_envs}, obs={vec_env.observation_space.shape}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_cb, ckpt_cb, escape_metrics_cb],
        progress_bar=True,
        reset_num_timesteps=False,
    )

    final_path = ckpt_dir / "escape_ppo_final"
    model.save(str(final_path))
    if not args.no_vec_normalize:
        vec_env.save(str(ckpt_dir / "vec_normalize.pkl"))
    print(f"[train] saved final policy to {final_path}.zip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--expert-transitions", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stuck-zone-prob", type=float, default=0.8)
    parser.add_argument("--obstacle-mode", choices=["none", "random", "fixed"], default="random")
    parser.add_argument("--mu-min", type=float, default=0.60)
    parser.add_argument("--mu-max", type=float, default=1.10)
    parser.add_argument("--normal-pretrain-timesteps", type=int, default=250_000)
    parser.add_argument("--normal-stuck-zone-prob", type=float, default=0.0)
    parser.add_argument("--normal-mu-min", type=float, default=0.75)
    parser.add_argument("--normal-mu-max", type=float, default=1.10)
    parser.add_argument("--max-episode-steps", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-mpc", action="store_true")
    parser.add_argument("--require-casadi", action="store_true",
                        help="Fail instead of using the fallback MPC controller if CasADi is unavailable")
    parser.add_argument("--no-vec-normalize", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
