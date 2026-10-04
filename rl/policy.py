"""PPO checkpoints and paired observation normalization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from envs.navigation_env import NavigationEnv

SCHEMA = "palmclimber_sensor_navigation"


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_env(config, seed):
    def create():
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(NavigationEnv(config))
        env.reset(seed=seed)
        return env
    return create


def build_vector(config, seed, n_envs=1):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    factories = [make_env(config, seed + 10000 * i) for i in range(n_envs)]
    if n_envs > 1:
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


def checkpoint_paths(path):
    model_path = Path(path).with_suffix(".zip")
    return model_path, model_path.with_suffix(".vecnormalize.pkl"), model_path.with_suffix(".json")


def save_policy(model, norm, path, manifest):
    model_path, norm_path, meta_path = checkpoint_paths(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    norm.save(str(norm_path))
    write_json(meta_path, {
        "schema": SCHEMA, "timesteps": int(model.num_timesteps),
        "model_sha256": file_hash(model_path),
        "normalization_sha256": file_hash(norm_path), "manifest": manifest,
    })


def load_policy(path, *, device="cpu"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    model_path, norm_path, meta_path = checkpoint_paths(path)
    metadata = json.loads(meta_path.read_text())
    if metadata.get("schema") != SCHEMA:
        raise ValueError("Expected a PalmClimber navigation checkpoint")
    for file, key in ((model_path, "model_sha256"), (norm_path, "normalization_sha256")):
        if file_hash(file) != metadata[key]:
            raise ValueError(f"Checkpoint checksum changed: {file.name}")
    config = metadata["manifest"]["config"]
    vector = build_vector(config, 0)
    if vector.env_method("contract")[0] != metadata["manifest"]["contract"]:
        vector.close()
        raise ValueError("Checkpoint and environment contracts differ")
    try:
        norm = VecNormalize.load(str(norm_path), vector)
        norm.training = False
        norm.norm_reward = False
        model = PPO.load(str(model_path), device=device)
        model.policy.set_training_mode(False)
        if model.observation_space != norm.observation_space or model.action_space != norm.action_space:
            raise ValueError("Checkpoint and environment spaces differ")
    except Exception:
        vector.close()
        raise
    return model, norm, metadata
