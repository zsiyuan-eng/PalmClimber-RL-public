"""
Deploy trained policy to the real robot.

Pipeline:
    1. Load trained PPO climbing model + VecNormalize stats
    2. Connect to Arduino over Serial
    3. Read IMU (gyro pitch/roll) from Arduino feedback
    4. Build observation, run policy forward pass
    5. Compute u_mpc (from ClimbingMPC) + Δu_rl (from policy)
    6. Send final wheel speed commands to Arduino

The Arduino protocol is the same single-char protocol from the original
control_arduinoside.ino -- here we extend it with numeric velocity commands.
Update ARDUINO_MODE in control_arduinoside.ino to accept "Vw1:12.5\n" style
messages for direct velocity override (see deploy/control_arduinoside_v2.ino).

Usage:
    python deploy/run_policy.py --port COM12 --target-height 2.0
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mpc.climbing_mpc import ClimbingMPC
from envs.tree_climber_env import (
    CONTROL_U_MAX,
    MAX_AZIMUTH_RATE,
    MAX_HEIGHT_PROGRESS_RATE,
    MIN_HEIGHT_PROGRESS_RATE,
    MIN_UPWARD_COMMAND,
    MPC_WEIGHT_MIN,
    RESIDUAL_SCALE_NORMAL,
    RESIDUAL_SCALE_STUCK,
    STUCK_AUTHORITY_TIME,
    TARGET_HEIGHT_TOL,
)


def load_models(model_dir: str, model_name: str):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        from envs.tree_climber_env import TreeClimberEnv

        model_path = os.path.join(model_dir, model_name)
        norm_path  = os.path.join(model_dir, "vec_normalize.pkl")

        dummy_env = DummyVecEnv([lambda: TreeClimberEnv()])
        expected_shape = dummy_env.observation_space.shape
        if os.path.exists(norm_path):
            dummy_env = VecNormalize.load(norm_path, dummy_env)
            dummy_env.training = False
            dummy_env.norm_reward = False

        model = PPO.load(model_path, env=dummy_env)
        if model.observation_space.shape != expected_shape:
            raise ValueError(
                f"model obs shape {model.observation_space.shape} != current env {expected_shape}; "
                "old checkpoints/vec_normalize are incompatible and must be retrained"
            )
        print(f"[deploy] loaded PPO model from {model_path}")
        return model, dummy_env
    except Exception as e:
        print(f"[deploy] could not load model: {e}")
        print("[deploy] running MPC-only mode")
        return None, None


def open_serial(port: str, baudrate: int = 115200, timeout: float = 0.02):
    import serial
    ser = serial.Serial(port, baudrate, timeout=timeout)
    time.sleep(2.0)
    print(f"[deploy] serial opened on {port}")
    return ser


def read_imu(ser, state: dict) -> dict:
    """
    Expect Arduino to send lines like:
        IMU pitch=0.03 roll=-0.01 height_est=1.45
    Falls back to zeros if parse fails.
    """
    state = state.copy()
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if "pitch=" in line:
            for tok in line.split():
                if "=" not in tok:
                    continue
                k, v = tok.split("=", 1)
                if k == "pitch":  state["tilt_x"] = float(v)
                if k == "roll":   state["tilt_y"] = float(v)
                if k == "gyro_x": state["tilt_rate_x"] = float(v)
                if k == "gyro_y": state["tilt_rate_y"] = float(v)
                if k == "gyro_z":
                    state["azimuth_rate"] = float(v)
                    state["_azimuth_rate_measured"] = True
                if k == "yaw_rate":
                    state["azimuth_rate"] = float(v)
                    state["_azimuth_rate_measured"] = True
                if k == "yaw": state["azimuth_est"] = float(v)
                if k == "height_est": state["height"] = float(v)
    except Exception:
        pass
    return state


def smooth_authority(stuck_timer, height, target_height, height_gain_rate, u_mpc, dt):
    commanding_up = np.mean(np.abs(u_mpc)) > MIN_UPWARD_COMMAND
    not_near_target = height < target_height - TARGET_HEIGHT_TOL
    low_progress = height_gain_rate < MIN_HEIGHT_PROGRESS_RATE

    if commanding_up and not_near_target and low_progress:
        stuck_timer += dt
    else:
        stuck_timer = max(0.0, stuck_timer - 2.0 * dt)

    stuck_raw = np.clip(stuck_timer / STUCK_AUTHORITY_TIME, 0.0, 1.0)
    stuck_level = stuck_raw * stuck_raw * (3.0 - 2.0 * stuck_raw)
    mpc_weight = 1.0 - (1.0 - MPC_WEIGHT_MIN) * stuck_level
    residual_scale = (
        RESIDUAL_SCALE_NORMAL
        + (RESIDUAL_SCALE_STUCK - RESIDUAL_SCALE_NORMAL) * stuck_level
    )
    return stuck_timer, stuck_level, mpc_weight, residual_scale


def compute_magic_lateral_cmd(u_final):
    u = np.asarray(u_final, dtype=np.float64)
    alternating = np.mean(np.array([u[0], -u[1], u[2], -u[3], u[4], -u[5]]))
    differential = np.mean(u[[0, 2, 4]]) - np.mean(u[[1, 3, 5]])
    alternating_cmd = float(np.clip(alternating / CONTROL_U_MAX, -1.0, 1.0))
    differential_cmd = float(np.clip(differential / CONTROL_U_MAX, -1.0, 1.0))
    return float(np.clip(0.6 * alternating_cmd + 0.4 * differential_cmd, -1.0, 1.0))


def build_policy_obs(state, target_height, last_u, u_mpc, stuck_level, mpc_weight, residual_scale, magic_lateral_cmd_prev):
    height = state["height"]
    velocity = state["velocity"]
    azimuth_est = state.get("azimuth_est", state.get("yaw", 0.0))
    obs = np.array([
        height, velocity,
        state["tilt_x"], state["tilt_y"],
        state["tilt_rate_x"], state["tilt_rate_y"],
        state.get("yaw", 0.0),
        np.clip(target_height - height, -3, 3),
        *(last_u[:4] / CONTROL_U_MAX),
        stuck_level,
        np.clip(state.get("height_gain_rate", 0.0) / MAX_HEIGHT_PROGRESS_RATE, -1.0, 1.0),
        0.0,  # azimuth error placeholder; real robot currently lacks this estimate
        np.clip((target_height - height) / 2.0, -1.0, 1.0),
        *(np.clip(u_mpc / CONTROL_U_MAX, -1.0, 1.0)),
        stuck_level,
        mpc_weight,
        residual_scale / RESIDUAL_SCALE_STUCK,
        np.sin(azimuth_est),
        np.cos(azimuth_est),
        np.clip(state.get("azimuth_rate", 0.0) / MAX_AZIMUTH_RATE, -1.0, 1.0),
        magic_lateral_cmd_prev,
    ], dtype=np.float32)
    return obs.reshape(1, -1)


def send_velocities(ser, velocities: np.ndarray):
    """
    Send 6 wheel velocity commands as 'V1:12.5,2:12.5,...\n'
    The updated Arduino firmware parses this format.
    """
    cmd = "V" + ",".join(f"{i+1}:{v:.1f}" for i, v in enumerate(velocities)) + "\n"
    ser.write(cmd.encode("utf-8"))


def run_loop(args):
    mpc   = ClimbingMPC(verbose=False)
    model, norm_env = load_models(
        os.path.join(os.path.dirname(__file__), "..", "checkpoints"),
        args.model,
    )

    ser = None
    if args.port:
        ser = open_serial(args.port)
    else:
        print("[deploy] no serial port specified -- running in simulation-only print mode")

    # state estimate (updated from IMU each cycle)
    state = {
        "height": 0.4, "velocity": 0.0,
        "tilt_x": 0.0, "tilt_y": 0.0,
        "tilt_rate_x": 0.0, "tilt_rate_y": 0.0,
        "azimuth_est": 0.0, "azimuth_rate": 0.0,
    }
    last_u = np.zeros(6)
    last_height = state["height"]
    last_time = time.time()
    stuck_timer = 0.0
    stuck_level = 0.0
    mpc_weight = 1.0
    residual_scale = RESIDUAL_SCALE_NORMAL
    target_height = args.target_height
    magic_lateral_cmd_prev = 0.0

    print(f"[deploy] climbing to {target_height}m -- press Ctrl+C to stop")
    try:
        while True:
            t0 = time.time()
            dt = max(t0 - last_time, 1e-3)

            # read IMU
            state["_azimuth_rate_measured"] = False
            if ser:
                state = read_imu(ser, state)

            raw_velocity = (state["height"] - last_height) / dt
            state["velocity"] = 0.8 * state["velocity"] + 0.2 * raw_velocity
            state["height_gain_rate"] = state["velocity"]
            last_height = state["height"]
            last_time = t0

            # MPC nominal action
            u_mpc = mpc.solve(state, target_height=target_height)
            stuck_timer, stuck_level, mpc_weight, residual_scale = smooth_authority(
                stuck_timer,
                state["height"],
                target_height,
                state["height_gain_rate"],
                u_mpc,
                dt,
            )

            # RL residual (if model loaded)
            if model is not None:
                obs_arr = build_policy_obs(
                    state,
                    target_height,
                    last_u,
                    u_mpc,
                    stuck_level,
                    mpc_weight,
                    residual_scale,
                    magic_lateral_cmd_prev,
                )

                if norm_env is not None:
                    obs_arr = norm_env.normalize_obs(obs_arr)

                action, _ = model.predict(obs_arr, deterministic=True)
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("policy produced non-finite action")
                delta_u   = action.flatten() * residual_scale
            else:
                delta_u = np.zeros(6)

            u_final = np.clip(mpc_weight * u_mpc + delta_u, -CONTROL_U_MAX, CONTROL_U_MAX)
            last_u  = u_final.copy()
            magic_lateral_cmd_prev = compute_magic_lateral_cmd(u_final)
            if not state.get("_azimuth_rate_measured", False):
                state["azimuth_rate"] = MAX_AZIMUTH_RATE * magic_lateral_cmd_prev
            state["azimuth_est"] = float(
                (state.get("azimuth_est", 0.0) + state["azimuth_rate"] * dt + np.pi)
                % (2 * np.pi) - np.pi
            )

            print(f"  h={state['height']:.3f}m  tilt=({state['tilt_x']:.3f},{state['tilt_y']:.3f})  "
                  f"stuck={stuck_level:.2f}  mpc_w={mpc_weight:.2f}  "
                  f"res_scale={residual_scale:.1f}  u_mpc={u_mpc[0]:.1f}  "
                  f"Δu={delta_u[0]:.2f}  → {u_final[0]:.1f}")

            if ser:
                send_velocities(ser, u_final)

            # check if target reached
            if state["height"] >= target_height - 0.05:
                print("[deploy] target height reached! stopping wheels.")
                if ser:
                    send_velocities(ser, np.zeros(6))
                break

            if abs(state["tilt_x"]) > 0.6 or abs(state["tilt_y"]) > 0.6:
                raise RuntimeError("tilt safety stop")

            # maintain ~50Hz loop
            elapsed = time.time() - t0
            time.sleep(max(0, 0.02 - elapsed))

    except KeyboardInterrupt:
        print("\n[deploy] interrupted")
    finally:
        if ser:
            send_velocities(ser, np.zeros(6))
            ser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",          type=str,   default=None,
                        help="Serial port (e.g. COM12 or /dev/ttyUSB0)")
    parser.add_argument("--target-height", type=float, default=2.0,
                        help="Target height in meters")
    parser.add_argument("--model", type=str, default="best_model",
                        help="Checkpoint name under checkpoints/ (e.g. best_model or climber_ppo_final)")
    args = parser.parse_args()
    run_loop(args)
