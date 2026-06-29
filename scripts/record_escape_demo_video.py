"""
Render a visual escape demo from the same base-only MuJoCo scene used for
training: climb, hit a zero-friction patch, rotate around the trunk, then
continue climbing.

Usage:
    MUJOCO_GL=egl python scripts/record_escape_demo_video.py
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = ROOT / "mujoco_models" / "scene.xml"
DEFAULT_OUTPUT = ROOT / "results" / "palmclimber_escape_demo.mp4"
ESCAPE_YAW = np.deg2rad(30.0)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def smoothstep(x: float) -> float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id >= 0:
        data.qpos[model.jnt_qposadr[joint_id]] = value


def overlay(frame: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    x, y = 16, 14
    pad = 8
    box = draw.textbbox((x, y), label)
    draw.rounded_rectangle(
        (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
        radius=5,
        fill=(0, 0, 0, 135),
    )
    draw.text((x, y), label, fill=(255, 255, 255))
    return np.asarray(img)


def trajectory(t: float) -> tuple[float, float, str]:
    if t < 0.34:
        p = smoothstep(t / 0.34)
        return 0.46 + 0.70 * p, 0.0, "normal climb"
    if t < 0.52:
        wiggle = 0.012 * np.sin(80.0 * t)
        return 1.16 + wiggle, 0.0, "stuck: wheel hits local zero-friction spot"
    if t < 0.68:
        p = smoothstep((t - 0.52) / 0.16)
        return 1.16 - 0.04 * p, ESCAPE_YAW * p, "escape: rotate into wheel gap"
    p = smoothstep((t - 0.68) / 0.32)
    return 1.12 + 1.05 * p, ESCAPE_YAW, "recovered: climb on good friction"


def pose_robot(model: mujoco.MjModel, data: mujoco.MjData, t: float):
    height, yaw, phase = trajectory(t)
    data.qpos[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, height]
    data.qpos[3:7] = yaw_quat(yaw)

    wheel_phase = 58.0 * t
    for idx in range(1, 7):
        direction = 1.0 if idx % 2 else -1.0
        set_joint_qpos(model, data, f"wheel{idx}_spin", direction * wheel_phase)

    mujoco.mj_forward(model, data)
    return height, yaw, phase


def render_video(output: Path, seconds: float, fps: int, width: int, height_px: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height_px, width=width)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.75
    camera.azimuth = -58.0
    camera.elevation = -12.0

    nframes = int(seconds * fps)
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for idx in range(nframes):
            t = idx / max(nframes - 1, 1)
            robot_height, yaw, phase = pose_robot(model, data, t)
            camera.lookat[:] = [0.0, 0.0, robot_height + 0.38]
            camera.azimuth = -55.0 + 16.0 * smoothstep(t)
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            frame = overlay(
                frame,
                f"zero-friction patch demo | {phase} | yaw {np.rad2deg(yaw):.0f} deg",
            )
            writer.append_data(frame)

    renderer.close()
    print(f"saved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    render_video(args.output, args.seconds, args.fps, args.width, args.height)


if __name__ == "__main__":
    main()
