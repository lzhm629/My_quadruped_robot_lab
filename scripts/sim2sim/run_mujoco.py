"""Run an IsaacLab-trained OpenRobot trot policy in MuJoCo."""

from __future__ import annotations

import argparse
import contextlib
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from config import DEFAULT_MODEL_PATH, DEFAULT_POLICY_PATH, JOINT_NAMES, PolicyConfig, RobotConfig
from controller import DelayedPDController
from observations import ObservationHistory, build_observation_terms


@dataclass
class ModelAddresses:
    joint_qpos: np.ndarray
    joint_dof: np.ndarray
    base_body_id: int


class CommandSource:
    def __init__(self, mode: str, command: np.ndarray):
        self.mode = mode
        self.command = command.astype(np.float64, copy=True)
        self.limits = np.array([0.8, 0.5, 0.6], dtype=np.float64)

    def key_callback(self, keycode: int) -> None:
        if self.mode != "keyboard":
            return
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        delta = np.array([0.1, 0.1, 0.1])
        if key == "w":
            self.command[0] += delta[0]
        elif key == "s":
            self.command[0] -= delta[0]
        elif key == "a":
            self.command[1] += delta[1]
        elif key == "d":
            self.command[1] -= delta[1]
        elif key == "q":
            self.command[2] += delta[2]
        elif key == "e":
            self.command[2] -= delta[2]
        elif key == " ":
            self.command[:] = 0.0
        else:
            return
        self.command[:] = np.clip(self.command, -self.limits, self.limits)
        print(
            f"[COMMAND] vx={self.command[0]:+.2f} m/s, vy={self.command[1]:+.2f} m/s, "
            f"yaw={self.command[2]:+.2f} rad/s"
        )


def resolve_addresses(model: mujoco.MjModel) -> ModelAddresses:
    qpos_addresses = []
    dof_addresses = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing policy joint {name!r}")
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        dof_addresses.append(model.jnt_dofadr[joint_id])
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_body_id < 0:
        raise ValueError("MuJoCo model is missing body 'base_link'")
    return ModelAddresses(np.array(qpos_addresses), np.array(dof_addresses), base_body_id)


def reset_simulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    addresses: ModelAddresses,
    robot_cfg: RobotConfig,
    controller: DelayedPDController,
    history: ObservationHistory,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = (0.0, 0.0, 0.69)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[addresses.joint_qpos] = robot_cfg.default_joint_pos
    controller.reset()
    history.reset()
    mujoco.mj_forward(model, data)


def base_observations(
    model: mujoco.MjModel, data: mujoco.MjData, addresses: ModelAddresses
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, addresses.base_body_id, velocity, 1
    )
    rotation_world_from_body = data.xmat[addresses.base_body_id].reshape(3, 3)
    projected_gravity = rotation_world_from_body.T @ np.array([0.0, 0.0, -1.0])
    return velocity[:3].copy(), projected_gravity


def open_log(path: Path | None):
    if path is None:
        return contextlib.nullcontext(), None
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(file_handle)
    writer.writerow(
        ["time", "command_x", "command_y", "command_yaw", "base_x", "base_y", "base_z", "max_abs_torque"]
    )
    return contextlib.closing(file_handle), writer


def run(args: argparse.Namespace) -> None:
    policy_cfg = PolicyConfig()
    robot_cfg = RobotConfig()
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    model.opt.timestep = policy_cfg.physics_dt
    data = mujoco.MjData(model)
    addresses = resolve_addresses(model)
    policy = torch.jit.load(str(args.policy.resolve()), map_location="cpu").eval()
    controller = DelayedPDController(robot_cfg, args.hip_thigh_delay, args.calf_delay)
    observation_history = ObservationHistory(policy_cfg.history_length)
    command_source = CommandSource(args.mode, np.array([args.vx, args.vy, args.yaw]))
    reset_simulation(model, data, addresses, robot_cfg, controller, observation_history)

    previous_action = np.zeros(12, dtype=np.float64)
    target_joint_pos = robot_cfg.default_joint_pos.copy()
    policy_step = 0
    physics_step = 0
    start_wall_time = time.perf_counter()
    print(f"[INFO] mode={args.mode}, policy={args.policy}")
    print(
        f"[INFO] command vx={args.vx:+.2f}, vy={args.vy:+.2f}, yaw={args.yaw:+.2f}; "
        f"delays hip/thigh={args.hip_thigh_delay}, calf={args.calf_delay} physics steps"
    )
    if args.mode == "keyboard":
        print("[KEYBOARD] W/S: forward/back, A/D: left/right, Q/E: yaw, Space: stop")

    viewer_context = (
        contextlib.nullcontext(None)
        if args.headless
        else mujoco.viewer.launch_passive(model, data, key_callback=command_source.key_callback)
    )
    log_context, log_writer = open_log(args.log)
    with viewer_context as viewer, log_context:
        if viewer is not None:
            viewer.cam.trackbodyid = addresses.base_body_id
            viewer.cam.distance = 4.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -20.0

        while (viewer is None or viewer.is_running()) and (args.duration <= 0 or data.time < args.duration):
            step_start = time.perf_counter()
            joint_pos = data.qpos[addresses.joint_qpos].copy()
            joint_vel = data.qvel[addresses.joint_dof].copy()

            if physics_step % policy_cfg.decimation == 0:
                base_ang_vel, projected_gravity = base_observations(model, data, addresses)
                terms = build_observation_terms(
                    policy_step=policy_step,
                    policy_dt=policy_cfg.policy_dt,
                    cycle_time=policy_cfg.cycle_time,
                    command=command_source.command,
                    base_angular_velocity=base_ang_vel,
                    projected_gravity=projected_gravity,
                    joint_pos=joint_pos,
                    joint_vel=joint_vel,
                    default_joint_pos=robot_cfg.default_joint_pos,
                    previous_action=previous_action,
                )
                observation = observation_history.update(terms)
                with torch.inference_mode():
                    action_tensor = policy(torch.from_numpy(observation).unsqueeze(0))
                previous_action = np.clip(
                    action_tensor.squeeze(0).cpu().numpy().astype(np.float64),
                    -policy_cfg.action_clip,
                    policy_cfg.action_clip,
                )
                target_joint_pos = controller.action_to_target(previous_action, policy_cfg.action_clip)
                policy_step += 1

            torque = controller.compute(target_joint_pos, joint_pos, joint_vel)
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[addresses.joint_dof] = torque
            mujoco.mj_step(model, data)
            physics_step += 1

            if log_writer is not None and physics_step % policy_cfg.decimation == 0:
                log_writer.writerow(
                    [
                        f"{data.time:.6f}",
                        *command_source.command.tolist(),
                        *data.qpos[0:3].tolist(),
                        float(np.max(np.abs(torque))),
                    ]
                )
            if viewer is not None:
                viewer.sync()
            if args.real_time:
                sleep_duration = policy_cfg.physics_dt - (time.perf_counter() - step_start)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

    elapsed = time.perf_counter() - start_wall_time
    print(f"[INFO] completed {data.time:.2f} simulated seconds in {elapsed:.2f} wall seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("default", "keyboard"), default="default")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vx", type=float, default=0.8, help="Initial/preset forward velocity in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Initial/preset lateral velocity in m/s.")
    parser.add_argument("--yaw", type=float, default=0.6, help="Initial/preset yaw velocity in rad/s.")
    parser.add_argument("--duration", type=float, default=120.0, help="Simulation duration; <=0 runs until closed.")
    parser.add_argument("--hip-thigh-delay", type=int, default=2, choices=range(0, 4))
    parser.add_argument("--calf-delay", type=int, default=2, choices=range(0, 4))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()
    if args.mode == "keyboard" and args.headless:
        parser.error("keyboard mode requires the viewer; remove --headless")
    for value, limit, name in zip((args.vx, args.vy, args.yaw), (0.8, 0.5, 0.6), ("vx", "vy", "yaw")):
        if abs(value) > limit:
            parser.error(f"--{name} must be within [-{limit}, {limit}], matching the training command range")
    if not args.policy.is_file():
        parser.error(f"policy does not exist: {args.policy}")
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}; run generate_openrobot_mjcf.py first")
    return args


if __name__ == "__main__":
    run(parse_args())
