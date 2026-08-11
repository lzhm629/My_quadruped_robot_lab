#!/usr/bin/env python3
"""Interactively tune the OpenRobot wheel-fixed default joint pose.

The tool deliberately does not modify the URDF or training configuration.  Use
the UI to inspect a candidate pose, then print or save the values for review.

Examples:

    python My_quadruped_robot_lab/scripts/tools/tune_openrobot_default_pose.py

    python My_quadruped_robot_lab/scripts/tools/tune_openrobot_default_pose.py \
        --device cpu --headless --num_steps 10
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base_height", type=float, default=0.69, help="Initial base-link height in metres.")
parser.add_argument(
    "--save_file",
    type=Path,
    default=Path("openrobot_pose_candidate.json"),
    help="JSON path used by the Save candidate button.",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=None,
    help="Stop after this many steps. Required with --headless; intended for smoke tests.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply

from My_quadruped_robot_lab.assets.openrobot import OPENROBOT_WHEELFIXED_CFG, OPENROBOT_WHEELFIXED_URDF_PATH


LEGS = ("FL", "FR", "RL", "RR")
JOINT_TYPES = ("hip", "thigh", "calf")
TUNING_JOINTS = tuple(f"{leg}_{joint_type}_joint" for leg in LEGS for joint_type in JOINT_TYPES)
FOOT_BODIES = tuple(f"{leg}_foot" for leg in LEGS)
PREVIEW_LIMIT = 2.2
DEFAULT_MARKER_SCALE = (1.0, 1.0, 1.0)


def _read_stl_vertices(path: Path) -> list[tuple[float, float, float]]:
    """Read vertices from a binary or ASCII STL without adding a trimesh dependency."""
    data = path.read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + triangle_count * 50:
            vertices: list[tuple[float, float, float]] = []
            for triangle_index in range(triangle_count):
                offset = 84 + triangle_index * 50 + 12
                for vertex_index in range(3):
                    vertices.append(struct.unpack_from("<3f", data, offset + vertex_index * 12))
            return vertices

    vertices = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append(tuple(float(value) for value in fields[1:]))
    if not vertices:
        raise ValueError(f"No STL vertices found in {path}")
    return vertices


def _load_foot_collision_vertices(device: str) -> dict[str, torch.Tensor]:
    """Load each foot collision mesh in its link-local frame."""
    root = ET.parse(OPENROBOT_WHEELFIXED_URDF_PATH).getroot()
    vertices: dict[str, torch.Tensor] = {}
    for foot_name in FOOT_BODIES:
        link = root.find(f"./link[@name='{foot_name}']")
        if link is None:
            raise ValueError(f"Missing URDF link: {foot_name}")
        mesh = link.find("collision/geometry/mesh")
        if mesh is None or not mesh.get("filename"):
            raise ValueError(f"Foot collision is not an STL mesh: {foot_name}")
        mesh_path = (OPENROBOT_WHEELFIXED_URDF_PATH.parent / mesh.get("filename")).resolve()
        vertices[foot_name] = torch.tensor(_read_stl_vertices(mesh_path), dtype=torch.float32, device=device)
    return vertices


def _make_foot_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/OpenRobotFootSoles",
        markers={
            "below_ground": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            ),
            "near_ground": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1)),
            ),
            "clear": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.05)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


class PoseTunerUI:
    """Small omni.ui panel whose models are consumed by the simulation loop."""

    def __init__(
        self,
        joint_names: list[str],
        default_positions: dict[str, float],
        original_limits: dict[str, tuple[float, float]],
        base_height: float,
        save_file: Path,
    ) -> None:
        import omni.ui as ui

        self._ui = ui
        self.joint_names = joint_names
        self.default_positions = default_positions
        self.original_limits = original_limits
        self.save_file = save_file.resolve()
        self._reset_requested = False
        self._print_requested = False
        self._save_requested = False

        self.mirror_model = ui.SimpleBoolModel(True)
        self.pose_lock_model = ui.SimpleBoolModel(True)
        self.respect_limits_model = ui.SimpleBoolModel(True)
        self.base_height_model = ui.SimpleFloatModel(base_height)
        self.group_models: dict[str, object] = {}
        self.joint_models: dict[str, object] = {}

        self.window = ui.Window(
            "OpenRobot default-pose tuner",
            width=610,
            height=900,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )
        with self.window.frame:
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                with ui.VStack(spacing=6, height=0):
                    ui.Label("OpenRobot default-pose tuner", height=28, style={"font_size": 20})
                    ui.Label(
                        "Pose lock teleports the base and joints every frame for geometric inspection. "
                        "Turn it off to test the same target under gravity.",
                        word_wrap=True,
                        height=45,
                    )
                    self._checkbox_row("Mirror left/right joint coordinates", self.mirror_model)
                    self._checkbox_row("Pose lock (fixed geometric preview)", self.pose_lock_model)
                    self._checkbox_row("Respect original URDF joint limits", self.respect_limits_model)
                    self._float_row("Base height [m]", self.base_height_model, 0.15, 1.20, 0.001)

                    ui.Separator(height=8)
                    ui.Label("Mirrored controls (right-leg coordinate; left is negated)", height=24)
                    for joint_type in JOINT_TYPES:
                        reference_name = f"FR_{joint_type}_joint"
                        model = ui.SimpleFloatModel(default_positions[reference_name])
                        self.group_models[joint_type] = model
                        self._float_row(joint_type, model, -PREVIEW_LIMIT, PREVIEW_LIMIT, 0.001)

                    ui.Separator(height=8)
                    ui.Label("Independent 12-joint controls", height=24)
                    for joint_name in joint_names:
                        model = ui.SimpleFloatModel(default_positions[joint_name])
                        self.joint_models[joint_name] = model
                        lower, upper = original_limits[joint_name]
                        self._float_row(
                            f"{joint_name}  [{lower:+.3f}, {upper:+.3f}]",
                            model,
                            -PREVIEW_LIMIT,
                            PREVIEW_LIMIT,
                            0.001,
                        )

                    ui.Separator(height=8)
                    with ui.HStack(height=30, spacing=4):
                        ui.Button("Load training default", clicked_fn=self.load_defaults)
                        ui.Button("Load zero pose", clicked_fn=self.load_zero)
                        ui.Button("Reset physics", clicked_fn=self._request_reset)
                    with ui.HStack(height=30, spacing=4):
                        ui.Button("Print Python config", clicked_fn=self._request_print)
                        ui.Button("Save candidate JSON", clicked_fn=self._request_save)

                    ui.Label(f"Save path: {self.save_file}", word_wrap=True, height=34)
                    ui.Separator(height=8)
                    ui.Label("Live diagnostics", height=24)
                    self.diagnostics_label = ui.Label(
                        "Waiting for first simulation step...", word_wrap=True, height=190
                    )

    def _checkbox_row(self, label: str, model) -> None:
        ui = self._ui
        with ui.HStack(height=24):
            ui.Label(label, width=420)
            ui.CheckBox(model=model, width=24)

    def _float_row(self, label: str, model, lower: float, upper: float, step: float) -> None:
        ui = self._ui
        with ui.HStack(height=24, spacing=4):
            ui.Label(label, width=260)
            ui.FloatDrag(model=model, width=90, step=step)
            ui.FloatSlider(model=model, min=lower, max=upper, step=step, width=ui.Fraction(1))

    def load_defaults(self) -> None:
        for name, value in self.default_positions.items():
            self.joint_models[name].set_value(value)
        for joint_type in JOINT_TYPES:
            self.group_models[joint_type].set_value(self.default_positions[f"FR_{joint_type}_joint"])
        self._reset_requested = True

    def load_zero(self) -> None:
        for model in self.joint_models.values():
            model.set_value(0.0)
        for model in self.group_models.values():
            model.set_value(0.0)
        self._reset_requested = True

    def _request_reset(self) -> None:
        self._reset_requested = True

    def _request_print(self) -> None:
        self._print_requested = True

    def _request_save(self) -> None:
        self._save_requested = True

    def consume_requests(self) -> tuple[bool, bool, bool]:
        requests = (self._reset_requested, self._print_requested, self._save_requested)
        self._reset_requested = False
        self._print_requested = False
        self._save_requested = False
        return requests

    def requested_positions(self) -> dict[str, float]:
        if self.mirror_model.as_bool:
            result = {}
            for name in self.joint_names:
                leg, joint_type, _ = name.split("_", maxsplit=2)
                value = self.group_models[joint_type].as_float
                result[name] = value if leg in ("FR", "RR") else -value
                if abs(self.joint_models[name].as_float - result[name]) > 1.0e-6:
                    self.joint_models[name].set_value(result[name])
            return result
        return {name: model.as_float for name, model in self.joint_models.items()}

    @property
    def base_height(self) -> float:
        return self.base_height_model.as_float

    @property
    def pose_locked(self) -> bool:
        return self.pose_lock_model.as_bool

    @property
    def respect_limits(self) -> bool:
        return self.respect_limits_model.as_bool

    def set_diagnostics(self, text: str, warning: bool) -> None:
        self.diagnostics_label.text = text
        self.diagnostics_label.style = {"color": 0xFF6666FF if warning else 0xFF88DD88}


def _ordered_joint_metadata(robot: Articulation) -> tuple[list[str], list[int]]:
    joint_ids, joint_names = robot.find_joints(list(TUNING_JOINTS), preserve_order=True)
    if joint_names != list(TUNING_JOINTS):
        raise RuntimeError(f"Expected joints {TUNING_JOINTS}, received {joint_names}")
    return joint_names, joint_ids


def _positions_tensor(
    robot: Articulation,
    joint_names: list[str],
    joint_ids: list[int],
    requested: dict[str, float],
    original_limits: torch.Tensor,
    respect_limits: bool,
) -> tuple[torch.Tensor, list[str]]:
    positions = robot.data.joint_pos.clone()
    requested_values = torch.tensor(
        [[requested[name] for name in joint_names]], dtype=positions.dtype, device=positions.device
    )
    lower = original_limits[:, joint_ids, 0]
    upper = original_limits[:, joint_ids, 1]
    violations = [
        name
        for index, name in enumerate(joint_names)
        if requested[name] < lower[0, index].item() or requested[name] > upper[0, index].item()
    ]
    if respect_limits:
        requested_values = torch.clamp(requested_values, lower, upper)
    positions[:, joint_ids] = requested_values
    return positions, violations


def _write_pose(robot: Articulation, joint_positions: torch.Tensor, base_height: float, reset_actuators: bool) -> None:
    root_pose = robot.data.default_root_state[:, :7].clone()
    root_pose[:, :3] = 0.0
    root_pose[:, 2] = base_height
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.default_root_state[:, 7:]))
    robot.write_joint_state_to_sim(joint_positions, torch.zeros_like(joint_positions))
    if reset_actuators:
        robot.reset()


def _sole_points(
    robot: Articulation,
    foot_names: list[str],
    foot_ids: list[int],
    local_vertices: dict[str, torch.Tensor],
) -> torch.Tensor:
    points = []
    for foot_name, body_id in zip(foot_names, foot_ids, strict=True):
        vertices = local_vertices[foot_name]
        quat = robot.data.body_quat_w[0, body_id].expand(vertices.shape[0], -1)
        world_vertices = quat_apply(quat, vertices) + robot.data.body_pos_w[0, body_id]
        points.append(world_vertices[torch.argmin(world_vertices[:, 2])])
    return torch.stack(points)


def _format_python_config(requested: dict[str, float], mirrored: bool, base_height: float) -> str:
    lines = ["pos=(0.0, 0.0, %.6f)," % base_height, "joint_pos={"]
    if mirrored:
        for joint_type in JOINT_TYPES:
            right_value = requested[f"FR_{joint_type}_joint"]
            left_value = requested[f"FL_{joint_type}_joint"]
            lines.append(f'    "^(FR|RR)_{joint_type}_joint$": {right_value:+.6f},')
            lines.append(f'    "^(FL|RL)_{joint_type}_joint$": {left_value:+.6f},')
    else:
        for name in TUNING_JOINTS:
            lines.append(f'    "^{name}$": {requested[name]:+.6f},')
    lines.append("},")
    return "\n".join(lines)


def _save_candidate(
    path: Path,
    requested: dict[str, float],
    base_height: float,
    violations: list[str],
    original_limits: dict[str, tuple[float, float]],
) -> None:
    payload = {
        "base_height": base_height,
        "joint_pos_rad": requested,
        "joint_pos_deg": {name: math.degrees(value) for name, value in requested.items()},
        "outside_original_urdf_limits": violations,
        "original_urdf_limits_rad": original_limits,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO]: Saved candidate pose to {path}")


def main() -> None:
    if args_cli.headless and args_cli.num_steps is None:
        raise ValueError("This is a GUI tool. With --headless, also pass --num_steps for a finite smoke test.")
    if args_cli.num_steps is not None and args_cli.num_steps < 1:
        raise ValueError("--num_steps must be positive")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim.set_camera_view(eye=(2.6, 2.3, 1.5), target=(-0.2, 0.0, 0.35))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = OPENROBOT_WHEELFIXED_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    sim.reset()

    joint_names, joint_ids = _ordered_joint_metadata(robot)
    foot_ids, foot_names = robot.find_bodies(list(FOOT_BODIES), preserve_order=True)
    if foot_names != list(FOOT_BODIES):
        raise RuntimeError(f"Expected feet {FOOT_BODIES}, received {foot_names}")

    original_limits_tensor = robot.data.joint_pos_limits.clone()
    original_limits = {
        name: (
            original_limits_tensor[0, joint_id, 0].item(),
            original_limits_tensor[0, joint_id, 1].item(),
        )
        for name, joint_id in zip(joint_names, joint_ids, strict=True)
    }
    defaults = {
        name: robot.data.default_joint_pos[0, joint_id].item()
        for name, joint_id in zip(joint_names, joint_ids, strict=True)
    }
    foot_vertices = _load_foot_collision_vertices(robot.device)
    markers = None if args_cli.headless else _make_foot_markers()
    tuner = (
        None
        if args_cli.headless
        else PoseTunerUI(joint_names, defaults, original_limits, args_cli.base_height, args_cli.save_file)
    )

    requested = defaults.copy()
    base_height = args_cli.base_height
    respect_limits = True
    pose_locked = True
    mirrored = True
    previous_respect_limits = True
    previous_pose_locked = True
    reset_requested = True
    violations: list[str] = []
    step = 0

    print("[INFO]: OpenRobot pose tuner started")
    print(f"[INFO]: joint_names={joint_names}")
    print(f"[INFO]: original_limits={original_limits}")

    while simulation_app.is_running() and (args_cli.num_steps is None or step < args_cli.num_steps):
        if tuner is not None:
            requested = tuner.requested_positions()
            base_height = tuner.base_height
            respect_limits = tuner.respect_limits
            pose_locked = tuner.pose_locked
            mirrored = tuner.mirror_model.as_bool
            ui_reset, print_requested, save_requested = tuner.consume_requests()
            reset_requested = reset_requested or ui_reset
        else:
            print_requested = False
            save_requested = False

        if respect_limits != previous_respect_limits:
            if respect_limits:
                robot.write_joint_position_limit_to_sim(original_limits_tensor)
            else:
                preview_limits = original_limits_tensor.clone()
                preview_limits[..., 0] = -PREVIEW_LIMIT
                preview_limits[..., 1] = PREVIEW_LIMIT
                robot.write_joint_position_limit_to_sim(preview_limits, warn_limit_violation=False)
            previous_respect_limits = respect_limits
            reset_requested = True

        joint_positions, violations = _positions_tensor(
            robot, joint_names, joint_ids, requested, original_limits_tensor, respect_limits
        )
        robot.set_joint_position_target(joint_positions)

        if pose_locked or reset_requested or pose_locked != previous_pose_locked:
            _write_pose(robot, joint_positions, base_height, reset_actuators=reset_requested)
            reset_requested = False
        previous_pose_locked = pose_locked

        if print_requested:
            print("\n[OPENROBOT CANDIDATE]\n" + _format_python_config(requested, mirrored, base_height))
            if violations:
                print(f"[WARNING]: Outside original URDF limits: {violations}")
        if save_requested:
            _save_candidate(tuner.save_file, requested, base_height, violations, original_limits)

        robot.write_data_to_sim()
        sim.step(render=not args_cli.headless)
        robot.update(sim.get_physics_dt())

        soles = _sole_points(robot, foot_names, foot_ids, foot_vertices)
        if markers is not None:
            marker_indices = torch.where(
                soles[:, 2] < -0.002,
                torch.zeros(4, dtype=torch.int32, device=robot.device),
                torch.where(
                    soles[:, 2] <= 0.01,
                    torch.ones(4, dtype=torch.int32, device=robot.device),
                    torch.full((4,), 2, dtype=torch.int32, device=robot.device),
                ),
            )
            markers.visualize(
                translations=soles,
                scales=torch.tensor([DEFAULT_MARKER_SCALE] * 4, device=robot.device),
                marker_indices=marker_indices,
            )

        if tuner is not None and step % 5 == 0:
            sole_rows = [
                f"{name[:2]}=({point[0].item():+.3f}, {point[1].item():+.3f}, {point[2].item():+.3f})"
                for name, point in zip(foot_names, soles, strict=True)
            ]
            support_x = soles[:, 0].max().item() - soles[:, 0].min().item()
            support_y = soles[:, 1].max().item() - soles[:, 1].min().item()
            warning_lines = []
            if violations:
                warning_lines.append("OUTSIDE ORIGINAL LIMITS: " + ", ".join(violations))
                if respect_limits:
                    warning_lines.append("Applied joint state is clamped to the original URDF limits.")
            if not respect_limits:
                warning_lines.append("Preview limits active; this pose is not a valid URDF stability test.")
            text = "\n".join(
                [
                    f"Mode: {'POSE LOCK' if pose_locked else 'GRAVITY / PD'} | root z={robot.data.root_pos_w[0, 2].item():.3f} m",
                    "Sole xyz [m]: " + "  ".join(sole_rows),
                    f"Lowest sole z={soles[:, 2].min().item():+.4f} m | support span x/y={support_x:.3f}/{support_y:.3f} m",
                    *warning_lines,
                ]
            )
            tuner.set_diagnostics(text, warning=bool(warning_lines) or soles[:, 2].min().item() < -0.002)

        step += 1

    print(f"[INFO]: Completed {step} steps")
    print("[INFO]: Final candidate:\n" + _format_python_config(requested, mirrored, base_height))
    if violations:
        print(f"[WARNING]: Outside original URDF limits: {violations}")
    sim.stop()
    sim.clear()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
