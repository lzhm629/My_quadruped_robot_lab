#!/usr/bin/env python3
"""Load the wheel-fixed OpenRobot URDF and run a short physics smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_URDF = (
    Path(__file__).resolve().parents[3]
    / "openrobot_urdf_new"
    / "urdf"
    / "openrobot_urdf_new_wheelfixed.urdf"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext


LEG_JOINTS = [
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
]
FOOT_BODIES = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")]


def main() -> None:
    urdf_path = args_cli.urdf.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)

    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1500.0))
    prim_utils.create_prim("/World/RobotOrigin", "Xform")

    robot_cfg = ArticulationCfg(
        prim_path="/World/RobotOrigin/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path),
            activate_contact_sensors=True,
            force_usd_conversion=True,
            fix_base=False,
            merge_fixed_joints=False,
            self_collision=False,
            make_instanceable=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.55), joint_pos={".*": 0.0}),
        actuators={
            "setz120": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
                effort_limit_sim=117.0,
                velocity_limit_sim=12.46,
                stiffness=0.0,
                damping=0.0,
                armature=0.133886,
            ),
            "setz160": ImplicitActuatorCfg(
                joint_names_expr=[".*_calf_joint"],
                effort_limit_sim=300.0,
                velocity_limit_sim=9.425,
                stiffness=0.0,
                damping=0.0,
                armature=0.695525,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    sim.reset()

    if robot.num_joints != 12:
        raise RuntimeError(f"Expected 12 movable joints, received {robot.num_joints}: {robot.joint_names}")
    if set(robot.joint_names) != set(LEG_JOINTS):
        raise RuntimeError(f"Unexpected movable joints: {robot.joint_names}")
    missing_feet = sorted(set(FOOT_BODIES) - set(robot.body_names))
    if missing_feet:
        raise RuntimeError(f"Foot bodies missing after import: {missing_feet}; bodies={robot.body_names}")

    for _ in range(args_cli.steps):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())

    tensors = (robot.data.root_pos_w, robot.data.joint_pos, robot.data.joint_vel)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("Non-finite articulation state detected during physics stepping")

    print(f"[PASS] URDF: {urdf_path}")
    print(f"[PASS] movable joints ({robot.num_joints}): {robot.joint_names}")
    print(f"[PASS] rigid bodies ({robot.num_bodies}): {robot.body_names}")
    print(f"[PASS] completed {args_cli.steps} physics steps; root z={robot.data.root_pos_w[0, 2].item():.6f}")
    sim.stop()
    sim.clear()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
