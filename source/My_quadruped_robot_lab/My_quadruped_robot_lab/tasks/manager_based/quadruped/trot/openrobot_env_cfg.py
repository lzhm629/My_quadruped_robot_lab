"""Trot environment for the 12-DOF wheel-fixed OpenRobot."""

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from My_quadruped_robot_lab.assets import OPENROBOT_WHEELFIXED_CFG

from . import mdp
from .trot_env_cfg import (
    ActionsCfg,
    Go2TrotEnvCfg,
    Go2TrotSceneCfg,
    GO2_JOINTS,
)


@configclass
class OpenRobotTrotSceneCfg(Go2TrotSceneCfg):
    """Use the large OpenRobot articulation with the shared flat terrain and sensors."""

    robot = OPENROBOT_WHEELFIXED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class OpenRobotActionsCfg(ActionsCfg):
    """Position targets centered on the tuned OpenRobot nominal stance."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=GO2_JOINTS,
        preserve_order=True,
        scale={
            ".*_hip_joint": 0.02,
            ".*_thigh_joint": 0.10,
            ".*_calf_joint": 0.20,
        },
        use_default_offset=True,
        clip={
            ".*_hip_joint": (-0.2, 0.2),
            "^(FR|RR)_thigh_joint$": (-1.5, 0.0),
            "^(FR|RR)_calf_joint$": (-1.6, 0.6),
            "^(FL|RL)_thigh_joint$": (0.0, 1.5),
            "^(FL|RL)_calf_joint$": (-0.6, 1.6),
        },
    )


@configclass
class OpenRobotTrotEnvCfg(Go2TrotEnvCfg):
    """OpenRobot-specific physics, nominal height, contacts, and randomization."""

    scene: OpenRobotTrotSceneCfg = OpenRobotTrotSceneCfg(num_envs=4096, env_spacing=3.0)
    actions: OpenRobotActionsCfg = OpenRobotActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # A slower gait clock suits the 110 kg platform and its 0.45 m lower legs.
        self.observations.policy.gait_clock.params["cycle_time"] = 0.8
        self.observations.critic.gait_clock.params["cycle_time"] = 0.8

        self.rewards.trot.params["cycle_time"] = 0.8
        self.rewards.feet_clearance.params["cycle_time"] = 0.8
        self.rewards.feet_clearance.params["target_height"] = 0.08
        self.rewards.base_height.params["target_height"] = 0.69

        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_ang_vel_z.weight = 5.0
        self.rewards.trot.weight = 1.5

        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)

        # The root body is named base_link in the SolidWorks export.
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.events.add_base_mass.params["mass_distribution_params"] = (-5.0, 5.0)
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.events.reset_robot_joints.params["position_range"] = (-0.02, 0.02)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.1, 0.1),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="base_link"
        )
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[".*_thigh", ".*_calf", ".*_wheel"]
        )

        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.viewer.eye = (3.5, 2.5, 2.0)
        self.viewer.lookat = (-0.3, 0.0, 0.4)


@configclass
class OpenRobotTrotEnvCfg_PLAY(OpenRobotTrotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.scale_link_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.push_robot = None
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.commands.base_velocity.debug_vis = True

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
