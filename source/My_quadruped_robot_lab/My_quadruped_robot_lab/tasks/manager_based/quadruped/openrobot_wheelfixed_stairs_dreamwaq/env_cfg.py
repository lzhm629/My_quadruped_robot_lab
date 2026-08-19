"""OpenRobot wheel-fixed, foot-supported stairs environment using DreamWaQ."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from My_quadruped_robot_lab.assets import OPENROBOT_WHEELFIXED_CFG
from My_quadruped_robot_lab.tasks.manager_based.quadruped.go2_stairs_dreamwaq.env_cfg import (
    EventCfg,
    ObservationsCfg,
)
from My_quadruped_robot_lab.tasks.manager_based.quadruped.trot.openrobot_env_cfg import (
    OpenRobotActionsCfg,
)
from My_quadruped_robot_lab.tasks.manager_based.quadruped.trot.trot_env_cfg import (
    GO2_FEET,
    GO2_JOINTS,
)

from . import mdp
from .terrain_cfg import (
    OPENROBOT_DREAMWAQ_PLAY_DOWN_TERRAINS_CFG,
    OPENROBOT_DREAMWAQ_PLAY_UP_TERRAINS_CFG,
    OPENROBOT_DREAMWAQ_TERRAINS_CFG,
)


ROBOT_CFG = SceneEntityCfg("robot", joint_names=GO2_JOINTS, preserve_order=True)
FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=GO2_FEET, preserve_order=True)
FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=GO2_FEET, preserve_order=True)
BODY_CONTACT_CFG = SceneEntityCfg(
    "contact_forces", body_names=[".*_thigh", ".*_calf", ".*_wheel"]
)
BASE_HEIGHT_SENSOR_CFG = SceneEntityCfg("base_height_scanner")
HIP_CFG = SceneEntityCfg("robot", joint_names=[".*_hip_joint"], preserve_order=True)


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=OPENROBOT_DREAMWAQ_TERRAINS_CFG,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    robot = OPENROBOT_WHEELFIXED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0), ordering="yx"),
        mesh_prim_paths=["/World/ground"],
    )
    base_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.3, 0.4)),
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.005,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 0.8),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
            heading=None,
        ),
    )


@configclass
class RewardsCfg:
    tracking_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    tracking_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.10)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.25)
    base_height = RewTerm(
        func=mdp.terrain_relative_base_height_l2,
        weight=-1.0,
        params={"target_height": 0.69, "sensor_cfg": BASE_HEIGHT_SENSOR_CFG},
    )
    normalized_torques = RewTerm(
        func=mdp.normalized_joint_torques_l2,
        weight=-0.10,
        params={"asset_cfg": ROBOT_CFG},
    )
    normalized_power = RewTerm(
        func=mdp.normalized_joint_power_l1,
        weight=-0.05,
        params={"asset_cfg": ROBOT_CFG},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7, params={"asset_cfg": ROBOT_CFG})
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-5, params={"asset_cfg": ROBOT_CFG})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    illegal_body_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": BODY_CONTACT_CFG, "threshold": 1.0},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={"command_name": "base_velocity", "sensor_cfg": FOOT_SENSOR_CFG, "threshold": 0.5},
    )
    landing_impact = RewTerm(
        func=mdp.foot_landing_impact,
        weight=-0.02,
        params={"sensor_cfg": FOOT_SENSOR_CFG, "threshold": 700.0},
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip,
        weight=-0.05,
        params={"sensor_cfg": FOOT_SENSOR_CFG, "asset_cfg": FOOT_BODY_CFG},
    )
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.5,
        params={"command_name": "base_velocity", "asset_cfg": ROBOT_CFG},
    )
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": ROBOT_CFG})
    hip_pos = RewTerm(func=mdp.hip_position_l2, weight=-0.05, params={"asset_cfg": HIP_CFG})
    stumble = RewTerm(func=mdp.stumble, weight=-0.1, params={"sensor_cfg": FOOT_SENSOR_CFG})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 1.0},
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class OpenRobotWheelFixedStairsDreamWaQEnvCfg(ManagerBasedRLEnvCfg):
    seed = 1
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: OpenRobotActionsCfg = OpenRobotActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # The default 64 MiB narrowphase stack overflows with 4096 OpenRobot
        # instances on the mixed trimesh terrain and silently drops contacts.
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.base_height_scanner.update_period = self.decimation * self.sim.dt

        # Start with restrained randomization for the large platform.
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.2)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.1)
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.events.add_base_mass.params["mass_distribution_params"] = (-5.0, 5.0)
        self.events.scale_link_mass.params["mass_distribution_params"] = (0.95, 1.05)
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        self.events.base_com.params["com_range"] = {
            "x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)
        }
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.95, 1.05)
        self.events.actuator_gains.params["damping_distribution_params"] = (0.95, 1.05)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.2, 0.2), "y": (-0.2, 0.2),
            "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.3, 0.3),
        }
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-0.1, 0.1)
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
        }
        self.events.reset_robot_joints.params["position_range"] = (-0.02, 0.02)
        self.viewer.eye = (5.0, 3.5, 2.8)
        self.viewer.lookat = (2.5, 0.0, 0.7)


class _PlayMixin:
    play_terrain = None
    play_speed = 0.25

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = self.play_terrain
        self.scene.terrain.max_init_terrain_level = 0
        self.observations.policy.enable_corruption = False
        self.observations.history.enable_corruption = False
        for name in (
            "physics_material", "add_base_mass", "scale_link_mass", "base_com",
            "actuator_gains", "push_robot",
        ):
            setattr(self.events, name, None)
        self.curriculum.terrain_levels = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
        }
        self.events.reset_base.params["velocity_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_x = (self.play_speed, self.play_speed)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0


@configclass
class OpenRobotWheelFixedStairsDreamWaQEnvCfg_PLAY_UP(
    _PlayMixin, OpenRobotWheelFixedStairsDreamWaQEnvCfg
):
    play_terrain = OPENROBOT_DREAMWAQ_PLAY_UP_TERRAINS_CFG
    play_speed = 0.50


@configclass
class OpenRobotWheelFixedStairsDreamWaQEnvCfg_PLAY_DOWN(
    _PlayMixin, OpenRobotWheelFixedStairsDreamWaQEnvCfg
):
    play_terrain = OPENROBOT_DREAMWAQ_PLAY_DOWN_TERRAINS_CFG
    play_speed = 0.45
