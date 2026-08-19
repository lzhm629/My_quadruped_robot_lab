"""Isaac Lab manager-based environment for Go2 stairs DreamWaQ."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from My_quadruped_robot_lab.assets import GO2_DREAMWAQ_CFG
from My_quadruped_robot_lab.tasks.manager_based.quadruped.trot.trot_env_cfg import GO2_FEET, GO2_JOINTS

from . import mdp
from .terrain_cfg import GO2_DREAMWAQ_PLAY_TERRAINS_CFG, GO2_DREAMWAQ_TERRAINS_CFG


ROBOT_CFG = SceneEntityCfg("robot", joint_names=GO2_JOINTS, preserve_order=True)
FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=GO2_FEET, preserve_order=True)
BODY_CONTACT_CFG = SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf", "base"])
HEIGHT_SENSOR_CFG = SceneEntityCfg("height_scanner")
BASE_HEIGHT_SENSOR_CFG = SceneEntityCfg("base_height_scanner")
HIP_CFG = SceneEntityCfg("robot", joint_names=[".*_hip_joint"], preserve_order=True)


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GO2_DREAMWAQ_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        ),
    )
    robot = GO2_DREAMWAQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0), ordering="yx"),
        mesh_prim_paths=["/World/ground"],
    )
    base_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.3, 0.4)),
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, update_period=0.005
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9))
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0), rel_standing_envs=0.0,
        rel_heading_envs=0.0, heading_command=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=None
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=GO2_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True
    )


class _ObsTerms:
    @staticmethod
    def current(noisy: bool):
        noise = Unoise(n_min=-0.2, n_max=0.2) if noisy else None
        return [
            ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=(2.0, 2.0, 0.25)),
            ObsTerm(func=mdp.base_ang_vel, noise=noise, scale=0.25),
            ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05) if noisy else None),
            ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ROBOT_CFG}, noise=Unoise(n_min=-0.02, n_max=0.02) if noisy else None),
            ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ROBOT_CFG}, noise=Unoise(n_min=-1.5, n_max=1.5) if noisy else None, scale=0.05),
            ObsTerm(func=mdp.last_action),
        ]


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=(2.0, 2.0, 0.25))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ROBOT_CFG}, noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ROBOT_CFG}, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    @configclass
    class HistoryCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=(2.0, 2.0, 0.25))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ROBOT_CFG}, noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ROBOT_CFG}, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        terrain = ObsTerm(func=mdp.terrain_height_profile, params={"sensor_cfg": HEIGHT_SENSOR_CFG})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        friction = ObsTerm(func=mdp.material_friction, params={"asset_cfg": ROBOT_CFG})
        restitution = ObsTerm(func=mdp.material_restitution, params={"asset_cfg": ROBOT_CFG})
        stiffness = ObsTerm(func=mdp.stiffness_multipliers, params={"asset_cfg": ROBOT_CFG})
        damping = ObsTerm(func=mdp.damping_multipliers, params={"asset_cfg": ROBOT_CFG})
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=(2.0, 2.0, 0.25))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ROBOT_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ROBOT_CFG}, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 3
            self.flatten_history_dim = True

    @configclass
    class VelocityCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    @configclass
    class ReconstructionCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, scale=(2.0, 2.0, 0.25))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ROBOT_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ROBOT_CFG}, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    history: HistoryCfg = HistoryCfg()
    critic: CriticCfg = CriticCfg()
    velocity: VelocityCfg = VelocityCfg()
    reconstruction: ReconstructionCfg = ReconstructionCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.2, 1.25), "dynamic_friction_range": (0.2, 1.25), "restitution_range": (0.0, 0.5), "num_buckets": 64},
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_distribution_params": (-1.0, 1.0), "operation": "add"},
    )
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_(hip|thigh|calf|foot)"), "mass_distribution_params": (0.9, 1.1), "operation": "scale"},
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)}},
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains, mode="startup",
        params={"asset_cfg": ROBOT_CFG, "stiffness_distribution_params": (0.9, 1.1), "damping_distribution_params": (0.9, 1.1), "operation": "scale"},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(4.0, 4.0),
        params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "roll": (-0.6, 0.6), "pitch": (-0.6, 0.6), "yaw": (-0.6, 0.6)}},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-0.1, 0.1)}, "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)}},
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset", params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)}
    )


@configclass
class RewardsCfg:
    tracking_lin_vel = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    tracking_ang_vel = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    base_height = RewTerm(func=mdp.terrain_relative_base_height_l2, weight=-8.0, params={"target_height": 0.40, "sensor_cfg": BASE_HEIGHT_SENSOR_CFG})
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4, params={"asset_cfg": ROBOT_CFG})
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": ROBOT_CFG})
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-5, params={"asset_cfg": ROBOT_CFG})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    collision = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": BODY_CONTACT_CFG, "threshold": 0.1})
    feet_air_time = RewTerm(func=mdp.feet_air_time, weight=1.0, params={"command_name": "base_velocity", "sensor_cfg": FOOT_SENSOR_CFG, "threshold": 1.0})
    stand_still = RewTerm(func=mdp.stand_still_joint_deviation_l1, weight=-0.5, params={"command_name": "base_velocity", "asset_cfg": ROBOT_CFG})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": ROBOT_CFG})
    joint_power = RewTerm(func=mdp.joint_power_l1, weight=-2.0e-5, params={"asset_cfg": ROBOT_CFG})
    power_distribution = RewTerm(func=mdp.power_distribution, weight=-1.0e-5, params={"asset_cfg": ROBOT_CFG})
    hip_pos = RewTerm(func=mdp.hip_position_l2, weight=-0.1, params={"asset_cfg": HIP_CFG})
    stumble = RewTerm(func=mdp.stumble, weight=-0.1, params={"sensor_cfg": FOOT_SENSOR_CFG})
    x_command_hip_regular = RewTerm(func=mdp.zero_command_hip_symmetry, weight=-0.5, params={"command_name": "base_velocity", "asset_cfg": HIP_CFG})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0})


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class Go2StairsDreamWaQEnvCfg(ManagerBasedRLEnvCfg):
    seed = 1
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
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
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.base_height_scanner.update_period = self.decimation * self.sim.dt
        self.viewer.eye = (6.0, 4.0, 3.0)
        self.viewer.lookat = (3.0, 0.0, 0.8)


@configclass
class Go2StairsDreamWaQEnvCfg_PLAY(Go2StairsDreamWaQEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator = GO2_DREAMWAQ_PLAY_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 4
        self.observations.policy.enable_corruption = False
        self.observations.history.enable_corruption = False
        for name in ("physics_material", "add_base_mass", "scale_link_mass", "base_com", "actuator_gains", "push_robot"):
            setattr(self.events, name, None)
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
