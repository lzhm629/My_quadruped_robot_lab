"""Unitree Go2 asset migrated from My_unitree_go2_gym."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from My_quadruped_robot_lab import My_QUADRUPED_ROBOT_LAB_ROOT_DIR


GO2_URDF_PATH = My_QUADRUPED_ROBOT_LAB_ROOT_DIR / "assets" / "data" / "go2" / "urdf" / "go2.urdf"


GO2_LEGACY_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(GO2_URDF_PATH),
        activate_contact_sensors=True,
        fix_base=False,
        link_density=0.001,
        merge_fixed_joints=False,
        self_collision=True,
        replace_cylinders_with_capsules=True,
        make_instanceable=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.7,
            velocity_limit=30.1,
            stiffness=20.0,
            damping=0.5,
            armature=0.0,
            friction=0.0,
            min_delay=1,
            max_delay=3,
        )
    },
)
"""Legacy Go2 URDF with the trot task's PD gains and 5--15 ms action delay."""
