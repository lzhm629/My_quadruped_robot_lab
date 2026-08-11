"""OpenRobot wheel-fixed asset configuration for quadruped locomotion."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from My_quadruped_robot_lab import My_QUADRUPED_ROBOT_LAB_ROOT_DIR


OPENROBOT_WHEELFIXED_URDF_PATH = (
    My_QUADRUPED_ROBOT_LAB_ROOT_DIR
    / "assets"
    / "data"
    / "openrobot_wheelfixed"
    / "urdf"
    / "openrobot_wheelfixed.urdf"
)


OPENROBOT_WHEELFIXED_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(OPENROBOT_WHEELFIXED_URDF_PATH),
        activate_contact_sensors=True,
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=False,
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
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Nominal mirrored stance measured with the default-pose tuning tool.
        # The base height leaves the foot collision meshes just above the ground.
        pos=(0.0, 0.0, 0.583),
        joint_pos={
            ".*_hip_joint": 0.0,
            "^(FR|RR)_thigh_joint$": -0.3,
            "^(FL|RL)_thigh_joint$": 0.3,
            "^(FR|RR)_calf_joint$": -0.1,
            "^(FL|RL)_calf_joint$": 0.1,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "setz120": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            # Datasheet output-side peak torque.  Continuous thermal loading
            # (117 Nm) is encouraged through the task's torque penalty; a
            # thermal/current-window constraint is still required on hardware.
            effort_limit=603.0,
            velocity_limit=12.8,
            stiffness=1500.0,
            damping=40.0,
            armature=0.133886,
            friction=0.0,
            min_delay=1,
            max_delay=3,
        ),
        "setz160": DelayedPDActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            # Datasheet output-side peak torque (continuous: 300 Nm).
            effort_limit=900.0,
            velocity_limit=9.425,
            stiffness=1500.0,
            damping=60.0,
            armature=0.695525,
            friction=0.0,
            min_delay=1,
            max_delay=3,
        ),
    },
)
