# My_quadruped_robot_lab：Go2 / OpenRobot 运动任务（Isaac Lab）

本目录包含从 `My_unitree_go2_gym` 迁移出的 Go2 trot 训练任务，以及课题组自研 OpenRobot 轮子固定版的 trot 任务和基于 DreamWaQ 的 stairs 任务。

## 当前实现

- 直接复用旧项目的 Go2 URDF 与 DAE 网格，保留 URDF 中的相对网格路径。
- 使用 Isaac Lab 2.2 的 manager-based 环境和 RSL-RL PPO。
- 12 维关节位置动作，控制频率 50 Hz，物理频率 200 Hz。
- policy 观测 470 维（47 维 x 10 帧），critic 观测 204 维（68 维 x 3 帧）。
- 迁移 trot 时钟、对角腿接触、抬脚高度、速度跟踪和姿态/力矩等 16 个奖励项。
- 启用摩擦、质量、质心、执行器增益、初始状态和外力推扰随机化。
- 执行器使用 1--3 个物理步的随机延迟，并在 PPO 中启用左右镜像损失。
- OpenRobot 资产采用低模凸碰撞体，膝部轮子通过固定关节保留为刚体，控制空间仅包含 12 个腿部关节。
- 提供 Go2 和 OpenRobot 轮固定版的 DreamWaQ 楼梯任务。

## 安装

```bash
conda activate env_isaaclab
python -m pip install -e My_quadruped_robot_lab/source/My_quadruped_robot_lab --no-deps
```

如果暂时不安装，也可以在每条命令前设置：

```bash
export PYTHONPATH="$PWD/My_quadruped_robot_lab/source/My_quadruped_robot_lab:$PYTHONPATH"
```

## 调试

检查任务

```bash
python My_quadruped_robot_lab/scripts/list_envs.py
```

调试 OpenRobot 默认关节姿态时，启动独立的可视化调姿工具：

```bash
python My_quadruped_robot_lab/scripts/tools/tune_openrobot_default_pose.py
```

工具直接复用训练资产配置。默认开启左右镜像联动和姿态锁定；关闭姿态锁定可在重力与训练 PD 参数下检查候选姿态。关闭“Respect original URDF joint limits”只用于预览限位外的几何姿态，不代表该姿态能直接用于训练。候选值可打印为 `openrobot.py` 配置片段，或保存到 `openrobot_pose_candidate.json`；工具本身不会修改 URDF 或训练配置。

## 任务

### trot

Go2 和 OpenRobot 默认均使用 4096 个环境，每环境每轮 24 步，最多训练 15000 次迭代。显存不足时可添加 `--num_envs 1024` 或更小的值。

#### 从头训练

```bash
# Go2
python My_quadruped_robot_lab/scripts/rsl_rl/train.py \
  --task go2_trot \
  --headless

# OpenRobot wheel-fixed
python My_quadruped_robot_lab/scripts/rsl_rl/train.py \
  --task openrobot_wheelfixed_trot \
  --headless
```

训练输出分别写入：

```text
My_quadruped_robot_lab/logs/rsl_rl/go2_trot/<运行时间>/
My_quadruped_robot_lab/logs/rsl_rl/openrobot_wheelfixed_trot/<运行时间>/
```

#### 断点重训

下面的命令默认从对应任务最新一次运行的最新 checkpoint 继续训练。也可以将 `--load_run` 和 `--checkpoint` 分别替换为指定的运行目录名和模型文件名。

```bash
# Go2
python My_quadruped_robot_lab/scripts/rsl_rl/train.py \
  --task go2_trot \
  --resume --load_run -1 --checkpoint -1 \
  --headless

# OpenRobot wheel-fixed
python My_quadruped_robot_lab/scripts/rsl_rl/train.py \
  --task openrobot_wheelfixed_trot \
  --resume --load_run -1 --checkpoint -1 \
  --headless
```

#### 回放模型

```bash
# Go2
python My_quadruped_robot_lab/scripts/rsl_rl/play.py \
  --task go2_trot_play \
  --checkpoint My_quadruped_robot_lab/logs/rsl_rl/go2_trot/<运行时间>/model_<迭代数>.pt

# OpenRobot wheel-fixed
python My_quadruped_robot_lab/scripts/rsl_rl/play.py \
  --task openrobot_wheelfixed_trot_play \
  --checkpoint My_quadruped_robot_lab/logs/rsl_rl/openrobot_wheelfixed_trot/<运行时间>/model_<迭代数>.pt
```

回放脚本同时导出 JIT 和 ONNX 策略。checkpoint 必须是本任务新训练出的模型；旧项目 checkpoint 的网络输入和运行时接口不兼容。

### stairs_dreamwaq

Go2 stairs 任务使用独立的混合课程地形和项目内 DreamWaQ runner。默认训练张量为当前观测 45 维、5 帧历史 225 维、3 帧 critic 特权观测 783 维，context VAE 输出 3 维速度估计和 16 维隐变量。

OpenRobot wheel-fixed stairs 任务以四个 `foot` 作为支撑和步态接触体，固定膝部轮子只作为非期望碰撞体。训练命令固定从 `vx [-0.8, 0.8] m/s`、`vy [-0.3, 0.3] m/s` 和 `wz [-0.5, 0.5] rad/s` 采样，不使用命令课程。地形课程同时增加楼梯高度并减小踏步深度，最高等级覆盖 `0.25 x 0.15 m` 目标楼梯。4096 环境下，任务将 PhysX GPU narrowphase `collisionStackSize` 配置为 `2**28`（256 MiB），避免默认 64 MiB 缓冲区溢出后丢弃接触。

#### 从头训练

```bash
# Go2
python My_quadruped_robot_lab/scripts/rsl_rl/train_dreamwaq.py \
  --task go2_stairs_dreamwaq \
  --headless

# OpenRobot wheel-fixed
python My_quadruped_robot_lab/scripts/rsl_rl/train_dreamwaq.py \
  --task openrobot_wheelfixed_stairs_dreamwaq \
  --headless
```

#### 断点重训

下面的命令默认从对应任务最新一次运行的最新 checkpoint 继续训练。

```bash
# Go2
python My_quadruped_robot_lab/scripts/rsl_rl/train_dreamwaq.py \
  --task go2_stairs_dreamwaq \
  --resume --load_run -1 --checkpoint -1 \
  --headless

# OpenRobot wheel-fixed
python My_quadruped_robot_lab/scripts/rsl_rl/train_dreamwaq.py \
  --task openrobot_wheelfixed_stairs_dreamwaq \
  --resume --load_run -1 --checkpoint -1 \
  --headless
```

#### 回放模型

Go2 回放任务默认加载最新 checkpoint；通过 `--checkpoint <path>` 可以指定模型。

```bash
python My_quadruped_robot_lab/scripts/rsl_rl/play_dreamwaq.py \
  --task go2_stairs_dreamwaq_play \
  --checkpoint My_quadruped_robot_lab/logs/rsl_rl/go2_stairs_dreamwaq/<运行时间>/model_<迭代数>.pt
```

OpenRobot 可在精确 `0.25 m` 深、`0.15 m` 高的楼梯上分别回放上楼和下楼策略。两个回放任务通过`--checkpoint <path>` 指定模型。

```bash
python My_quadruped_robot_lab/scripts/rsl_rl/play_dreamwaq.py \
  --task openrobot_wheelfixed_stairs_dreamwaq_play_up \
  --checkpoint My_quadruped_robot_lab/logs/rsl_rl/openrobot_wheelfixed_stairs_dreamwaq/<运行时间>/model_<迭代数>.pt

python My_quadruped_robot_lab/scripts/rsl_rl/play_dreamwaq.py \
  --task openrobot_wheelfixed_stairs_dreamwaq_play_down \
  --checkpoint My_quadruped_robot_lab/logs/rsl_rl/openrobot_wheelfixed_stairs_dreamwaq/<运行时间>/model_<迭代数>.pt
```

DreamWaQ 回放时会将双输入 JIT 策略导出到 checkpoint 同级的 `exported/policy_dreamwaq.pt`。其前向接口为 `policy(current_obs, observation_history)`，输入维度分别为 45 和 225。

所有任务的 Hydra 运行元数据统一写入 `My_quadruped_robot_lab/outputs/`。

## 代码入口与扩展方式

- `assets/go2.py`：Go2 资产、初始姿态和执行器配置。
- `assets/openrobot.py`：OpenRobot 轮子固定版资产、调试后的默认站姿和 SETZ120/SETZ160 执行器配置。
- `trot/trot_env_cfg.py`：场景、观测、动作、命令、随机化、奖励和终止条件。
- `trot/openrobot_env_cfg.py`：OpenRobot 专用动作范围、机身/足端接触、随机化和高度参数。
- `trot/mdp/`：自定义 trot 观测、奖励与镜像变换。
- `trot/agents/rsl_rl_ppo_cfg.py`：PPO 与镜像损失配置。
- `openrobot_wheelfixed_stairs_dreamwaq/`：独立 DreamWaQ 楼梯地形、环境、奖励和上下楼回放配置。

增加其他机器狗时，新建独立的 `assets/<robot>.py`，并以共享 locomotion 配置为基类覆盖关节/足端名称、默认姿态、执行器和机器人特有参数。不要把 Go2 的名称或物理参数继续写入共享 MDP 函数。

## 已验证范围与资产注意事项

Go2 已在 `env_isaaclab` 中通过 4 环境 x 20 步环境测试，以及 32 环境、1 次 PPO 迭代测试。OpenRobot 轮子固定版的资产导入、物理步进、零动作站立和单次 PPO 迭代曾在旧默认姿态下通过；修正关节限位与默认站姿后已完成 URDF 一致性、配置编译和足底高度检查，正式训练前应在有可用 GPU 的 Isaac Sim 环境中重新运行本页的零动作与单次 PPO 测试。两类任务的镜像损失实现均已实际计算。

OpenRobot 电机模型以规格书输出端峰值作为瞬时硬限幅：SETZ120 为 603 N·m，SETZ160 为 900 N·m。规格书连续堵转力矩分别只有 117 N·m 和 300 N·m；当前力矩奖励会抑制高输出，但并不等价于热保护。实机部署前必须增加电流/温升模型或滑动窗口连续力矩约束，并结合调试后的默认站姿复核各关节静态负载。

旧 URDF 可以直接用于当前训练，但 Isaac Sim 会提示 `imu`、`radar` 固定子链缺少有效惯量，以及部分固定关节轴不与主轴对齐。导入器会自动近似/重定向，当前不阻断训练。正式部署前，应在 URDF 中补齐这些 link 的质量和惯量，并复核碰撞体、关节限位与官方 Go2 参数。

更完整的迁移设计与阶段规划见 [MIGRATION_PLAN.md](MIGRATION_PLAN.md)。
