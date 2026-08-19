# My_quadruped_robot_lab：Go2 / OpenRobot 运动任务（Isaac Lab）

本目录包含从 `My_unitree_go2_gym` 迁移出的 Go2 trot 训练任务，以及课题组自研 OpenRobot 轮子固定版的 trot 和 stairs 任务。旧 checkpoint 不参与迁移；模型从头训练。

## 当前实现

- 直接复用旧项目的 Go2 URDF 与 DAE 网格，保留 URDF 中的相对网格路径。
- 使用 Isaac Lab 2.2 的 manager-based 环境和 RSL-RL PPO。
- 12 维关节位置动作，控制频率 50 Hz，物理频率 200 Hz。
- policy 观测 470 维（47 维 x 10 帧），critic 观测 204 维（68 维 x 3 帧）。
- 迁移 trot 时钟、对角腿接触、抬脚高度、速度跟踪和姿态/力矩等 16 个奖励项。
- 启用摩擦、质量、质心、执行器增益、初始状态和外力推扰随机化。
- 执行器使用 1--3 个物理步的随机延迟，并在 PPO 中启用左右镜像损失。
- OpenRobot 资产采用低模凸碰撞体，膝部轮子通过固定关节保留为刚体，控制空间仅包含 12 个腿部关节。
- 提供 OpenRobot 轮固定版的三级楼梯课程，台阶高度固定为 0.15 m，踏步深度依次为 0.30、0.28、0.25 m。
- 提供 OpenRobot 轮固定版的 MuJoCo sim2sim 部署，支持预设速度和 viewer 键盘速度控制。

## 安装

```bash
conda activate env_isaaclab
python -m pip install -e My_quadruped_robot_lab/source/My_quadruped_robot_lab --no-deps
```

如果暂时不安装，也可以在每条命令前设置：

```bash
export PYTHONPATH="$PWD/My_quadruped_robot_lab/source/My_quadruped_robot_lab:$PYTHONPATH"
```

## 检查任务

```bash
python My_quadruped_robot_lab/scripts/list_envs.py

python My_quadruped_robot_lab/scripts/zero_agent.py \
  --task go2_trot \
  --num_envs 4 --num_steps 20 --headless

python My_quadruped_robot_lab/scripts/zero_agent.py \
  --task openrobot_wheelfixed_trot_play \
  --num_envs 4 --num_steps 300 --headless

python My_quadruped_robot_lab/scripts/zero_agent.py \
  --task openrobot_wheelfixed_stairs_play \
  --num_envs 1 --num_steps 300 --headless
```

调试 OpenRobot 默认关节姿态时，启动独立的可视化调姿工具：

```bash
python My_quadruped_robot_lab/scripts/tools/tune_openrobot_default_pose.py
```

工具直接复用训练资产配置。默认开启左右镜像联动和姿态锁定；关闭姿态锁定可在重力与训练 PD 参数下检查候选姿态。关闭“Respect original URDF joint limits”只用于预览限位外的几何姿态，不代表该姿态能直接用于训练。候选值可打印为 `openrobot.py` 配置片段，或保存到 `openrobot_pose_candidate.json`；工具本身不会修改 URDF 或训练配置。

## 从头训练

```bash
python My_quadruped_robot_lab/scripts/rsl_rl/train.py \
  --task go2_trot \
  --headless

python My_quadruped_robot_lab/scripts/rsl_rl/train.py \
  --task openrobot_wheelfixed_trot \
  --headless

python My_quadruped_robot_lab/scripts/rsl_rl/train.py \
  --task openrobot_wheelfixed_stairs \
  --headless
```

Go2 和 OpenRobot 默认均使用 4096 个环境，每环境每轮 24 步；trot 任务最多 15000 次迭代，OpenRobot stairs 任务最多 30000 次迭代。显存不足时添加 `--num_envs 1024` 或更小的值。训练输出写入：

```text
My_quadruped_robot_lab/logs/rsl_rl/go2_trot/<运行时间>/
My_quadruped_robot_lab/logs/rsl_rl/openrobot_wheelfixed_trot/<运行时间>/
My_quadruped_robot_lab/logs/rsl_rl/openrobot_wheelfixed_stairs/<运行时间>/
```

Hydra 的运行元数据统一写入 `My_quadruped_robot_lab/outputs/`。

## 回放新模型

```bash
python My_quadruped_robot_lab/scripts/rsl_rl/play.py \
  --task go2_trot_play \
  --checkpoint My_quadruped_robot_lab/logs/rsl_rl/go2_trot/<运行时间>/model_<迭代数>.pt

python My_quadruped_robot_lab/scripts/rsl_rl/play.py \
  --task openrobot_wheelfixed_trot_play \
  --checkpoint My_quadruped_robot_lab/logs/rsl_rl/openrobot_wheelfixed_trot/<运行时间>/model_<迭代数>.pt

python My_quadruped_robot_lab/scripts/rsl_rl/play.py \
  --task openrobot_wheelfixed_stairs_play \
  --checkpoint My_quadruped_robot_lab/logs/rsl_rl/openrobot_wheelfixed_stairs/<运行时间>/model_<迭代数>.pt
```

回放脚本同时导出 JIT 和 ONNX 策略。这里的 checkpoint 必须是本任务新训练出的模型；旧项目 checkpoint 的网络输入和运行时接口不兼容。

## MuJoCo Sim2Sim

部署器使用 IsaacLab 回放时导出的 TorchScript `policy.pt`，物理频率为 200 Hz、策略频率为 50 Hz。它复刻了训练环境的 470 维 policy 输入，包括按观测项排列的 10 帧历史、投影重力、逐关节动作缩放、默认关节偏置、延迟 PD、armature 和力矩限制。

安装项目时会自动安装官方 MuJoCo；已有 editable 安装可单独补装：

```bash
conda activate env_isaaclab
python -m pip install "mujoco>=3.2,<4"
```

仓库已经包含生成后的 OpenRobot MJCF。修改 URDF、碰撞网格或惯量后，重新生成并校验模型：

```bash
python My_quadruped_robot_lab/scripts/sim2sim/generate_openrobot_mjcf.py
```

### default：预设速度命令

`default` 模式在整个仿真期间使用命令行给定的速度。速度范围与训练配置一致：`vx` 为 ±0.8 m/s、`vy` 为 ±0.5 m/s、`yaw` 为 ±0.6 rad/s。

```bash
python My_quadruped_robot_lab/scripts/sim2sim/run_mujoco.py \
  --mode default --vx 0.8 --vy 0.0 --yaw 0.0
```

不打开 viewer 的快速检查或数据采集：

```bash
python My_quadruped_robot_lab/scripts/sim2sim/run_mujoco.py \
  --mode default --vx 0.0 --vy 0.0 --yaw 0.0 \
  --duration 5 --headless --no-real-time \
  --log My_quadruped_robot_lab/logs/mujoco/standing.csv
```

### keyboard：键盘速度控制

```bash
python My_quadruped_robot_lab/scripts/sim2sim/run_mujoco.py \
  --mode keyboard --vx 0.0 --vy 0.0 --yaw 0.0 --duration 0
```

MuJoCo viewer 窗口获得焦点后使用：

- `W` / `S`：增加 / 减小前向速度。
- `A` / `D`：增加 / 减小横向速度。
- `Q` / `E`：增加 / 减小偏航角速度。
- `Space`：立即把三个速度命令清零。

默认加载最新训练成功运行导出的 `policy.pt`。部署其他 checkpoint 时先用 IsaacLab `play.py` 导出，再显式传入：

```bash
python My_quadruped_robot_lab/scripts/sim2sim/run_mujoco.py \
  --mode default \
  --policy My_quadruped_robot_lab/logs/rsl_rl/openrobot_wheelfixed_trot/<运行时间>/exported/policy.pt
```

执行器延迟默认固定为 2 个物理步，可用 `--hip-thigh-delay` 和 `--calf-delay` 在 0--3 步间分别调整。训练时每次环境重置会在 1--3 步随机采样，而单机器人部署使用固定值以保证实验可重复。

当前已验证 MJCF 可加载、关节映射为 12 个策略自由度、JIT 输入输出为 `470 -> 12`，并可在 headless 模式完成闭环步进和 CSV 记录。由于 PhysX 与 MuJoCo 的网格接触、固定轮碰撞和求解器差异，当前策略在 MuJoCo 中长时间运动仍可能失稳；正式评价 sim2sim 成功率前，应继续标定足端/轮子碰撞、接触参数和执行器模型。该限制不影响两种命令模式及部署链路的使用。

## 代码入口与扩展方式

### Go2 stairs DreamWaQ

该任务使用独立的混合课程地形和项目内 DreamWaQ runner，不依赖 `openrobot_wheelfixed_stairs`。
默认训练张量为当前观测 45 维、5 帧历史 225 维、3 帧 critic 特权观测 783 维，
context VAE 输出 3 维速度估计和 16 维隐变量。

训练：

```bash
python My_quadruped_robot_lab/scripts/rsl_rl/train_dreamwaq.py \
  --task go2_stairs_dreamwaq --headless
```

小规模检查：

```bash
python My_quadruped_robot_lab/scripts/rsl_rl/train_dreamwaq.py \
  --task go2_stairs_dreamwaq --num_envs 32 --max_iterations 1 --headless
```

回放最新 checkpoint，并导出双输入 JIT 策略：

```bash
python My_quadruped_robot_lab/scripts/rsl_rl/play_dreamwaq.py \
  --task go2_stairs_dreamwaq_play
```

指定 checkpoint：

```bash
python My_quadruped_robot_lab/scripts/rsl_rl/play_dreamwaq.py \
  --task go2_stairs_dreamwaq_play \
  --checkpoint My_quadruped_robot_lab/logs/rsl_rl/go2_stairs_dreamwaq/<run>/model_<iteration>.pt
```

回放时策略自动导出到 checkpoint 同级的 `exported/policy_dreamwaq.pt`，其前向接口为
`policy(current_obs, observation_history)`，输入维度分别为 45 和 225。

### OpenRobot wheel-fixed stairs DreamWaQ

该任务与旧的 `openrobot_wheelfixed_stairs` 相互独立。四个 `foot` 是支撑和步态接触体，
固定膝部轮子只作为非期望碰撞体。训练命令固定从 `vx [-0.8, 0.8] m/s`、
`vy [-0.3, 0.3] m/s` 和 `wz [-0.5, 0.5] rad/s` 采样，不使用命令课程。
地形课程同时增加楼梯高度并减小踏步深度，最高等级覆盖 `0.25 x 0.15 m` 目标楼梯。
4096 环境下 OpenRobot 的刚体和三角网格接触量较大，任务将 PhysX GPU narrowphase
`collisionStackSize` 配置为 `2**28`（256 MiB），避免默认 64 MiB 缓冲区溢出后丢弃接触。

训练或断点续训使用通用 DreamWaQ 脚本：

```bash
python My_quadruped_robot_lab/scripts/rsl_rl/train_dreamwaq.py \
  --task openrobot_wheelfixed_stairs_dreamwaq --headless

python My_quadruped_robot_lab/scripts/rsl_rl/train_dreamwaq.py \
  --task openrobot_wheelfixed_stairs_dreamwaq \
  --resume --load_run -1 --load_checkpoint -1 --headless
```

必须使用 `train_dreamwaq.py`。普通 `scripts/rsl_rl/train.py` 使用标准 RSL-RL runner，
不会训练 DreamWaQ 的历史编码器和 VAE。正常日志应包含 `vae` 和 `vel` 损失。
如果 PhysX 输出 `collisionStackSize buffer overflow` 或 `Contacts have been dropped`，
说明接触求解已经不完整，不能继续使用该次训练结果；应检查是否加载了本任务的最新配置，
或降低 `--num_envs` 后重新启动训练。

在精确 `0.25 m` 深、`0.15 m` 高的楼梯上分别回放上楼和下楼策略：

```bash
python My_quadruped_robot_lab/scripts/rsl_rl/play_dreamwaq.py \
  --task openrobot_wheelfixed_stairs_dreamwaq_play_up

python My_quadruped_robot_lab/scripts/rsl_rl/play_dreamwaq.py \
  --task openrobot_wheelfixed_stairs_dreamwaq_play_down
```

两个回放任务默认加载 `openrobot_wheelfixed_stairs_dreamwaq` 的最新 checkpoint，
也可以通过 `--checkpoint <path>` 指定模型。回放时会同步导出双输入 JIT 策略。

- `assets/go2.py`：Go2 资产、初始姿态和执行器配置。
- `assets/openrobot.py`：OpenRobot 轮子固定版资产、调试后的默认站姿和 SETZ120/SETZ160 执行器配置。
- `trot/trot_env_cfg.py`：场景、观测、动作、命令、随机化、奖励和终止条件。
- `trot/openrobot_env_cfg.py`：OpenRobot 专用动作范围、机身/足端接触、随机化和高度参数。
- `trot/mdp/`：自定义 trot 观测、奖励与镜像变换。
- `trot/agents/rsl_rl_ppo_cfg.py`：PPO 与镜像损失配置。
- `stairs/terrain_cfg.py`：0.15 m 高、三级离散踏步深度的单向楼梯。
- `stairs/openrobot_env_cfg.py`：OpenRobot stairs 的观测、奖励、课程和训练/回放配置。
- `openrobot_wheelfixed_stairs_dreamwaq/`：独立 DreamWaQ 楼梯地形、环境、奖励和上下楼回放配置。
- `scripts/sim2sim/`：MuJoCo 模型生成、观测历史、显式 PD 和双模式运行入口。

增加其他机器狗时，新建独立的 `assets/<robot>.py`，并以共享 locomotion 配置为基类覆盖关节/足端名称、默认姿态、执行器和机器人特有参数。不要把 Go2 的名称或物理参数继续写入共享 MDP 函数。

## 已验证范围与资产注意事项

Go2 已在 `env_isaaclab` 中通过 4 环境 x 20 步环境测试，以及 32 环境、1 次 PPO 迭代测试。OpenRobot 轮子固定版的资产导入、物理步进、零动作站立和单次 PPO 迭代曾在旧默认姿态下通过；修正关节限位与默认站姿后已完成 URDF 一致性、配置编译和足底高度检查，正式训练前应在有可用 GPU 的 Isaac Sim 环境中重新运行本页的零动作与单次 PPO 测试。两类任务的镜像损失实现均已实际计算。

OpenRobot 电机模型以规格书输出端峰值作为瞬时硬限幅：SETZ120 为 603 N·m，SETZ160 为 900 N·m。规格书连续堵转力矩分别只有 117 N·m 和 300 N·m；当前力矩奖励会抑制高输出，但并不等价于热保护。实机部署前必须增加电流/温升模型或滑动窗口连续力矩约束，并结合调试后的默认站姿复核各关节静态负载。

旧 URDF 可以直接用于当前训练，但 Isaac Sim 会提示 `imu`、`radar` 固定子链缺少有效惯量，以及部分固定关节轴不与主轴对齐。导入器会自动近似/重定向，当前不阻断训练。正式进行 sim-to-real 标定前，应在 URDF 中补齐这些 link 的质量和惯量，并复核碰撞体、关节限位与官方 Go2 参数。

更完整的迁移设计与阶段规划见 [MIGRATION_PLAN.md](MIGRATION_PLAN.md)。
