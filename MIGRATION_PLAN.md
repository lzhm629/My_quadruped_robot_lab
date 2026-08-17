# My_unitree_go2_gym → Isaac Lab 迁移方案

> 目标目录：`My_quadruped_robot_lab`  
> 目标环境：Conda `env_isaaclab`  
> 调研日期：2026-08-07

## 1. 结论摘要

建议将 `My_quadruped_robot_lab` 建成一个独立于 Isaac Lab 主仓库的外部 extension，并采用 **Manager-Based RL** 作为主架构。迁移对象不是旧项目约 7 份高度重复的千行环境类，而是其中真正有价值的任务语义：观测、动作、奖励、终止、课程、随机化、延迟、特权信息和翻转/跳跃状态机。

推荐的总体设计是：

```text
机器人资产配置（Go2，后续可加 Go1/A1/B2/自研四足）
                   ↓
四足公共 MDP 组件（观测、动作、奖励、事件、终止、课程）
                   ↓
任务族配置（locomotion / jump / stand / flip）
                   ↓
机器人 × 任务薄配置 + Gymnasium 注册 + RSL-RL 配置
```

迁移优先级建议为：

1. `go2_trot`：建立最小闭环和基线。
2. `go2_stairs`：验证地形、射线高度图和课程。
3. `go2_jump`：验证周期相位、自定义奖励、历史观测和对称损失。
4. `go2_handstand`、`go2_leggedstand`：抽象为同一 `two_leg_stand` 任务族，通过支撑腿配置区分。
5. `go2_spring_jump`：迁移单次技能状态机和辅助脉冲课程。
6. `go2_backflip`：最后迁移；旧仓库无对应 checkpoint，且任务本身依赖辅助脉冲与阶段判定，风险最高。

不建议直接把旧环境类逐行翻译成 `DirectRLEnv`。这样虽然短期看似省事，但会继续保留 80%～99% 的任务间重复代码，无法满足后续接入更多机器狗的目标。对极少数有状态逻辑的技能，可在 Manager-Based 框架内用有状态的 command/reward/event term 实现；只有当 Manager 生命周期确实无法表达时，再为该任务单独使用 `DirectRLEnv`。

## 2. 已确认的环境基线

本机 `env_isaaclab` 已检测到：

| 组件 | 版本/位置 |
|---|---|
| Python | 3.11.15 |
| Isaac Lab 源码 | `/home/liming/Workspace/env/IsaacLab`，Git `v2.2.0` |
| Isaac Sim | 5.0.0.0 |
| `isaaclab` | 0.44.9 |
| `isaaclab_tasks` | 0.10.45 |
| `isaaclab_assets` | 0.2.2 |
| `isaaclab_rl` | 0.2.3 |
| `rsl-rl-lib` | 2.3.3 |
| PyTorch | 2.7.0+cu128 |
| Gymnasium | 1.2.0 |

Isaac Lab 现有源码中已经具备：

- 官方 `UNITREE_GO2_CFG` 和 Go2 USD；
- `Isaac-Velocity-Flat-Unitree-Go2-v0`、`Isaac-Velocity-Rough-Unitree-Go2-v0` 示例；
- `ObservationGroupCfg.history_length`，可替代旧项目手写 `deque` 堆帧；
- `DelayedPDActuatorCfg`，可表达物理步级的动作延迟；
- policy/critic 两个 observation group，可表达非对称 Actor-Critic；
- RSL-RL 2.3.3 原生 symmetry augmentation / mirror loss；
- 标准 train、play、JIT/ONNX 导出及 checkpoint 恢复流程。

因此目标项目应依赖当前环境中的官方包，不应复制 `isaaclab`、`isaaclab_tasks` 或旧版 `rsl_rl` 到 `My_quadruped_robot_lab`。

## 3. 旧项目框架梳理

### 3.1 执行链路

旧项目是典型的 `legged_gym` 派生结构：

```text
train.py / play.py
  → envs/__init__.py 注册任务
  → TaskRegistry 创建环境和 OnPolicyRunner
  → 每个任务自己的 BaseTask 子类
  → Isaac Gym tensor API + PhysX
  → 仓库内置 rsl_rl 1.0.2 分支
```

`envs/__init__.py` 注册了 7 个任务：`go2_trot`、`go2_stairs`、`go2_jump`、`go2_handstand`、`go2_leggedstand`、`go2_spring_jump`、`go2_backflip`。

每个环境大体重复以下生命周期：动作裁剪 → 4 个物理子步 → PD 力矩 → 刷新张量 → 计算派生状态 → 终止/奖励 → reset → 观测 → 更新历史缓存。公共仿真步长为 `dt=0.005 s`，控制 decimation 为 4，所以策略频率是 **50 Hz**，物理频率是 **200 Hz**。

### 3.2 代码特征

任务环境文件合计超过 7,000 行，而且复制程度很高：

- handstand 与 leggedstand 环境文件文本相似度约 99.7%；真正差异主要是前腿/后腿角色与少量奖励参数；
- trot、stairs、jump 之间约 82%～84% 相似；
- backflip 与 spring_jump 约 89% 相似。

这表明应把共性拆成 MDP term，把差异保留在配置层，不能继续按“一个任务复制一份完整环境类”的方式组织。

### 3.3 资产和控制

旧项目使用 `resources/robots/go2/urdf/go2.urdf`，12 个关节，动作是相对默认关节角的 position target：

```text
q_target = q_default + 0.25 × delayed_action + motor_zero_offset
τ = Kp × (q_target - q) - Kd × q_dot
```

移动、跳跃和翻转任务通常使用 `Kp=20, Kd=0.5`；双腿站立任务使用 `Kp=40, Kd=1.0`。旧 URDF 的关节顺序、刚体名称、惯量、碰撞体和力矩限制共同决定旧策略行为，不能仅凭“都是 Go2”就假定与官方 USD 完全等价。

Isaac Lab 官方 Go2 资产默认是 `Kp=25, Kd=0.5`、effort/saturation limit 23.5、velocity limit 30.0。迁移时必须显式覆盖这些参数，并进行资产对账，而不是直接接受官方默认值。

### 3.4 各任务的关键差异

| 任务 | Actor / Critic 维度 | 历史 | 地形 | 主要任务语义 | 旧 PPO 学习率 / 迭代 |
|---|---:|---:|---|---|---:|
| trot | 470 / 204 | 10 / 3 | 平面 | 速度跟踪、对角步态相位、足端抬高、静止约束 | 1e-5 / 15k |
| stairs | 45 / 235 | 1 / 1 | trimesh | 187 点高度观测、地形课程、速度跟踪 | 1e-5 / 30k |
| jump | 470 / 210 | 10 / 3 | 平面 | 周期跳跃相位、腾空/足高/落地、速度跟踪 | 1e-4 / 15k |
| handstand | 48 / 89 | 1 / 1 | 平面 | 后足支撑、前足腾空与高度、姿态和存活 | 1e-3 / 15k |
| leggedstand | 48 / 89 | 1 / 1 | 平面 | 前足支撑、后足腾空与高度、姿态和存活 | 1e-3 / 15k |
| spring_jump | 470 / 195 | 10 / 3 | 平面 | 单次起跳、腾空、落点、落地稳定、辅助上推 | 1e-5 / 50k |
| backflip | 470 / 150 | 10 / 3 | 平面 | 单次起跳、绕 Y 轴旋转、落点/着陆、上推与旋转辅助 | 1e-5 / 50k |

说明：历史栏分别为 policy/critic 堆帧数。维度已与仓库中的 checkpoint 首层权重核对；旧仓库没有 backflip checkpoint。

### 3.5 观测语义

旧项目不是统一观测空间，必须逐任务保持顺序并写成 schema：

- trot/jump：单帧 policy 为 47 维，基本组成是 `[sin_phase, cos_phase, command(3), base_ang_vel(3), base_euler(3), q_rel(12), qd(12), last_action(12)]`，再堆 10 帧；
- spring_jump/backflip：也是 47 维，但前两个量是零占位，command 未必与 trot 使用相同缩放；
- stairs：45 维 policy，没有相位，critic 额外加入 base linear velocity 和 187 点高度；
- handstand/leggedstand：48 维 policy，包含两个零占位、`stand_command`、角速度、projected gravity、速度命令、关节状态和动作；
- critic 额外使用接触状态、机身线速度、摩擦/质量/质心/PD/关节物性等仿真特权量，具体集合随任务不同。

这里存在一个迁移陷阱：Euler XYZ 与 projected gravity 不是等价观测。若目标是加载旧策略，必须保留旧字段、顺序、缩放和历史展平顺序；若目标只是重训，则可切换到更稳健的 projected gravity，但要建立新的 policy schema 版本。

### 3.6 随机化与延迟

旧项目包含以下 Sim-to-Real 随机化：

- 摩擦、恢复系数；
- base mass、各 link mass、base COM；
- PD 增益、输出力矩倍率、电机零位；
- joint friction、damping、armature（站立任务使用较多）；
- 随机推扰；
- motor/IMU observation latency 和 command/action latency。

延迟缓冲在每个物理子步更新，因此旧配置的 `[1, 3]` 对应大约 **5～15 ms**，不是 1～3 个 50 Hz policy step。迁移时必须明确延迟的时间基准。动作侧优先使用 `DelayedPDActuatorCfg(min_delay=1, max_delay=3)`；观测侧需要自定义一个在物理步更新的 delayed sensor/buffer，再由 observation term 读取，不能简单把按策略步更新的 observation history 当作延迟替代。

### 3.7 PPO 的本地修改

旧 `rsl_rl` 相比早期上游版本最重要的修改是镜像对称损失：通过带符号 permutation matrix 对 policy observation 和 action 做左右镜像，再约束动作均值一致。trot、jump、spring_jump 启用了该功能。

目标环境的 `rsl-rl-lib 2.3.3` 已原生支持 data augmentation 和 mirror loss，所以应实现一个按 observation term 名称工作的 `quadruped_mirror()`，并配置新版 symmetry API。不要移植旧 `ppo.py`；旧实现硬编码 `.cuda()`，并依赖难维护的浮点索引技巧（如 `-0.0001` 表示带符号的第 0 项）。

## 4. 目标工程设计

### 4.1 推荐目录

```text
My_quadruped_robot_lab/
├── README.md
├── pyproject.toml
├── setup.py
├── source/My_quadruped_robot_lab/
│   ├── config/extension.toml
│   └── My_quadruped_robot_lab/
│       ├── __init__.py
│       ├── assets/
│       │   ├── robots/
│       │   │   ├── go2.py
│       │   │   └── registry.py
│       │   └── data/go2/              # 仅放确需自维护的 URDF/USD/mesh
│       └── tasks/manager_based/quadruped/
│           ├── mdp/
│           │   ├── observations.py
│           │   ├── rewards.py
│           │   ├── commands.py
│           │   ├── events.py
│           │   ├── terminations.py
│           │   ├── curriculums.py
│           │   ├── modifiers.py
│           │   └── symmetry.py
│           ├── common/
│           │   ├── scene_cfg.py
│           │   ├── observations_cfg.py
│           │   ├── events_cfg.py
│           │   └── ppo_cfg.py
│           ├── locomotion/
│           │   ├── velocity_env_cfg.py
│           │   └── config/go2/{trot,stairs}_env_cfg.py
│           ├── jump/
│           │   ├── jump_env_cfg.py
│           │   └── config/go2/{jump,spring_jump}_env_cfg.py
│           ├── stand/
│           │   ├── two_leg_stand_env_cfg.py
│           │   └── config/go2/{handstand,legstand}_env_cfg.py
│           └── flip/
│               ├── flip_env_cfg.py
│               └── config/go2/backflip_env_cfg.py
├── scripts/
│   ├── list_envs.py
│   ├── zero_agent.py
│   ├── random_agent.py
│   ├── rsl_rl/{train,play}.py
│   └── tools/{inspect_asset,compare_rollout,convert_checkpoint}.py
├── tests/
│   ├── test_config.py
│   ├── test_observation_schema.py
│   ├── test_rewards.py
│   ├── test_symmetry.py
│   └── test_checkpoint_schema.py
└── docs/
    ├── observation_schemas.md
    ├── asset_parity.md
    └── migration_checklist.md
```

该结构遵循本机 Isaac Lab 的外部 extension 模板，项目可用 `pip install -e source/My_quadruped_robot_lab` 安装，不修改 `/home/liming/Workspace/env/IsaacLab`。

### 4.2 多机器人扩展边界

需要严格分开三类配置：

1. **RobotSpec（机器人事实）**：资产路径、关节/刚体正则、默认姿态、执行器、力矩/速度限制、足端集合、机身名称、左右镜像映射。
2. **TaskSpec（任务意图）**：命令范围、观测项、奖励项、终止、课程、episode 长度，不出现 `FL_foot` 等 Go2 硬编码名称。
3. **RobotTaskCfg（绑定层）**：把某个 RobotSpec 注入某个 TaskSpec，只覆盖必要的机器人特定参数。

建议接口示意：

```python
@configclass
class QuadrupedRobotSpec:
    articulation_cfg: ArticulationCfg
    base_body: str
    foot_bodies: tuple[str, ...]
    thigh_bodies: tuple[str, ...]
    joint_order: tuple[str, ...]
    mirror_joint_map: tuple[int, ...]

class UnitreeGo2TrotEnvCfg(TrotEnvCfg):
    robot_spec = GO2_SPEC
```

奖励和观测 term 通过 `SceneEntityCfg` 或 RobotSpec 解析出的名称工作。新增机器狗时原则上只新增资产配置和一组薄绑定配置，不复制 reward/observation 实现。

### 4.3 任务注册命名

建议统一使用简短的 Gymnasium ID：

```text
go2_trot
go2_stairs
go2_jump
go2_handstand
go2_leggedstand
go2_spring_jump
go2_backflip
```

每个训练任务同时注册 `_play` 变体：减少环境数量、关闭 corruption/随机推扰/课程并固定命令。不要用运行时 `cfg.env.test` 分支改变同一个环境定义。

## 5. Isaac Gym → Isaac Lab 映射

| 旧实现 | Isaac Lab 目标实现 | 迁移注意点 |
|---|---|---|
| `BaseTask` + 手写 step | `ManagerBasedRLEnv` | 让 managers 接管顺序和 reset 生命周期 |
| `create_sim/_create_envs` | `InteractiveSceneCfg` | 不在任务中手写 actor/environment 创建 |
| URDF asset options | `ArticulationCfg` + spawn cfg | 逐项对账质量、惯量、碰撞、关节限位 |
| 手写 PD torque | `JointPositionActionCfg` + actuator cfg | 保持 offset、scale、Kp/Kd、effort clipping |
| `compute_observations` | `ObservationTermCfg` / groups | policy 与 critic 分组；顺序写入 schema 测试 |
| `deque` 堆帧 | observation `history_length` | 核对 reset 时的 padding 行为和展平顺序 |
| 特权观测 tensor | `critic` observation group | RSL wrapper 会自动识别 |
| `_reward_*` + scales | `RewardTermCfg` | 权重仍按每秒乘 `step_dt`；先核对量纲 |
| `check_termination` | `TerminationTermCfg` | timeout 必须标记 `time_out=True` |
| `_process_*_props` | startup/reset `EventTermCfg` | 不同物性应按正确 mode 随机化 |
| `_push_robots` | interval/reset event | “训练辅助脉冲”和“鲁棒性扰动”必须区分命名 |
| `Terrain` | `TerrainImporterCfg` + generator | stairs 单独配置比例、行列和难度课程 |
| task registry | `gym.register()` | env cfg 与 RSL cfg 作为 entry point |
| 旧 OnPolicyRunner | 官方 RSL-RL runner | 不复制旧算法库 |
| 手写 symmetry matrix | RSL-RL symmetry callback | 按 term 和 RobotSpec 镜像，不硬编码扁平索引 |
| play 导出 JIT | Isaac Lab play 导出 JIT/ONNX | 同时导出 observation schema 和 normalization 元数据 |

## 6. 分阶段实施方案

### 阶段 0：冻结基线和建立可比较数据

交付物：

- 记录 `env_isaaclab` 的完整包锁定、GPU/driver 信息和 Isaac Lab Git commit；
- 为 7 个旧任务生成机器可读 manifest：观测字段与切片、动作关节顺序、缩放、奖励公式/权重、终止、随机化范围；
- 保存旧环境固定 seed、关闭噪声后的短 rollout：初始 root/joint state、policy observations、torques、rewards、done；
- 对现有 8 个 checkpoint 建立输入输出维度清单和不可变备份；
- 生成旧 URDF 与官方 Go2 USD 的资产差异报告。

门禁：任何观测维度不能只靠配置注释判断，必须由运行时 shape assertion 验证。

### 阶段 1：工程骨架和 Go2 资产

使用本机 Isaac Lab extension template 建立 `My_quadruped_robot_lab`，先注册一个最小 Go2 平地环境。

资产策略分两步：

1. **迁移基准资产**：从旧 URDF 导入/转换 USD，尽可能保持旧碰撞和惯性，用于行为对齐；
2. **生产资产候选**：评估官方 `UNITREE_GO2_CFG`。只有资产 parity 测试通过，或接受重新训练结果后，才切换为官方 USD。

门禁：1 个环境可启动；zero/random agent 可运行 1,000 步；无 NaN；关节、足端、base 名称和顺序断言全部通过。

### 阶段 2：迁移 trot，建立公共 MDP

先实现：

- 50 Hz policy / 200 Hz physics；
- 相对关节位置动作、动作缩放和执行器限制；
- 47×10 policy、68×3 critic schema；
- gait phase、trot/contact/feet-clearance 和通用正则奖励；
- friction/mass/COM/PD/zero-offset/random push；
- observation/action latency；
- 新版 quadruped mirror callback；
- RSL-RL cfg 和 train/play/export。

门禁：观测逐字段 shape、dtype、device、顺序通过；镜像两次等于原值；零动作站立不爆炸；短训 reward 有上升趋势。

### 阶段 3：迁移 stairs

实现与旧地形比例对应的 terrain generator、187 点 yaw-aligned ray scan、terrain level curriculum 和 play 固定地形配置。

注意：Isaac Lab 默认 height scanner 是 17×11=187 点，与旧项目采样网格天然对应，但仍需确认点排列顺序、射线高度定义、相对 base offset 和 clipping。维度相同不代表数值语义相同。

门禁：可视化射线落点；平地扫描接近常数；台阶高度和旧实现抽样误差满足约定阈值；terrain curriculum 能升降级。

### 阶段 4：迁移 jump 和双腿站立

`jump` 复用 locomotion 公共项，只增加 phase/stance mask、腾空、足高和跳跃奖励。`handstand` 与 `leggedstand` 合并成 `TwoLegStandEnvCfg`，参数化：

- `support_feet`；
- `air_feet`；
- 目标 base 高度；
- 目标姿态轴；
- 各奖励权重。

门禁：前/后腿互换不改环境代码；每个 task 的 policy/critic 维度与 manifest 一致；接触筛选指向正确足端。

### 阶段 5：迁移 spring jump 与 backflip

把旧环境里的隐式布尔量显式建模为 per-env skill state：

```text
WAIT → TRIGGERED → TAKEOFF → FLIGHT → LANDED → DONE
```

状态至少包含 trigger frame、was_in_flight、has_landed、landing pose、max height、max pitch rate、辅助脉冲标记。使用有状态 command term 管理阶段，reward/observation/event term 只读取统一状态，避免各自重复推断接触阶段。

旧代码中“朝目标推机器人”的上推/旋转速度注入属于训练课程，不是 domain randomization。目标实现应命名为 `assisted_takeoff_curriculum`，可记录辅助比例并逐步退火到 0；否则即使 reward 上升，也不能证明机器人学会了自主技能。

门禁：状态机转换单元测试；每个环境独立 reset；辅助概率退火可观测；无辅助评估单独出指标；backflip 完整旋转用累计/unwrap 后角度或姿态误差判断，不能只用 Euler 瞬时值。

### 阶段 6：策略、回归与部署接口

- 新训练 checkpoint 使用官方 RSL-RL 2.3.3 格式；
- play 同时导出 JIT 和 ONNX；
- 导出包包含关节顺序、default pose、action scale、clip、policy observation schema、history 初始化规则和控制频率；
- 用现有 MuJoCo viewer 做 sim2sim，但将公共观测构造提取为独立、可测试的 inference adapter；
- 添加 deterministic evaluation：固定 100 个 episode，报告成功率、episode return、速度误差、足滑、峰值力矩、落地误差/翻转成功率。

## 7. 旧 checkpoint 处理策略

现有 checkpoint：

| 实验 | 数量 | 网络输入 |
|---|---:|---|
| go2_trot_2 | 1 | actor 470 / critic 204 |
| go2_stairs_raw | 1 | actor 45 / critic 235 |
| go2_jump | 2 | actor 470 / critic 210 |
| go2_handstand | 2 | actor 48 / critic 89 |
| go2_leggedstand | 1 | actor 48 / critic 89 |
| go2_spring_jump | 1 | actor 470 / critic 195 |
| go2_backflip | 0 | — |

建议分三种兼容级别：

- **L0：仅归档**。保留 `.pt`、旧代码 commit 和运行说明。
- **L1：权重导入**。在网络层名和维度一致时转换 actor/critic tensor；optimizer 和 runner 状态通常不保证兼容。
- **L2：行为兼容**。只有 policy observation、历史 padding、关节顺序、action scale/default offset、资产动力学和控制频率都对齐后，才允许把旧策略用于 Isaac Lab 推理。

默认按 L0+重新训练推进，L2 作为独立验证工作流。不能仅因为 `state_dict` 能加载就宣称策略迁移成功。对同一输入张量，旧 actor 与转换后 actor 的输出应达到数值一致；随后还要通过闭环 rollout 对齐。

## 8. 迁移前应修正或冻结的旧实现问题

以下问题不应无审查地带入新框架：

1. README 声称部分任务仍“有问题”，且实物验证状态不明确；必须给每个任务标注 baseline 状态。
2. 多个环境的注释维度与实际拼接项不一致；维度应由 schema 自动计算和断言。
3. spring/backflip 的 command ranges 类缺失，实际通过 `commands[:, 2]` 作为触发位，语义混入速度命令字段。
4. latency 配置名称是“帧”，实际 motor/IMU buffer 在物理子步更新；需要统一成毫秒和 physics ticks。
5. 旧对称映射用带符号浮点索引和 `.cuda()`，设备与字段语义都不安全。
6. handstand/leggedstand privileged observation 中 `restitution_coeffs` 拼接了两次，需确认是有意占位还是 bug。
7. handstand/leggedstand reset 后 `reset_buf[env_ids] = 1`，且部分派生 base 状态没有在 reset 后立即完整刷新；新框架应由生命周期测试覆盖。
8. backflip 的成功判据大量依赖 `max_ang_vel_y > 7`，它不能证明完成约 2π 旋转；应另行定义 rotation progress 与稳定落地。
9. spring jump 的姿态判据使用 Euler 分量求和而非绝对值/姿态距离，可能正负抵消。
10. 旧项目在 7 个环境类中复制底层仿真与 randomization，修复容易只落到某一个任务；新代码必须集中实现。
11. reward scale 在旧框架 `_prepare_reward_function()` 中乘以 `dt`；新框架也通常按 step duration 加权，但自定义项必须用单元测试核对，避免重复或漏乘时间。
12. 旧导出只有网络文件，缺少 observation/action 元数据，容易造成 sim2sim 维度正确但语义错误。

对于这些项，迁移原则是：先在 manifest 标为 `legacy_behavior` 或 `confirmed_bug`。需要复现旧 checkpoint 时保留 legacy 分支；重新训练的 v1 任务采用修正行为，二者不能静默混用。

## 9. 验证矩阵与验收标准

### 9.1 静态和单元测试

- 所有 Gym ID 能被发现，Train/Play cfg 均能构造；
- RobotSpec 的 12 个关节和 4 个足端唯一匹配；
- policy/critic 每个 term 的维度和切片写入 snapshot；
- reward 对合成状态给出可手算结果；
- symmetry 对 observation/action 做两次映射后恢复原值；
- randomization 样本都在配置区间内；
- reset 只改变指定 env，历史和状态机正确清零；
- CPU 配置导入测试不启动 Simulator，仿真 smoke test 单独执行。

### 9.2 动力学和时序验证

- 固定 joint state/action 时，PD target 与 torque clipping 符合公式；
- physics dt=0.005、decimation=4、policy dt=0.02；
- 动作延迟 1～3 physics ticks，观测延迟采用相同明确定义；
- 旧 URDF 与候选 USD 的总质量、COM、关节轴/限位、默认姿态、碰撞接触点逐项对账；
- 关闭随机化时，同 seed reset 可复现。

### 9.3 任务验收

| 任务 | 核心指标 |
|---|---|
| trot | 速度 RMSE、偏航 RMSE、足滑、跌倒率、力矩/功率 |
| stairs | 分级地形通过率、上/下台阶成功率、课程最终等级 |
| jump | 腾空率、峰值高度、落地稳定率、命令跟踪 |
| hand/leg stand | 持续成功时长、支撑足滑动、非支撑足高度、恢复率 |
| spring jump | 无辅助起跳率、落点误差、落地姿态/速度 |
| backflip | 完整旋转率、稳定落地率、无辅助成功率、峰值力矩 |

验收不能只看训练 return。每项至少报告 5 个 seed 的均值和离散程度，并区分 train randomization 与 deterministic evaluation。

## 10. 工作量与风险

在硬件、驱动和现有 Isaac Lab 环境可正常启动的前提下，单人首次迁移的工程量粗估为 **15～25 个工作日**，不含为得到高质量 backflip 策略而进行的大规模奖励调参和实机验证：

| 阶段 | 粗估 |
|---|---:|
| 基线/manifest/资产对账 | 2～4 日 |
| extension 骨架与资产 | 2～3 日 |
| trot + 公共 MDP + symmetry/latency | 4～6 日 |
| stairs | 2～3 日 |
| jump + two-leg stand | 3～5 日 |
| spring/backflip 状态机 | 3～6 日 |
| 回归、导出、文档 | 2～4 日 |

主要风险按优先级排列：

1. **资产动力学差异**：官方 USD 与旧 URDF 不等价，直接导致旧策略不可用或奖励分布改变。
2. **观测语义漂移**：Euler/projected gravity、历史顺序、command scaling、reset padding 任一变化都会破坏 checkpoint。
3. **延迟时间基准错误**：policy tick 与 physics tick 相差 4 倍。
4. **技能状态机迁移错误**：单次 jump/flip 的跨步状态不是纯函数奖励。
5. **接触模型差异**：足端接触阈值、contact sensor history 和 PhysX 参数影响腾空/落地判定。
6. **奖励量纲漂移**：step_dt、平方/绝对值、坐标系和布尔 mask 优先级容易改变。
7. **旧 checkpoint 过度承诺**：权重可加载不代表闭环能工作。

## 11. 推荐的第一批可执行任务

实际开工时建议把第一个里程碑限定为：

1. 用官方 template 在 `My_quadruped_robot_lab` 生成外部 extension；
2. 建立 `GO2_LEGACY_CFG`，优先复现旧 URDF 控制参数；
3. 注册 `go2_trot` 及 `go2_trot_play` 变体；
4. 先实现无延迟、无随机化、单帧观测的 deterministic smoke test；
5. 再依次打开 10 帧历史、critic group、随机化、物理步延迟和 symmetry；
6. 每打开一项都保存回归数据，避免一次性迁移后无法定位偏差；
7. 完成 trot 短训和导出闭环后，再进入 stairs。

建议的常用命令形态为：

```bash
conda activate env_isaaclab
python -m pip install -e source/My_quadruped_robot_lab
python scripts/list_envs.py
python scripts/zero_agent.py --task go2_trot --num_envs 16
python scripts/rsl_rl/train.py --task go2_trot --headless
python scripts/rsl_rl/play.py --task go2_trot_play --checkpoint <path>
```

## 12. 最终交付定义

迁移完成应至少包含：

- 7 个任务的 Train/Play 注册；
- Go2 legacy/production 资产配置与对账文档；
- 可复用的四足 MDP term 和 RobotSpec 接口；
- RSL-RL 训练、恢复、评估、JIT/ONNX 导出；
- observation/action schema 与部署 metadata；
- 旧 checkpoint 清单、转换工具和兼容等级说明；
- 单元测试、仿真 smoke test、每任务 deterministic benchmark；
- 新增第二种机器狗的示例薄配置，用来证明多机器人抽象确实成立；
- README 中完整的安装、训练、播放、评估和新增机器人指南。

当“新增第二种四足机器人”只需要资产/执行器/名称映射和少量 task override，而无需复制环境或奖励代码时，才算真正实现了本次迁移的可扩展性目标。
