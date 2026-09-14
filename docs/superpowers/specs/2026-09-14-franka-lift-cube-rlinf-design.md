# Franka Lift Cube RLinf 场景设计

## 目标

在不修改现有 `franka-lift-cube` 环境行为的前提下，基于其场景资产和实现模式新增独立环境 `franka-lift-cube-rlinf`。第一版用于验证 MotrixLab 中机械臂操作任务的完整仿真闭环，并为后续对接 RLinf 的 rollout、奖励和终止接口提供稳定基线。

## 范围

第一版包含桌面、Franka 机械臂、夹爪和单个方块。任务是让机械臂接近、夹持并将方块抬升到指定高度。环境必须支持创建、重置、步进和小规模 PPO 训练。

第一版不接入 π0.5、视觉编码器、RLinf 分布式训练、真机控制、复杂领域随机化或多物体任务。这些能力在基础场景通过验收后单独设计。

## 实现策略

复用现有 `franka-lift-cube` 的机器人与 MJCF 资产、配置结构、控制方式和注册模式，但注册为新的环境名称。新增代码只覆盖 RLinf 基线真正需要区分的任务配置和接口，不复制大段通用实现，也不改变原环境的公开契约。

环境配置使用仓库约定的 `@configclass`，并继承与现有 Franka 操作任务一致的运行时配置基类。环境配置先注册，环境类后注册；仿真后端继续通过配置选择，不把 MotrixSim 专用类型泄漏到 backend-neutral 环境逻辑中。

## 场景组成

- 固定桌面和地面。
- 固定基座的 Franka 机械臂及双指夹爪。
- 一个可自由运动、颜色醒目的立方体。
- 与现有场景一致的灯光和观察相机，保证可视化检查方便。
- 单环境先验证，随后允许通过 `num_envs` 扩展并行环境数量。

## 环境接口

### 观测

第一版使用低维状态观测：机械臂关节位置与速度、夹爪状态、末端执行器位姿、方块位姿，以及方块相对末端执行器的位置。观测字段顺序和 shape 固定，并由正向契约测试验证。

### 动作

沿用现有 `franka-lift-cube` 的动作表达和控制器，避免第一版同时改变任务与控制语义。动作覆盖机械臂控制和夹爪开合，并保持现有动作缩放、限幅和仿真步长设置。

### 奖励

奖励采用分阶段结构：

1. 末端执行器接近方块的连续奖励。
2. 夹爪形成稳定抓取的奖励。
3. 方块离开桌面的抬升奖励。
4. 方块达到目标高度的成功奖励。

第一版以“成功抬升”为最终目标，不加入步数优化、动作平滑惩罚或复杂 shaped reward，避免奖励来源难以诊断。

### 终止与重置

达到目标高度时标记成功终止；超过 episode horizon 时截断。方块或机器人状态在 reset 时恢复到合法初始范围。若复用环境已有掉落处理，则保持其现有语义；不额外创造不一致的生命周期管理。

## 配置与运行入口

- 环境注册名：`franka-lift-cube-rlinf`。
- PPO 配置沿用现有 Franka SKRL/PyTorch 配方，只覆盖新环境名称和确有必要的任务参数。
- 可视化入口：`python scripts/view.py env=franka-lift-cube-rlinf`。
- 训练入口：`python scripts/train.py task=franka-lift-cube-rlinf/skrl.ppo task.train_backend=torch`。

由于本机 RTX 3060 Laptop 只有 6GB 显存，验证时使用小规模 `num_envs`；大规模训练不属于本次场景搭建验收范围。

## 错误处理与可诊断性

依赖环境、资产解析、注册名或场景编译失败时直接报告具体异常，不静默切换后端或自动降级。奖励和终止信息应能从环境输出或训练日志中区分，便于后续 RLinf rollout adapter 映射 success、timeout 和失败轨迹。

## 测试与验收

实现采用测试驱动流程，至少验证：

- 新环境及配置能通过 registry 创建。
- observation/action space 与实际 reset/step 输出一致。
- 单环境 reset 和多步 step 输出有限值且 shape 正确。
- 成功条件、终止与超时截断语义正确。
- 奖励在接近、抓取和抬升阶段按预期变化。
- 现有 `franka-lift-cube` 注册和基本行为不受影响。
- MotrixSim 后端能够编译新场景。
- 使用小规模 `num_envs` 启动 SKRL/PyTorch PPO smoke run。

验收完成的定义是：上述聚焦测试通过，场景可视化可启动，PPO smoke run 能产生正常的 observation、action、reward、termination 和 checkpoint/日志输出。第一版不以训练收敛或达到特定成功率作为完成条件。

## 后续 RLinf 接口边界

后续接入 RLinf 时，保持本场景的 reset、step、观测、动作、奖励和终止语义不变，在独立 adapter 中完成张量格式、并行 rollout 和 π0.5 action chunk 映射。视觉观测、语言 prompt、action chunk、executed horizon 和 RLinf checkpoint lineage 将作为下一阶段设计，不混入本次基础环境实现。
