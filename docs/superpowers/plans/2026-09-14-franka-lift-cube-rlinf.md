# Franka Lift Cube RLinf Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增独立的 `franka-lift-cube-rlinf` 环境，复用现有 Franka 场景资产，并提供可诊断的低维观测、分阶段奖励、成功终止和小规模 SKRL/PyTorch PPO 配置。

**Architecture:** 在现有 `motrix_envs.manipulation.franka_lift_cube` 包内增加一个派生环境与配置，继承原环境的场景编译、控制、reset 和 simulator query，不复制 MJCF。派生环境只覆盖 observation、reward/termination bookkeeping，并通过独立 registry 名称暴露；训练配置继承现有 Franka PPO 配方并降低本机 smoke 默认规模。

**Tech Stack:** Python 3.10、NumPy、Gymnasium、MotrixLab DirectEnv、MotrixSim 0.10.1、Hydra、SKRL/PyTorch、pytest。

## Global Constraints

- 不修改现有 `franka-lift-cube` 的公开行为或注册名。
- 新环境注册名固定为 `franka-lift-cube-rlinf`。
- 复用现有 Franka MJCF、网格和控制语义，不复制资产。
- 环境保持 backend-neutral，不直接导入 MotrixSim 具体类型。
- 第一版不接入 π0.5、视觉编码器、RLinf 分布式 runtime、真机控制或多物体任务。
- 本机验证使用小规模 `num_envs`，不要求策略收敛或达到指定成功率。

---

## File Structure

- Create `motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/rlinf.py`: 派生配置、奖励纯函数和派生环境注册。
- Modify `motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/__init__.py`: 在基础环境注册后导入 RLinf 变体。
- Create `motrix_envs/tests/test_franka_lift_cube_rlinf.py`: registry、观测、奖励、成功终止和原环境回归测试。
- Modify `motrix_envs/tests/test_manipulation_direct_contract.py`: 将新环境纳入通用 DirectEnv 生命周期契约测试。
- Create `configs/task/franka-lift-cube-rlinf/skrl.ppo.yaml`: 新环境的共享 PPO 配置。
- Create `configs/task/franka-lift-cube-rlinf/skrl.ppo.torch.yaml`: PyTorch backend delta。

### Task 1: 注册独立 RLinf 环境

**Files:**
- Create: `motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/rlinf.py`
- Modify: `motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/__init__.py`
- Test: `motrix_envs/tests/test_franka_lift_cube_rlinf.py`

**Interfaces:**
- Consumes: `FrankaLiftCubeEnvCfg` 与 `FrankaLiftCubeEnv`。
- Produces: `FrankaLiftCubeRLinfEnvCfg`、`FrankaLiftCubeRLinfEnv` 和 registry 名称 `franka-lift-cube-rlinf`。

- [ ] **Step 0: 安装测试与格式检查工具**

Run: `powershell -ExecutionPolicy Bypass -File install.ps1 --all --gpu cuda --skrl-torch`

Expected: exit 0；`.venv\Scripts\python.exe -m pytest --version` 和 `.venv\Scripts\python.exe -m ruff --version` 均成功。

- [ ] **Step 1: 写 registry 失败测试**

```python
import motrix_envs  # noqa: F401
from motrix_env_core import registry


def test_rlinf_variant_is_registered_without_replacing_original():
    original = registry.make_env_config("franka-lift-cube")
    rlinf = registry.make_env_config("franka-lift-cube-rlinf")

    assert type(original).__name__ == "FrankaLiftCubeEnvCfg"
    assert type(rlinf).__name__ == "FrankaLiftCubeRLinfEnvCfg"
    assert rlinf.scene.file == original.scene.file
```

- [ ] **Step 2: 运行测试并确认因新注册名不存在而失败**

Run: `.venv\Scripts\python.exe -m pytest motrix_envs/tests/test_franka_lift_cube_rlinf.py::test_rlinf_variant_is_registered_without_replacing_original -v`

Expected: FAIL，错误包含 `franka-lift-cube-rlinf` 未注册。

- [ ] **Step 3: 添加最小派生配置和环境**

Create `rlinf.py` with:

```python
import gymnasium as gym
import numpy as np

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState, NpObs
from motrix_env_core.config import configclass

from .cfg import FrankaLiftCubeEnvCfg
from .franka_lift_cube_np import FrankaLiftCubeEnv


@registry.envcfg("franka-lift-cube-rlinf")
@configclass
class FrankaLiftCubeRLinfEnvCfg(FrankaLiftCubeEnvCfg):
    success_height: float = 0.15
    grasp_distance: float = 0.05


@registry.env("franka-lift-cube-rlinf")
class FrankaLiftCubeRLinfEnv(FrankaLiftCubeEnv):
    _cfg: FrankaLiftCubeRLinfEnvCfg
```

Append to `__init__.py` after the existing base import:

```python
from . import rlinf  # noqa: F401
```

- [ ] **Step 4: 运行 registry 测试并确认通过**

Run: `.venv\Scripts\python.exe -m pytest motrix_envs/tests/test_franka_lift_cube_rlinf.py::test_rlinf_variant_is_registered_without_replacing_original -v`

Expected: PASS。

- [ ] **Step 5: 提交环境注册**

```bash
git add motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/rlinf.py motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/__init__.py motrix_envs/tests/test_franka_lift_cube_rlinf.py
git commit -m "feat: register Franka lift cube RLinf environment"
```

### Task 2: 增加 RLinf 观测、奖励和成功语义

**Files:**
- Modify: `motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/rlinf.py`
- Modify: `motrix_envs/tests/test_franka_lift_cube_rlinf.py`
- Modify: `motrix_envs/tests/test_manipulation_direct_contract.py`

**Interfaces:**
- Consumes: 父环境缓存字段 `robot_joint_pos`、`robot_joint_vel`、`gripper_pos`、`gripper_quat`、`cube_pos`、`cube_quat` 及 `state.info`。
- Produces: 46 维 policy observation；`state.info["success"]`；`reward_reach`、`reward_grasp`、`reward_lift`、`reward_success`；成功时 `state.terminated=True`。

- [ ] **Step 1: 写纯奖励函数和生命周期失败测试**

Add tests:

```python
import numpy as np

from motrix_env_core import registry
from motrix_envs.manipulation.franka_lift_cube.rlinf import compute_rlinf_reward_terms


def test_reward_terms_progress_from_reach_to_success():
    gripper = np.array([[0.0, 0.0, 0.10]], dtype=np.float32)
    cube_far = np.array([[0.20, 0.0, 0.05]], dtype=np.float32)
    cube_grasped = np.array([[0.01, 0.0, 0.10]], dtype=np.float32)
    cube_lifted = np.array([[0.01, 0.0, 0.20]], dtype=np.float32)

    far = compute_rlinf_reward_terms(gripper, cube_far, 0.05, 0.15)
    grasped = compute_rlinf_reward_terms(gripper, cube_grasped, 0.05, 0.15)
    lifted = compute_rlinf_reward_terms(gripper, cube_lifted, 0.15, 0.15)

    assert grasped["reach"][0] > far["reach"][0]
    assert grasped["grasp"][0] == 1.0
    assert lifted["lift"][0] > grasped["lift"][0]
    assert lifted["success"][0]


def test_rlinf_environment_step_contract():
    env = registry.make("franka-lift-cube-rlinf", num_envs=2)
    state = env.step(np.zeros((2, 8), dtype=np.float32))

    assert env.observation_space.shape == (46,)
    assert state.obs.policy.shape == (2, 46)
    assert np.isfinite(state.obs.policy).all()
    assert np.isfinite(state.reward).all()
    assert state.info["success"].shape == (2,)
    for name in ("reach", "grasp", "lift", "success"):
        assert state.info[f"reward_{name}"].shape == (2,)
```

Also append `"franka-lift-cube-rlinf"` to `_MANIPULATION_ENVS` in `test_manipulation_direct_contract.py` so the existing reset/read/observation lifecycle checks execute against the new variant.

- [ ] **Step 2: 运行测试并确认缺少奖励函数或 46 维观测而失败**

Run: `.venv\Scripts\python.exe -m pytest motrix_envs/tests/test_franka_lift_cube_rlinf.py -v`

Expected: FAIL，首先指向 `compute_rlinf_reward_terms` 不存在。

- [ ] **Step 3: 实现纯奖励函数**

Add to `rlinf.py`:

```python
def compute_rlinf_reward_terms(
    gripper_pos: np.ndarray,
    cube_pos: np.ndarray,
    grasp_distance: float,
    success_height: float,
) -> dict[str, np.ndarray]:
    distance = np.linalg.norm(cube_pos - gripper_pos, axis=-1)
    reach = 1.0 - np.tanh(distance / 0.1)
    grasp = (distance < grasp_distance).astype(np.float32)
    lift = np.clip((cube_pos[:, 2] - 0.05) / (success_height - 0.05), 0.0, 1.0).astype(np.float32)
    success = cube_pos[:, 2] >= success_height
    return {"reach": reach, "grasp": grasp, "lift": lift, "success": success}
```

- [ ] **Step 4: 实现 46 维观测**

In `FrankaLiftCubeRLinfEnv.__init__`, call `super()` and replace `_obs_dim` and `_observation_space`:

```python
self._obs_dim = 46
self._observation_space = gym.spaces.Box(-np.inf, np.inf, (self._obs_dim,), dtype=np.float32)
```

Override `compute_observation`:

```python
def compute_observation(self, state: ArrayEnvState):
    dof_pos = self.get_dof_pos(slice(None))
    dof_vel = self.get_dof_vel(slice(None))
    gripper_pos = self.sim_data["gripper_pos"]
    gripper_quat = self.sim_data["gripper_quat"]
    cube_pose = self.get_cube_pose(slice(None))
    cube_relative = cube_pose[:, :3] - gripper_pos
    obs = np.concatenate(
        [
            self._get_joint_pos_rel(dof_pos),
            self._get_joint_vel_rel(dof_vel),
            gripper_pos,
            gripper_quat,
            cube_pose,
            cube_relative,
            state.info["commands"],
            state.info["current_actions"],
        ],
        axis=-1,
    ).astype(np.float32)
    setattr(state, "obs", NpObs(policy=obs))
    return state
```

- [ ] **Step 5: 实现奖励、成功标记和终止**

Override `reset` to add correctly shaped diagnostic arrays:

```python
def reset(self, env_ids):
    info = super().reset(env_ids)
    count = len(env_ids)
    info["success"] = np.zeros(count, dtype=bool)
    for name in ("reach", "grasp", "lift", "success"):
        info[f"reward_{name}"] = np.zeros(count, dtype=np.float32)
    return info
```

Override `compute_transition`:

```python
def compute_transition(self, state: ArrayEnvState):
    self.sim_data.execute()
    failure = super()._check_termination(state)
    terms = compute_rlinf_reward_terms(
        self.sim_data["gripper_pos"],
        self.sim_data["cube_pos"],
        self._cfg.grasp_distance,
        self._cfg.success_height,
    )
    success = terms["success"]
    state.reward = (
        1.5 * terms["reach"]
        + 5.0 * terms["grasp"]
        + 20.0 * terms["lift"]
        + 100.0 * success.astype(np.float32)
    ).astype(np.float32)
    state.terminated = np.logical_or(failure, success)
    state.info["success"] = success
    for name, value in terms.items():
        state.info[f"reward_{name}"] = value.astype(np.float32)
    return state
```

- [ ] **Step 6: 运行新环境与现有 manipulation contract 测试**

Run: `.venv\Scripts\python.exe -m pytest motrix_envs/tests/test_franka_lift_cube_rlinf.py motrix_envs/tests/test_manipulation_direct_contract.py -v`

Expected: 全部 PASS。

- [ ] **Step 7: 提交环境行为**

```bash
git add motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/rlinf.py motrix_envs/tests/test_franka_lift_cube_rlinf.py
git commit -m "feat: add RLinf lift reward and success contract"
```

### Task 3: 添加小规模 PyTorch PPO 配置并完成 smoke 验收

**Files:**
- Create: `configs/task/franka-lift-cube-rlinf/skrl.ppo.yaml`
- Create: `configs/task/franka-lift-cube-rlinf/skrl.ppo.torch.yaml`
- Test: `motrix_rl/tests/test_task_configs.py`

**Interfaces:**
- Consumes: registry 环境 `franka-lift-cube-rlinf` 与 `/algo_base@algo: skrl.ppo`。
- Produces: Hydra task options `franka-lift-cube-rlinf/skrl.ppo` 和 `franka-lift-cube-rlinf/skrl.ppo.torch`。

- [ ] **Step 1: 先运行不存在的 Hydra task 并确认失败**

Run: `.venv\Scripts\python.exe scripts/train.py task=franka-lift-cube-rlinf/skrl.ppo.torch num_envs=2 algo.trainer.timesteps=1`

Expected: FAIL，Hydra 报告找不到 `franka-lift-cube-rlinf/skrl.ppo.torch`。

- [ ] **Step 2: 添加共享 PPO 配置**

Create `configs/task/franka-lift-cube-rlinf/skrl.ppo.yaml`:

```yaml
# @package _global_
defaults:
  - /algo_base@algo: skrl.ppo
  - _self_
task:
  env: franka-lift-cube-rlinf
  rllib: skrl
  algo: ppo
  train_backend: null
num_envs: 64
play_num_envs: 1
seed: 42
algo:
  trainer:
    timesteps: 100000
  agent:
    rollouts: 24
```

- [ ] **Step 3: 添加 PyTorch 配置 delta**

Create `configs/task/franka-lift-cube-rlinf/skrl.ppo.torch.yaml`:

```yaml
# @package _global_
defaults:
  - /task/franka-lift-cube-rlinf/skrl.ppo@_global_
  - _self_
task:
  train_backend: torch
algo:
  agent:
    learning_epochs: 8
    mini_batches: 4
    learning_rate: 0.0003
    learning_rate_scheduler_kwargs:
      kl_threshold: 0.01
    entropy_loss_scale: 0.001
    rewards_shaper_scale: 0.01
```

- [ ] **Step 4: 验证所有 Hydra task 配置**

Run: `.venv\Scripts\python.exe -m pytest motrix_rl/tests/test_task_configs.py -v`

Expected: 全部 PASS，并发现新 task 的两个配置选项。

- [ ] **Step 5: 运行最小无界面 PPO smoke**

Run: `.venv\Scripts\python.exe scripts/train.py task=franka-lift-cube-rlinf/skrl.ppo.torch num_envs=2 algo.trainer.timesteps=48 render=false checkpoint.interval=0`

Expected: exit 0；控制台完成至少一个 rollout/update，reward 和 observation 无 NaN。

- [ ] **Step 6: 运行最终聚焦回归**

Run: `.venv\Scripts\python.exe -m pytest motrix_envs/tests/test_franka_lift_cube_rlinf.py motrix_envs/tests/test_manipulation_direct_contract.py motrix_rl/tests/test_task_configs.py -v`

Expected: 全部 PASS。

- [ ] **Step 7: 检查格式与工作区差异**

Run: `.venv\Scripts\python.exe -m ruff check motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/rlinf.py motrix_envs/tests/test_franka_lift_cube_rlinf.py`

Expected: `All checks passed!`。

Run: `git diff --check`

Expected: 无输出，exit 0。

- [ ] **Step 8: 提交训练配置**

```bash
git add configs/task/franka-lift-cube-rlinf motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/rlinf.py motrix_envs/tests/test_franka_lift_cube_rlinf.py
git commit -m "feat: add Franka RLinf PPO smoke configuration"
```

### Task 4: 推送到用户仓库

**Files:**
- No source changes.

**Interfaces:**
- Consumes: 已通过测试的本地提交和目标仓库 `https://github.com/Liuqingsong985/RLinf_study.git`。
- Produces: 用户仓库中的功能分支 `franka-lift-cube-rlinf`。

- [ ] **Step 1: 确认目标远端和分支不会覆盖现有默认分支**

Run: `git ls-remote --heads https://github.com/Liuqingsong985/RLinf_study.git`

Expected: 成功列出现有远端分支，或空输出表示新仓库尚无分支。

- [ ] **Step 2: 添加独立远端并确认当前功能分支**

Run: `git remote add study https://github.com/Liuqingsong985/RLinf_study.git`

Expected: exit 0；`origin` 仍指向 `Motphys/MotrixLab`。

Run: `git branch --show-current`

Expected: 当前分支为 `franka-lift-cube-rlinf`。

- [ ] **Step 3: 推送功能分支**

Run: `git push -u study franka-lift-cube-rlinf`

Expected: 远端创建 `franka-lift-cube-rlinf`，不修改其默认分支。
