# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""RLinf-oriented Franka lift-cube environment variant."""

import gymnasium as gym
import numpy as np

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState, NpObs
from motrix_env_core.config import configclass

from .cfg import FrankaLiftCubeEnvCfg
from .franka_lift_cube_np import FrankaLiftCubeEnv


def compute_rlinf_reward_terms(
    gripper_pos: np.ndarray,
    cube_pos: np.ndarray,
    grasp_distance: float,
    success_height: float,
) -> dict[str, np.ndarray]:
    """Compute independently inspectable reward stages for RL rollouts."""
    distance = np.linalg.norm(cube_pos - gripper_pos, axis=-1)
    reach = 1.0 - np.tanh(distance / 0.1)
    grasp = (distance < grasp_distance).astype(np.float32)
    lift = np.clip((cube_pos[:, 2] - 0.05) / (success_height - 0.05), 0.0, 1.0).astype(np.float32)
    success = cube_pos[:, 2] >= success_height
    return {"reach": reach, "grasp": grasp, "lift": lift, "success": success}


@registry.envcfg("franka-lift-cube-rlinf")
@configclass
class FrankaLiftCubeRLinfEnvCfg(FrankaLiftCubeEnvCfg):
    """Configuration for the RLinf-oriented lift-cube contract."""

    success_height: float = 0.15
    grasp_distance: float = 0.05


@registry.env("franka-lift-cube-rlinf")
class FrankaLiftCubeRLinfEnv(FrankaLiftCubeEnv):
    """Franka lift-cube task with explicit RLinf rollout semantics."""

    _cfg: FrankaLiftCubeRLinfEnvCfg

    def __init__(self, cfg: FrankaLiftCubeRLinfEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs=num_envs, backend=backend)
        self._obs_dim = 46
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (self._obs_dim,), dtype=np.float32)

    def compute_observation(self, state: ArrayEnvState):
        """Publish the low-dimensional policy contract used by RL rollouts."""
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

    def compute_transition(self, state: ArrayEnvState):
        """Refresh simulator data and expose reward stages and success."""
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
            1.5 * terms["reach"] + 5.0 * terms["grasp"] + 20.0 * terms["lift"] + 100.0 * success.astype(np.float32)
        ).astype(np.float32)
        state.terminated = np.logical_or(failure, success)
        state.info["success"] = success
        state.metrics = {}
        for name, value in terms.items():
            diagnostic = value.astype(np.float32)
            state.info[f"reward_{name}"] = diagnostic
            state.metrics[f"reward/{name}"] = diagnostic
        return state
