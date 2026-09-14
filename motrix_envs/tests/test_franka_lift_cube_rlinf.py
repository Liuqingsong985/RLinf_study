# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Franka lift-cube RLinf environment variant."""

import numpy as np

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry
from motrix_envs.manipulation.franka_lift_cube import rlinf


def test_rlinf_variant_is_registered_without_replacing_original():
    original = registry.make_env_config("franka-lift-cube")
    rlinf = registry.make_env_config("franka-lift-cube-rlinf")

    assert type(original).__name__ == "FrankaLiftCubeEnvCfg"
    assert type(rlinf).__name__ == "FrankaLiftCubeRLinfEnvCfg"
    assert rlinf.scene.file == original.scene.file


def test_reward_terms_progress_from_reach_to_success():
    gripper = np.array([[0.0, 0.0, 0.10]], dtype=np.float32)
    cube_far = np.array([[0.20, 0.0, 0.05]], dtype=np.float32)
    cube_grasped = np.array([[0.01, 0.0, 0.10]], dtype=np.float32)
    cube_lifted = np.array([[0.01, 0.0, 0.20]], dtype=np.float32)

    far = rlinf.compute_rlinf_reward_terms(gripper, cube_far, 0.05, 0.15)
    grasped = rlinf.compute_rlinf_reward_terms(gripper, cube_grasped, 0.05, 0.15)
    lifted = rlinf.compute_rlinf_reward_terms(gripper, cube_lifted, 0.15, 0.15)

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
