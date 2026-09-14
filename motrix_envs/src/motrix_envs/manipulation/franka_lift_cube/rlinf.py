# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""RLinf-oriented Franka lift-cube environment variant."""

from motrix_env_core import registry
from motrix_env_core.config import configclass

from .cfg import FrankaLiftCubeEnvCfg
from .franka_lift_cube_np import FrankaLiftCubeEnv


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
