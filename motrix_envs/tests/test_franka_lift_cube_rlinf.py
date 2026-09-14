# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Franka lift-cube RLinf environment variant."""

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry


def test_rlinf_variant_is_registered_without_replacing_original():
    original = registry.make_env_config("franka-lift-cube")
    rlinf = registry.make_env_config("franka-lift-cube-rlinf")

    assert type(original).__name__ == "FrankaLiftCubeEnvCfg"
    assert type(rlinf).__name__ == "FrankaLiftCubeRLinfEnvCfg"
    assert rlinf.scene.file == original.scene.file
