"""Тесты цикла обучения RSL-RL для PandaPickCube."""

import os
import tempfile

from device_utils import (
    apply_mjx_overrides_to_playground_cfg,
    rslrl_brax_wrapper_jax_device_rank,
)

import jax
import pytest
import torch
from mujoco_playground import registry
from mujoco_playground import wrapper_torch
from mujoco_playground.config import manipulation_params
from rsl_rl.runners import OnPolicyRunner

from config import build_runner_cfg


ENV_NAME = "PandaPickCube"

jax.config.update("jax_default_matmul_precision", "highest")


def test_rsl_rl_config_loads():
    """RSL-RL конфиг загружается через manipulation_params.rsl_rl_config."""
    cfg = manipulation_params.rsl_rl_config(ENV_NAME)
    assert cfg is not None
    assert "policy" in cfg
    assert "algorithm" in cfg
    assert cfg.algorithm.class_name == "PPO"


@pytest.fixture(scope="module")
def brax_env_and_raw():
    """Создаёт среду и wrapper (один раз на модуль)."""
    env_cfg = registry.get_default_config(ENV_NAME)
    apply_mjx_overrides_to_playground_cfg(env_cfg)
    raw_env = registry.load(ENV_NAME, config=env_cfg)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_rank = rslrl_brax_wrapper_jax_device_rank(device)

    brax_env = wrapper_torch.RSLRLBraxWrapper(
        raw_env,
        num_actors=64,
        seed=1,
        episode_length=env_cfg.episode_length,
        action_repeat=1,
        device_rank=device_rank,
    )
    brax_env.device = torch.device(device)
    # RSL-RL 3.x Logger обращается к env.cfg, которого нет в RSLRLBraxWrapper
    brax_env.cfg = {}
    return brax_env, raw_env


def test_wrapper_creates(brax_env_and_raw):
    """RSLRLBraxWrapper создаётся корректно."""
    brax_env, raw_env = brax_env_and_raw
    assert brax_env is not None
    assert brax_env.num_envs == 64
    assert brax_env.num_actions == raw_env.action_size


def test_short_training(brax_env_and_raw):
    """OnPolicyRunner проходит 2 итерации без ошибок и сохраняет чекпоинт."""
    brax_env, raw_env = brax_env_and_raw
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg_dict = build_runner_cfg(raw_env.observation_size)
    cfg_dict["max_iterations"] = 2
    cfg_dict["save_interval"] = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = OnPolicyRunner(brax_env, cfg_dict, tmpdir, device=device)
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)

        checkpoints = [f for f in os.listdir(tmpdir) if f.startswith("model_") and f.endswith(".pt")]
        assert len(checkpoints) > 0, "Чекпоинт не был сохранён после обучения"
