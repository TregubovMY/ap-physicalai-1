"""Тесты создания и базовой работы среды PandaPickCube."""

from device_utils import apply_mjx_overrides_to_playground_cfg

import jax
import pytest
from mujoco_playground import registry


ENV_NAME = "PandaPickCube"

jax.config.update("jax_default_matmul_precision", "highest")


@pytest.fixture(scope="module")
def env():
    cfg = registry.get_default_config(ENV_NAME)
    apply_mjx_overrides_to_playground_cfg(cfg)
    return registry.load(ENV_NAME, config=cfg)


def test_env_loads(env):
    """Среда PandaPickCube загружается через registry.load."""
    assert env is not None
    assert env.mj_model is not None
    assert env.mjx_model is not None


def test_observation_size(env):
    """Размер наблюдений — целое число > 0."""
    obs_size = env.observation_size
    if isinstance(obs_size, dict):
        assert "state" in obs_size
        for v in obs_size.values():
            if isinstance(v, tuple):
                assert all(d > 0 for d in v)
            else:
                assert v > 0
    else:
        assert obs_size > 0


def test_action_size(env):
    """Размер действия совпадает с конфигурацией PandaPickCube в playground (7 DOF + захват)."""
    assert env.action_size == 8


def test_reset(env):
    """env.reset() работает без ошибок."""
    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)
    assert state is not None
    assert state.obs is not None
    assert state.reward is not None
    assert state.done is not None


def test_step(env):
    """env.step() работает без ошибок."""
    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)

    action = jax.numpy.zeros(env.action_size)
    next_state = jax.jit(env.step)(state, action)

    assert next_state is not None
    assert next_state.obs is not None
    assert next_state.reward is not None
    assert next_state.done is not None
