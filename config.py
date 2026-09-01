"""Общий конфиг среды и RSL-RL для MyCobotPickCube.

Импортируется из train.py и inference.py — изменения здесь
автоматически применяются к обоим скриптам.
"""

import os

import mycobot_env  # noqa: F401  side effect: регистрирует "MyCobotPickCube"
from mujoco_playground.config import manipulation_params

# Через окружение подменяется на эталонную PandaPickCube из того же Playground.
ENV_NAME = os.environ.get("ENV_NAME", "MyCobotPickCube")

# Гиперпараметры сети
ACTOR_HIDDEN_DIMS = [512, 256, 128]
CRITIC_HIDDEN_DIMS = [512, 256, 128]
ACTIVATION = "elu"
INIT_NOISE_STD = 1.0

# Потолок разброса действий политики.
ACTION_STD_RANGE = (1e-6, 1.0)


def build_runner_cfg(obs_size) -> dict:
    """Строит конфиг OnPolicyRunner для RSL-RL 3.x.

    Args:
        obs_size: raw_env.observation_size — int или dict (асимметричные obs).

    Returns:
        Словарь, готовый для передачи в OnPolicyRunner(..., train_cfg=...).
    """
    cfg = manipulation_params.rsl_rl_config(ENV_NAME).to_dict()

    # Переопределения гиперпараметров через переменные окружения.
    for env_key, cfg_key in (("PPO_EPOCHS", "num_learning_epochs"),
                             ("PPO_MINIBATCHES", "num_mini_batches"),
                             ("PPO_LR", "learning_rate"),
                             ("PPO_ENTROPY", "entropy_coef")):
        raw = os.environ.get(env_key)
        if raw is not None:
            val = float(raw) if "." in raw or "e" in raw.lower() else int(raw)
            cfg["algorithm"][cfg_key] = val

    std_range = tuple(ACTION_STD_RANGE)
    std_max = os.environ.get("PPO_STD_MAX")
    if std_max is not None:
        std_range = (std_range[0], float(std_max))

    steps = os.environ.get("PPO_STEPS_PER_ENV")
    if steps:
        cfg["num_steps_per_env"] = int(steps)

    # Конвертируем формат playground (ключ «policy») в RSL-RL 3.x (отдельные
    # ключи «actor» и «critic» с полем class_name).
    cfg.pop("policy", {})
    cfg["actor"] = {
        "class_name": "rsl_rl.models.mlp_model.MLPModel",
        "hidden_dims": ACTOR_HIDDEN_DIMS,
        "activation": ACTIVATION,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": INIT_NOISE_STD,
            "std_range": std_range,
        },
    }
    cfg["critic"] = {
        "class_name": "rsl_rl.models.mlp_model.MLPModel",
        "hidden_dims": CRITIC_HIDDEN_DIMS,
        "activation": ACTIVATION,
    }

    if isinstance(obs_size, dict):
        cfg["obs_groups"] = {"actor": ["state"], "critic": ["privileged_state"]}
    else:
        cfg["obs_groups"] = {"actor": ["state"], "critic": ["state"]}
    return cfg
