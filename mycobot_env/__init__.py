"""Регистрирует MyCobotPickCube в реестре манипуляций mujoco_playground."""

from mujoco_playground._src import manipulation
from mujoco_playground._src import mjx_env

from mycobot_env.mycobot_pick import MyCobotPickCube, default_config

# Манипуляция.load() безоговорочно вызывает ensure_menagerie_exists(),
# которая при первом использовании клонировала бы репозиторий mujoco_menagerie.
mjx_env.MENAGERIE_PATH.mkdir(parents=True, exist_ok=True)

manipulation.register_environment(
    "MyCobotPickCube", MyCobotPickCube, default_config
)
