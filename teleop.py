"""Интерактивное управление роботом через MuJoCo GUI.

Робот берётся из config.ENV_NAME — по умолчанию myCobot 280 PI,
ENV_NAME=PandaPickCube откроет эталонную руку.

Запуск:
    python3 teleop.py

В окне viewer'а нажмите F1 — оверлей покажет полный список горячих
клавиш именно этой сборки MuJoCo. Вкладки справа "Joint" и "Control"
открывают слайдеры по каждому суставу и приводу.
"""

import signal
import sys
import time

from device_utils import apply_mjx_overrides_to_playground_cfg

import jax
import mujoco
import mujoco.viewer
from mujoco_playground import registry

from config import ENV_NAME

signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

jax.config.update("jax_default_matmul_precision", "highest")

env_cfg = registry.get_default_config(ENV_NAME)
apply_mjx_overrides_to_playground_cfg(env_cfg)
_env = registry.load(ENV_NAME, config=env_cfg)

model = _env.mj_model
data = mujoco.MjData(model)
if model.nkey > 0: # есть ли позы
    mujoco.mj_resetDataKeyframe(model, data, 0)
else:
    mujoco.mj_resetData(model, data)

print(f"Сцена «{ENV_NAME}» (MuJoCo Playground).")
print("  F1      — список всех горячих клавиш этой сборки MuJoCo")
print("  Пробел  — пауза")
print("  Esc     — выход")
print()

# открытие окна без физики
with mujoco.viewer.launch_passive(model, data) as viewer: 
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
