"""Автовыбор устройства PyTorch и настроек MuJoCo под хост (macOS / Linux, CUDA / CPU / MPS)."""

from __future__ import annotations

import platform
from typing import Any, Optional


def apply_mjx_overrides_to_playground_cfg(env_cfg: Any) -> Any:
    """На CPU переключает MJX бэкенд Playground с warp (требует CUDA) на jax.

    Вызывать после ``get_default_config`` и до ``registry.load``.
    На GPU-хосте ничего не делает.
    """
    import jax

    try:
        if jax.devices("cuda"):
            return env_cfg
    except RuntimeError:
        pass

    with env_cfg.unlocked():
        env_cfg.impl = "jax"

    return env_cfg


def default_torch_device(explicit: Optional[str] = None) -> str:
    """Возвращает явное устройство или лучший доступный: cuda → mps → cpu."""
    if explicit is not None:
        return explicit
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if platform.system() == "Darwin":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    return "cpu"


def rslrl_brax_wrapper_jax_device_rank(torch_device: str) -> Optional[int]:
    """device_rank для RSLRLBraxWrapper: GPU rank при CUDA, None иначе."""
    if "cuda" in torch_device:
        return int(torch_device.split(":")[-1])
    return None


def jax_to_torch_synced(obs: Any, device: str) -> Any:
    """Наблюдение из JAX в PyTorch, дождавшись готовности данных.

    Штатная пара ``wrapper_torch._jax_to_torch`` / ``_torch_to_jax`` передаёт
    буфер на видеокарте по указателю, без копирования. Это быстро, но JAX и
    PyTorch работают каждый в своём потоке CUDA, и передача указателя не
    добавляет ожидания между ними: PyTorch может начать читать наблюдение
    раньше, чем JAX закончил его писать, а JAX — прочитать команду раньше,
    чем PyTorch её дописал.

    Гонка измерена: один и тот же чекпоинт с одним и тем же зерном давал в
    третьем эпизоде подъём то 101.2 мм, то 2.4 мм. Через хост данные идут
    медленнее, но результат воспроизводим побитово. Для оценки и просмотра
    скорость не важна, а воспроизводимость — единственное, ради чего они
    существуют.
    """
    import jax
    import numpy as np
    import torch

    return torch.as_tensor(np.asarray(jax.block_until_ready(obs)), device=device)


def torch_to_jax_synced(action: Any, device: str) -> Any:
    """Команда из PyTorch в JAX, после того как PyTorch её дописал."""
    import jax.numpy as jnp
    import numpy as np
    import torch

    if "cuda" in str(device):
        torch.cuda.synchronize()
    return jnp.asarray(np.asarray(action.detach().cpu()))


def default_num_envs(explicit: Optional[int] = None) -> int:
    """Параллельные среды: 4096 на GPU, 64 на CPU/MPS."""
    if explicit is not None:
        return explicit
    import torch

    if torch.cuda.is_available():
        return 4096
    return 64
