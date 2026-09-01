"""Базовый класс myCobot 280 PI для MuJoCo Playground (копия Panda).
https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/manipulation/franka_emika_panda/panda.py
"""

from pathlib import Path
from typing import Dict, Optional, Union

import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
import jax.numpy as jp
import numpy as np

_ARM_JOINTS = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint7_to_joint6",
]
_FINGER_JOINTS = ["left_finger_joint", "right_finger_joint"]

XML_DIR = Path(__file__).parent / "xmls"


def get_assets() -> Dict[str, bytes]:
  assets = {}
  mjx_env.update_assets(assets, XML_DIR, "*.xml")
  mjx_env.update_assets(assets, XML_DIR / "meshes", "*.stl")
  return assets


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.06,
      impl="jax",
      nconmax=12 * 8192,
      njmax=44,
  )


class MyCobotBase(mjx_env.MjxEnv):
  """Базовая среда для манипулятора myCobot 280 PI."""

  def __init__(
      self,
      xml_path: Path,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
  ):
    super().__init__(config, config_overrides)

    self._xml_path = str(xml_path)
    xml = xml_path.read_text()
    self._model_assets = get_assets()
    mj_model = mujoco.MjModel.from_xml_string(xml, assets=self._model_assets)
    mj_model.opt.timestep = self.sim_dt # шаг интегрирования физики

    self._mj_model = mj_model
    self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)
    self._action_scale = config.action_scale

  def _post_init(self, obj_name: str, keyframe: str):
    all_joints = _ARM_JOINTS + _FINGER_JOINTS
    self._robot_arm_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in _ARM_JOINTS
    ])
    self._robot_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in all_joints
    ])
    self._gripper_site = self._mj_model.site("gripper").id
    self._left_finger_geom = self._mj_model.geom("left_finger_pad").id
    self._right_finger_geom = self._mj_model.geom("right_finger_pad").id
    self._hand_geom = self._mj_model.geom("hand_capsule").id
    self._obj_body = self._mj_model.body(obj_name).id
    self._obj_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body(obj_name).jntadr[0]
    ]
    self._mocap_target = self._mj_model.body("mocap_target").mocapid
    self._floor_geom = self._mj_model.geom("floor").id
    self._init_q = self._mj_model.keyframe(keyframe).qpos
    self._init_obj_pos = jp.array(
        self._init_q[self._obj_qposadr : self._obj_qposadr + 3],
        dtype=jp.float32,
    )
    self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl
    self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
