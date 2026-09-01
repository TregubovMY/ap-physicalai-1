"""Поднесите куб к мишени с помощью myCobot 280 PI (PandaPickCube)."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member

from mycobot_env import mycobot_base as mycobot
from mycobot_env.mycobot_base import XML_DIR

# Ширина tanh-ядер награды: грубое приближение, точное приближение, подход
# к кубу. Подобраны под размер этой руки и её схвата.
REACH_STD = 0.05
GOAL_STD = 0.15
GOAL_FINE_STD = 0.025

# Шаг команды схвата за один вызов step, в метрах (у руки — в радианах,
# единицы разные, поэтому масштаб отдельный).
GRIPPER_ACTION_SCALE = 0.0025

# Высота подъёма кубика над стартовой точкой, которая засчитывается как
# "поднят".
MINIMAL_LIFT = 0.015

# Радиус вокруг цели, в котором кубик считается доставленным.
SUCCESS_RADIUS = 0.05

# Запас к геометрическому положению губки на кубике, отделяющий реальный
# захват от кулака, сомкнутого в пустоте.
GRASP_GATE_MARGIN = 0.0007


def default_config() -> config_dict.ConfigDict:
  """Возвращает конфигурацию по умолчанию для задачи bring_to_target."""
  config = config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.06,
      reward_config=config_dict.create(
          scales=config_dict.create(
              gripper_box=4.0,
              object_goal_tracking=6.0,
              object_goal_tracking_fine=2.0,
              no_floor_collision=0.25,
              robot_target_qpos=0.3,
          ),
      ),
      arm_init_noise=0.0,
      impl="jax",
      nconmax=24 * 2048,
      njmax=128,
  )
  return config


class MyCobotPickCube(mycobot.MyCobotBase):
  """Поднесите куб к мишени с помощью myCobot 280 PI."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
  ):
    xml_path = XML_DIR / "mycobot_pick_cube.xml"
    super().__init__(xml_path, config, config_overrides)
    self._post_init(obj_name="box", keyframe="home")

    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    ]
    self._pad_box_found_sensor = [
        self._mj_model.sensor(name).id
        for name in ["left_finger_box_found", "right_finger_box_found"]
    ]

    self._cube_size = 2.0 * float(
        self._mj_model.geom_size[self._mj_model.geom("box").id][0]
    )
    self._open_jaw_gap = self._measure_open_jaw_gap()

    self._action_scale_vec = jp.concatenate([
        jp.full((self._mjx_model.nu - 1,), self._action_scale),
        jp.array([GRIPPER_ACTION_SCALE]),
    ])

    jaws_on_cube = (self._open_jaw_gap - self._cube_size) / 2.0
    self._grip_gate_qpos = float(jaws_on_cube + GRASP_GATE_MARGIN)

    arm_jnt_ids = [self._mj_model.joint(j).id for j in mycobot._ARM_JOINTS]
    arm_range = self._mj_model.jnt_range[arm_jnt_ids]
    self._arm_lower = jp.array(arm_range[:, 0])
    self._arm_upper = jp.array(arm_range[:, 1])

  def _measure_open_jaw_gap(self) -> float:
    """Внутренний зазор между полностью открытыми накладками, в метрах."""
    model = self._mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    left = model.geom("left_finger_pad").id
    right = model.geom("right_finger_pad").id
    axis = data.geom_xpos[right] - data.geom_xpos[left]
    axis /= np.linalg.norm(axis)
    reach = {}
    for geom, side in ((left, "l"), (right, "r")):
      sx, sy, sz = model.geom_size[geom]
      corners = np.array(
          [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]
      )
      world = (
          data.geom_xmat[geom].reshape(3, 3) @ corners.T
      ).T + data.geom_xpos[geom]
      reach[side] = world @ axis
    return float(reach["r"].min() - reach["l"].max())

  def reset(self, rng: jax.Array) -> State:
    rng, rng_box, rng_target, rng_arm = jax.random.split(rng, 4)

    # Случайный старт: политика не должна полагаться на одну заученную траекторию.
    noise = self._config.arm_init_noise
    arm_noise = jax.random.uniform(rng_arm, (6,), minval=-noise, maxval=noise)
    arm_qpos = jp.clip(
        jp.array(self._init_q)[self._robot_arm_qposadr] + arm_noise,
        self._arm_lower,
        self._arm_upper,
    )

    box_pos = (
        jax.random.uniform(
            rng_box,
            (3,),
            minval=jp.array([-0.065, -0.065, 0.0]),
            maxval=jp.array([0.065, 0.065, 0.0]),
        )
        + self._init_obj_pos
    )

    target_pos = (
        jax.random.uniform(
            rng_target,
            (3,),
            minval=jp.array([-0.065, -0.065, 0.065]),
            maxval=jp.array([0.065, 0.065, 0.131]),
        )
        + self._init_obj_pos
    )
    target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    finger_qpos = self._init_q[self._robot_qposadr[6]]

    init_q = (
        jp.array(self._init_q)
        .at[self._robot_arm_qposadr]
        .set(arm_qpos)
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(box_pos)
        .at[self._obj_qposadr + 3 : self._obj_qposadr + 7]
        .set(jp.array([1.0, 0.0, 0.0, 0.0]))
        .at[self._robot_qposadr[6]]
        .set(finger_qpos)
        .at[self._robot_qposadr[7]]
        .set(finger_qpos)
    )
    # ctrl должен совпадать с qpos при старте: иначе привод-ПД на первом же
    # шаге дёрнет сустав обратно к прежней команде.
    init_ctrl = (
        jp.array(self._init_ctrl).at[:6].set(arm_qpos).at[-1].set(finger_qpos)
    )
    init_qvel = jp.zeros(self._mjx_model.nv, dtype=float)

    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=init_qvel,
        ctrl=init_ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
    )

    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        # Доля шагов эпизода, где задача решена (поднят, в схвате, у цели):
        # награда сама по себе про задачу не говорит, она может расти и при
        # нулевой доле решённых.
        "task_success": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }
    info = {
        "rng": rng,
        "target_pos": target_pos,
        "lifted": jp.array(0.0, dtype=float),
    }
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    state = State(data, obs, reward, done, metrics, info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    delta = action * self._action_scale_vec
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
    box_pos = data.xpos[self._obj_body]
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    goal_err = jp.linalg.norm(state.info["target_pos"] - box_pos)
    task_success = state.info["lifted"] * (goal_err < SUCCESS_RADIUS)

    state.metrics.update(
        **raw_rewards,
        out_of_bounds=out_of_bounds.astype(float),
        task_success=task_success.astype(float),
    )

    obs = self._get_obs(data, state.info)
    state = State(data, obs, reward, done, state.metrics, state.info)

    return state

  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    box_pos = data.xpos[self._obj_body]
    gripper_pos = data.site_xpos[self._gripper_site]

    # Держит = обе накладки касаются кубика И губки стоят на нём, а не
    # сомкнуты в кулак вокруг пустоты.
    pads = [
        data.sensordata[self._mj_model.sensor_adr[sid]] > 0
        for sid in self._pad_box_found_sensor
    ]
    jaws_on_object = data.qpos[self._robot_qposadr[6]] < self._grip_gate_qpos
    holding = ((sum(pads) > 1) & jaws_on_object).astype(float)
    high = (box_pos[2] > self._init_obj_pos[2] + MINIMAL_LIFT).astype(float)
    lifted = high * holding
    info["lifted"] = lifted

    # Награда за приближение удерживаемого кубика к цели — тем больше, чем
    # ближе, только пока держит правильно.
    goal_err = jp.linalg.norm(info["target_pos"] - box_pos)
    object_goal_tracking = holding * (1 - jp.tanh(goal_err / GOAL_STD))
    object_goal_tracking_fine = holding * (1 - jp.tanh(goal_err / GOAL_FINE_STD))

    gripper_box = 1 - jp.tanh(25 * jp.linalg.norm(box_pos - gripper_pos))
    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - jp.array(self._init_q)[self._robot_arm_qposadr]
        )
    )
    hand_floor = [
        data.sensordata[self._mj_model.sensor_adr[sid]] > 0
        for sid in self._floor_hand_found_sensor
    ]
    no_floor_collision = (1 - (sum(hand_floor) > 0)).astype(float)

    return {
        "gripper_box": gripper_box,
        "object_goal_tracking": object_goal_tracking,
        "object_goal_tracking_fine": object_goal_tracking_fine,
        "no_floor_collision": no_floor_collision,
        "robot_target_qpos": robot_target_qpos,
    }

  def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
        info["target_pos"] - data.xpos[self._obj_body],
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])

    return obs
