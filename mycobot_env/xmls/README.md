# Откуда взялись эти XML

Модель робота для MuJoCo/MJX собрана вручную по данным вендора (Elephant
Robotics), а не сконвертирована автоматически из URDF.

## mycobot_arm.xml — рука + схват

Источник: репозиторий [mycobot_ros](https://github.com/elephantrobotics/mycobot_ros),
`mycobot_description/urdf/`:

- **Рука** — из `mycobot_280_pi/mycobot_280_pi.urdf`: смещения звеньев,
  пределы суставов, меши.
- **Схват** — из `parallel_gripper/mycobot_parallel_gripper.urdf`: две
  губки с ходом 7 мм, меши.
- **Крепление схвата к руке** — сустав `joint6output_to_gripper_base`
  (`xyz="0 0 0.034" rpy="1.579 0 0"`) взят из
  `mycobot_280_jn/mycobot_280_jn_parallel_gripper.urdf`. Готовый файл
  «рука+схват» есть только для варианта на Jetson Nano (JN), поэтому руку
  берём из Pi-файла, а стык — из JN-файла.
- **Инерции** — из проекта `mycobot_mujoco` (тоже вариант под Jetson Nano).
  У Pi и JN звенья одинаковые, отличается только высота основания, так что
  инерции взял от туда.

Чего в URDF не было и что добавлено вручную:
- приводы (`<actuator>`);
- имена геометрий — на них ссылаются контактные сенсоры в `sensor.xml`;
- контрольная точка (`site`) на схвате;
- синхронное движение губок: в URDF это задано связью `mimic`, которую
  импортёр MuJoCo не переносит, поэтому здесь она сделана через `tendon` +
  `equality` по подобию  [franka_emika_panda]
  (https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/panda.xml);
- отключение столкновений схвата с собственной рукой.

Масса и инерция корпуса схвата (`gripper_base`) пересчитаны напрямую из
меша через `trimesh` (меш замкнут, плотность взята как у ABS-пластика,
1050 кг/м³), остальным вкладом в инерцию пренебрегли.

## mycobot_scene.xml, sensor.xml — сцена и сенсоры

Взяты за основу из примера Franka Panda в
[mujoco_playground](https://github.com/google-deepmind/mujoco_playground)
(`manipulation/franka_emika_panda/xmls/mjx_scene.xml` и `sensor.xml`).
Числовые параметры сцены (размеры контактных меток и т.п.) уменьшены
пропорционально размеру myCobot относительно Panda (коэффициент ~0.4).

## mycobot_pick_cube.xml

Собственная сцена под задачу pick-and-place: кубик, mocap-цель и keyframe
"home" — писались с нуля под этот проект, не заимствованы.
