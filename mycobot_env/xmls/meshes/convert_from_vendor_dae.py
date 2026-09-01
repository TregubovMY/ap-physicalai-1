"""Пересборка STL из вендорских COLLADA с учётом графа сцены."""

import os, sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

# Путь к клону mycobot_ros: 
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
VENDOR_DIR = os.environ.get(
    "MYCOBOT_ROS_DIR",
    os.path.join(_PROJECT_ROOT, "..", "mycobot_ros"),
)
VEN = os.path.join(VENDOR_DIR, "mycobot_description", "urdf")
DST = sys.argv[1] if len(sys.argv) > 1 else "/tmp/meshes_new"
os.makedirs(DST, exist_ok=True)

JOBS = [(f"{VEN}/mycobot_280_pi/{n}.dae", f"pi_{n}.stl")
        for n in ["G_base", "joint1_pi", "joint2", "joint3", "joint4",
                  "joint5", "joint6", "joint7"]]
JOBS += [(f"{VEN}/parallel_gripper/{n}.dae", f"pg_{n}.stl")
         for n in ["gripper_base", "gripper_left", "gripper_right"]]


def _without_material_bindings(src):
    tree = ET.parse(src)
    root = tree.getroot()
    ns = root.tag.split("}")[0][1:]
    ET.register_namespace("", ns)
    for parent in root.iter():
        for child in list(parent):
            if child.tag.split("}")[-1] == "bind_material":
                parent.remove(child)
    tmp = tempfile.NamedTemporaryFile(suffix=".dae", delete=False)
    tree.write(tmp.name, xml_declaration=True, encoding="utf-8")
    return tmp.name


def load(src):
    scene = trimesh.load(src, force="scene")
    has_geometry = sum(len(getattr(g, "faces", [])) for g in scene.geometry.values())
    if not has_geometry:
        scene = trimesh.load(_without_material_bindings(src), force="scene")
    return scene.to_geometry()


for src, name in JOBS:
    mesh = load(src)
    # joint6.dae и joint7.dae объявляют метры, будучи записаны в миллиметрах:
    # по заголовку получаются звенья в десятки метров. Ловим по физическому
    # размеру — у руки с вылетом 280 мм звено не может быть больше полуметра.
    if mesh.extents.max() > 0.5:
        mesh.apply_scale(0.001)
    mesh.export(os.path.join(DST, name))
    print(f"   {name:20s} габарит {np.round(mesh.extents * 1000, 1)}")

print(f"\nготово, файлы в {DST}")
