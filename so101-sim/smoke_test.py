# -*- coding: utf-8 -*-
"""SO-ARM100/101 MuJoCo 仿真冒烟测试：加载模型 -> 随机走几步 -> 离屏渲染一帧。"""
import sys
from pathlib import Path

import mujoco
import numpy as np

MODEL = Path(__file__).parent.parent / "mujoco_menagerie" / "trs_so_arm100" / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)

print(f"模型加载成功: {MODEL}")
print(f"  关节数: {model.njnt}, 执行器数: {model.nu}, 时间步: {model.opt.timestep}s")

for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    print(f"  joint[{j}]: {name}")

# 模拟 3 秒，让臂从初始位置自然沉降
for _ in range(int(3.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)

qpos = np.array(data.qpos)
print(f"仿真 3s 后 qpos: {np.round(qpos, 3)}")
assert np.all(np.isfinite(qpos)), "qpos 出现 NaN，物理不稳定"

# 离屏渲染一帧，验证渲染管线
renderer = mujoco.Renderer(model, height=480, width=640)
mujoco.mj_forward(model, data)
renderer.update_scene(data)
img = renderer.render()
out = Path(__file__).parent / "sim_frame.png"
from PIL import Image  # noqa: E402  (mujoco 依赖已含 pillow? 若缺失再装)

Image.fromarray(img).save(out)
print(f"渲染一帧已保存: {out} ({img.shape})")
print("OK: SO-ARM100 仿真跑通")
