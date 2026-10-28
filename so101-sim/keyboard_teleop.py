# -*- coding: utf-8 -*-
"""SO-ARM100 键盘遥操作:MuJoCo 交互窗口 + 键盘控制各关节。

用法 (WSL):
    source /mnt/g/SO-ARM100/so101-sim/.venv-wsl/bin/activate
    python /mnt/g/SO-ARM100/so101-sim/keyboard_teleop.py

按键 (按住持续转动, 松开停止):
    q/a : 关节1 Rotation(底座旋转)   +/-
    w/s : 关节2 Pitch(肩)            +/-
    e/d : 关节3 Elbow(肘)            +/-
    r/f : 关节4 Wrist_Pitch(腕俯仰)  +/-
    t/g : 关节5 Wrist_Roll(腕旋转)   +/-
    y/h : 关节6 Jaw(夹爪) 开/合
    空格: 回到初始姿态
    ESC : 退出
"""
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "/mnt/g/mujoco_menagerie/trs_so_arm100/scene.xml"
SPEED = 0.8  # rad/s

HELD = set()

# keycode -> (joint_index, direction)
KEYMAP = {
    ord("Q"): (0, +1), ord("A"): (0, -1),
    ord("W"): (1, +1), ord("S"): (1, -1),
    ord("E"): (2, +1), ord("D"): (2, -1),
    ord("R"): (3, +1), ord("F"): (3, -1),
    ord("T"): (4, +1), ord("G"): (4, -1),
    ord("Y"): (5, +1), ord("H"): (5, -1),
}

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 执行器对应的关节索引
ACT_JOINT = [model.actuator_trnid[i, 0] for i in range(model.nu)]
HOME = np.array([0.0, -1.5, 1.5, -1.0, 0.0, 0.0])  # 抬臂收爪的初始姿态

def set_targets(q):
    for i, j in enumerate(ACT_JOINT):
        lo, hi = model.jnt_range[j]
        data.ctrl[i] = np.clip(q[i], lo, hi)

def key_callback(keycode):
    if keycode == 256:  # ESC
        HELD.add(("quit",))
    elif keycode == 32:  # 空格回位
        set_targets(HOME)
    elif keycode in KEYMAP:
        HELD.add(KEYMAP[keycode])

data.qpos[ACT_JOINT] = HOME
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as v:
    v.sync()
    import time

    while v.is_running() and ("quit",) not in HELD:
        for joint, direction in list(HELD):
            if not isinstance(joint, int):
                continue
            lo, hi = model.jnt_range[joint]
            data.qpos[ACT_JOINT[joint]] = float(
                np.clip(data.qpos[ACT_JOINT[joint]] + direction * SPEED * 0.004, lo, hi)
            )
        # 位置伺服:ctrl 跟随当前 qpos(简单直接,视觉上即是键盘拖动关节)
        for i, j in enumerate(ACT_JOINT):
            data.ctrl[i] = data.qpos[ACT_JOINT[j]]
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.002)

print("退出")
