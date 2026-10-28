# -*- coding: utf-8 -*-
"""仿真「伪主臂」teleoperator:给 record 管道提供固定的脚本化动作。

record 强制要求 teleop,仿真场景下没有真主臂,就用一个输出固定轨迹的
假 teleop(正弦摆动 + 夹爪开合),让数据有规律、可复现,适配 ACT 冒烟。
"""
import math
from dataclasses import dataclass

import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator

KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


@TeleoperatorConfig.register_subclass("scripted_sim")
@dataclass(kw_only=True)
class ScriptedTeleopConfig(TeleoperatorConfig):
    """脚本化假主臂:输出频率可控的正弦摆动。"""

    amplitude_deg: float = 15.0   # 摆动幅度(度)
    period_s: float = 4.0         # 摆动周期(秒)


class ScriptedTeleop(Teleoperator):
    config_class = ScriptedTeleopConfig
    name = "scripted_sim"

    def __init__(self, config: ScriptedTeleopConfig):
        super().__init__(config)
        self.config = config
        self._t0 = None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @property
    def action_features(self) -> dict:
        return {k: float for k in KEYS}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self) -> None:
        self._t0 = None

    def disconnect(self) -> None:
        pass

    def get_action(self) -> dict:
        import time

        if self._t0 is None:
            self._t0 = time.monotonic()
        t = time.monotonic() - self._t0
        phase = 2 * math.pi * t / self.config.period_s
        amp = self.config.amplitude_deg
        # 前 1 秒保持零位,之后各关节小幅度正弦摆动,夹爪按半周期开合
        ramp = min(1.0, t / 1.0)
        action = {
            "shoulder_pan.pos": ramp * amp * math.sin(phase),
            "shoulder_lift.pos": ramp * amp * 0.5 * math.sin(phase + 0.5),
            "elbow_flex.pos": ramp * amp * 0.7 * math.sin(phase + 1.0),
            "wrist_flex.pos": ramp * amp * 0.3 * math.sin(phase + 1.5),
            "wrist_roll.pos": ramp * amp * 0.4 * math.sin(phase + 2.0),
            "gripper.pos": 50.0 + 50.0 * math.sin(phase / 2),  # 0~100 开合
        }
        return action

    def send_feedback(self, feedback: dict) -> None:
        pass
