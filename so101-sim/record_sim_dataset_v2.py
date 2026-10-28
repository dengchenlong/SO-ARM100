# -*- coding: utf-8 -*-
"""用 lerobot record 管道把 so101-nexus 仿真数据录成 LeRobotDataset。

SimSOFollower + SimCameraConfig 是 so101-nexus 官方给 lerobot 的适配层:
robot 包成标准 lerobot Robot,teleop 不传(None)时用固定动作驱动。

用法: source /mnt/g/SO-ARM100/so101-sim/.venv-wsl/bin/activate && python /mnt/g/SO-ARM100/so101-sim/record_sim_dataset_v2.py
"""
import warnings
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, "/mnt/g/SO-ARM100/so101-sim")

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.scripts.lerobot_record import RecordConfig, record
from so101_nexus.lerobot_adapter.sim_camera_config import SimCameraConfig
from so101_nexus.lerobot_adapter.sim_follower import SimSOFollowerConfig

from scripted_teleop import ScriptedTeleopConfig

REPO_ID = "local/sim_picklift_smoke"
NUM_EPISODES = 3
FPS = 30
EP_SECONDS = 8
ROOT = "/mnt/g/SO-ARM100/so101-sim/datasets"

robot_cfg = SimSOFollowerConfig(
    id="sim_follower",
    env_id="MuJoCoPickLift-v1",
    seed=0,
    env_kwargs={"render_mode": "rgb_array"},
    cameras={"laptop": SimCameraConfig(source="render", width=640, height=480, fps=FPS)},
)

record_cfg = RecordConfig(
    robot=robot_cfg,
    dataset=DatasetRecordConfig(
        repo_id=REPO_ID,
        single_task="PickLift-smoke",
        root=ROOT,
        fps=FPS,
        episode_time_s=EP_SECONDS,
        reset_time_s=1,
        num_episodes=NUM_EPISODES,
        video=True,
        push_to_hub=False,
    ),
    teleop=ScriptedTeleopConfig(amplitude_deg=15.0, period_s=4.0),
    play_sounds=False,
)

ds = record(record_cfg)
print(f"录制完成: {ds.num_episodes} episodes, {ds.num_frames} frames -> {ROOT}/{REPO_ID}")
