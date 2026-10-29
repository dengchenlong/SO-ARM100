# -*- coding: utf-8 -*-
"""ACT 正式训练：datasets_expert（25 eps / 7517 frames / 640x480 前置相机）。

配置：chunk_size 50 / n_action_steps 50 / batch 8 / 15k steps（4GB 显存档位）。
用法：
    wsl -d Debian -e bash -lc "cd /mnt/g/SO-ARM100/so101-sim && \
        ./.venv-wsl/bin/python -u train_act_expert.py 2>&1 | tee train_act.log"
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act import ACTConfig
from lerobot.scripts.lerobot_train import train

DATASET = "/mnt/g/SO-ARM100/so101-sim/datasets_expert"
OUTPUT = Path("/mnt/g/SO-ARM100/so101-sim/act_expert_out")
STEPS = 15000

ds_meta = LeRobotDataset(DATASET).meta
print(f"dataset: {DATASET} | {ds_meta.total_episodes} eps / {ds_meta.total_frames} frames")

act_cfg = ACTConfig(
    chunk_size=50,
    n_action_steps=50,
    device="cuda",
    push_to_hub=False,
)

train_cfg = TrainPipelineConfig(
    dataset=DatasetConfig(repo_id=DATASET),
    output_dir=OUTPUT,
    job_name="act_expert",
    policy=act_cfg,
    batch_size=8,
    steps=STEPS,
    save_freq=2500,
    log_freq=100,
    num_workers=0,
    save_checkpoint=True,
    save_checkpoint_to_hub=False,
)

print(f"device={act_cfg.device} chunk={act_cfg.chunk_size} batch={train_cfg.batch_size} steps={STEPS}")
train_cfg.validate()
try:
    train(train_cfg)
finally:
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"TRAIN_END | peak_vram {peak:.2f} GB / 4.29 GB")
