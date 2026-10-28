# -*- coding: utf-8 -*-
"""ACT 训练冒烟:用刚录的 sim_picklift_smoke 数据集,小步数验证训练管道 + 显存水位。

GTX 1050 Ti 只有 4G 显存,ACT 默认 chunk_size=100 太大,冒烟配置:
- chunk_size 20 / batch_size 4 / 30 步 / 优化器步数同步缩小
- 每 10 步打印 loss 和显存占用

用法: source /mnt/g/SO-ARM100/so101-sim/.venv-wsl/bin/activate && python /mnt/g/SO-ARM100/so101-sim/train_act_smoke.py
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.scripts.lerobot_train import train

DATASET = "/mnt/g/SO-ARM100/so101-sim/datasets"

OUTPUT = Path("/mnt/g/SO-ARM100/so101-sim/act_smoke_out")

ds_meta = LeRobotDataset(DATASET).meta

# 让 train() 自己从 ds_meta 推断 features(make_policy 内部会做 dataset_to_policy_features),
# 这里只设与 features 无关的冒烟超参
act_cfg = ACTConfig(
    chunk_size=20,
    n_action_steps=10,
    device="cuda",
    push_to_hub=False,
)

train_cfg = TrainPipelineConfig(
    dataset=DatasetConfig(repo_id=DATASET),
    output_dir=OUTPUT,
    job_name="act_smoke",
    policy=act_cfg,
    batch_size=4,
    steps=30,
    save_freq=30,
    log_freq=10,
    num_workers=0,
    save_checkpoint=True,
    save_checkpoint_to_hub=False,
)

print(f"device: {train_cfg.policy.device}")
print(f"数据集: {DATASET} ({ds_meta.total_episodes} eps)")
print("开始训练冒烟...")

train_cfg.validate()
try:
    train(train_cfg)
finally:
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"训练结束 | 显存峰值: {peak:.2f} GB / 4.29 GB")
