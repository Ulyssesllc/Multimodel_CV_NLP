"""Central configuration for Amazon multimodal workflow."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 8
    epochs: int = 50
    patience: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0  # 0 or None to disable
    scheduler: str = "cosine"  # cosine | plateau | none
    mixed_precision: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    backbone: str = "resnet18"  # resnet18 | resnet34
    freeze_backbone: bool = True  # freeze CNN early layers
    freeze_text: bool = False  # set True to freeze BERT encoder
    dropout: float = 0.3
    resume_checkpoint: Optional[str] = None


CONFIG = TrainConfig()
