"""Configuration for GLAMI-1M multimodal training workflow."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GlamiConfig:
    seed: int = 42
    batch_size: int = 256
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 0.01
    patience: int = 5  # early stopping on val (test) accuracy
    scheduler: str = "cosine"  # cosine | plateau | none
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    log_dir: str = "glami_logs"
    checkpoint_dir: str = "checkpoints"
    resume: Optional[str] = None  # path to checkpoint to resume


CONFIG = GlamiConfig()
