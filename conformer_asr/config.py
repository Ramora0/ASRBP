from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    encoder_hidden_size: int = 256
    encoder_num_hidden_layers: int = 16  # Conformer-M (Gulati et al. 2020)
    encoder_num_attention_heads: int = 4
    encoder_intermediate_size: int = 1024
    encoder_conv_depthwise_kernel_size: int = 31
    encoder_mask_time_prob: float = 0.05
    encoder_mask_feature_prob: float = 0.05
    decoder_d_model: int = 256
    decoder_layers: int = 4
    decoder_attention_heads: int = 4
    decoder_ffn_dim: int = 1024
    decoder_dropout: float = 0.1
    decoder_max_position_embeddings: int = 448


@dataclass
class DataConfig:
    dataset_id: str = "openslr/librispeech_asr"
    subset: str = "all960"  # "clean100" | "clean460" | "all960"
    eval_split: str = "validation.clean"
    test_split: str = "test.clean"
    sampling_rate: int = 16000
    max_audio_seconds: float = 20.0
    num_proc: int = 4
    tokenizer_dir: str = "tokenizer"
    # Everything downloaded / cached (audio, arrow shards, transformers hub, models)
    # lands under this directory. Point this at scratch so $HOME doesn't fill up.
    cache_dir: str = "/fs/scratch/PAS2836/lees_stuff/hf_cache"


@dataclass
class TrainConfig:
    output_dir: str = "outputs/run"
    per_device_train_batch_size: int = 128
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    warmup_ratio: float = 0.06
    num_train_epochs: float = 20.0
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    label_smoothing_factor: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 20
    save_total_limit: int = 3
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    group_by_length: bool = True
    dataloader_num_workers: int = 8
    report_to: str = "wandb"  # comma-separated: "wandb", "tensorboard", or "wandb,tensorboard"
    seed: int = 42
    generation_max_length: int = 300
    generation_num_beams: int = 1  # greedy during training eval; beam search in evaluate.py


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "conformer-asr-librispeech"
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    watch_model: bool = False  # turn on to log gradients/parameters (bandwidth-heavy)
    log_preds_table: bool = True  # log a sample predictions table at eval time
    log_preds_n: int = 32


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_from_dict(dc: Any, values: dict[str, Any]) -> None:
    allowed = {f.name for f in fields(dc)}
    for key, val in values.items():
        if key in allowed:
            setattr(dc, key, val)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config()
    if path is not None:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        _update_from_dict(cfg.model, raw.get("model", {}))
        _update_from_dict(cfg.data, raw.get("data", {}))
        _update_from_dict(cfg.train, raw.get("train", {}))
        _update_from_dict(cfg.wandb, raw.get("wandb", {}))
    if overrides:
        # Flat overrides are applied to whichever section owns the key.
        for key, val in overrides.items():
            if val is None:
                continue
            for section in (cfg.model, cfg.data, cfg.train, cfg.wandb):
                if key in {f.name for f in fields(section)}:
                    setattr(section, key, val)
                    break
    return cfg


def resolve_precision(train_cfg: TrainConfig) -> None:
    """Flip bf16→fp16 on GPUs that don't support bfloat16 (e.g. V100 / Volta).

    Mutates ``train_cfg`` in place. Intended to run after CLI overrides but
    before TrainingArguments / autocast dtype is built.
    """
    import torch

    if not torch.cuda.is_available():
        return
    if train_cfg.bf16 and not torch.cuda.is_bf16_supported():
        device_name = torch.cuda.get_device_name(0)
        print(f"[precision] bf16 not supported on {device_name}; using fp16.")
        train_cfg.bf16 = False
        train_cfg.fp16 = True


def autocast_dtype(train_cfg: TrainConfig) -> "torch.dtype":
    """Pick the autocast dtype implied by ``train_cfg`` (for manual autocast sites)."""
    import torch

    if train_cfg.bf16:
        return torch.bfloat16
    if train_cfg.fp16:
        return torch.float16
    return torch.float32
