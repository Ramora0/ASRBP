from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DownsamplerConfig:
    """Spectrogram → transformer-input bridge module (see
    ``conformer_asr/downsamplers/``).

    ``type`` selects an entry in ``downsamplers.DOWNSAMPLERS``; ``kwargs`` is
    passed through to that implementation's constructor, so adding a new
    downsampler doesn't require touching this schema. The default ``conv2d``
    stem takes ``strides``: a list of ``[time, mel]`` pairs, one per Conv2d
    layer. Kernel is uniformly ``(3, 3)``; padding is 1 on axes with
    stride 1 (preserves dim, Whisper-style) and 0 elsewhere (standard
    no-pad subsampling for stride-2 layers).
    """

    type: str = "conv2d"
    kwargs: dict[str, Any] = field(default_factory=lambda: {"strides": [[2, 2], [2, 2]]})


@dataclass
class ModelConfig:
    # Architecture selectors. See ``conformer_asr/encoders/`` and
    # ``conformer_asr/decoders/`` for the registered families.
    encoder_type: str
    decoder_type: str
    encoder_hidden_size: int
    encoder_num_hidden_layers: int
    encoder_num_attention_heads: int
    encoder_intermediate_size: int
    encoder_conv_depthwise_kernel_size: int
    encoder_hidden_dropout: float
    encoder_attention_dropout: float
    encoder_activation_dropout: float
    # Stochastic depth: uniform probability of skipping a whole encoder layer
    # during training. HF implements this as ``Wav2Vec2ConformerConfig.layerdrop``
    # (drop-the-layer form, not the paper's linear-per-depth schedule, but close
    # enough in practice). ~0.1 is standard for Conformer.
    encoder_layerdrop: float
    # Log-Mel frontend (see conformer_asr/features.py). Mel features are
    # computed offline at preprocess time; the encoder's downsampler stem
    # consumes them directly — no waveform feature encoder.
    n_mels: int
    n_fft: int
    hop_length: int
    decoder_d_model: int
    decoder_layers: int
    decoder_attention_heads: int
    decoder_ffn_dim: int
    decoder_dropout: float
    decoder_max_position_embeddings: int
    # Hybrid CTC/AED auxiliary loss. When enabled, a linear CTC head is added
    # on top of the encoder and the total training loss becomes
    # ``(1 - ctc_weight) * AED + ctc_weight * CTC``. CTC imposes monotonic
    # alignment in the encoder — the single biggest convergence lever for
    # from-scratch AED — and also powers a cheap greedy-decode WER readout at
    # eval time (``CTCEvalCallback`` in ``wandb_utils``).
    ctc_enabled: bool
    ctc_weight: float
    # Per-bin log-Mel normalization — running mean/var updated during training
    # until ``input_normalize_freeze_epochs`` have elapsed, then frozen for
    # the rest of training and all downstream eval. Mirrors SB's
    # ``InputNormalization(norm_type='global', update_until_epoch=N)``. The
    # running_mean / running_var buffers are saved in the state dict so
    # ``scripts/evaluate.py`` inherits the frozen stats via load_state_dict.
    input_normalize_freeze_epochs: int
    # SpecAugment applied **pre-stem** on (B, T_mel, n_mels), at the 100 Hz
    # frame rate. Deterministic K masks per sample, lengths sampled uniformly
    # from [low, high]. Zero-fill (after InputNormalization, zero == per-bin
    # mean). Faithful to Park et al. 2019 and SB's LibriSpeech recipe.
    spec_aug_time_masks: int
    spec_aug_time_length_low: int
    spec_aug_time_length_high: int
    spec_aug_feature_masks: int
    spec_aug_feature_length_low: int
    spec_aug_feature_length_high: int
    # Steps of pure supervised training before SpecAugment turns on. 0 = always
    # on. SB conformer_large uses 8000 to let the model learn a crude alignment
    # first; optional on short schedules.
    spec_aug_warmup_steps: int
    # Spectrogram → transformer-input bridge. Nested so the downsampler family
    # can carry its own knobs without bloating the top-level ``ModelConfig``.
    downsampler: DownsamplerConfig = field(default_factory=DownsamplerConfig)


@dataclass
class DataConfig:
    dataset_id: str
    subset: str
    eval_split: str
    test_split: str
    sampling_rate: int
    max_audio_seconds: float
    num_proc: int
    # Kaldi-style 3-way speed perturbation (Ko et al. 2015). Each training clip
    # is replicated at every listed speed via rational-factor resampling — 0.9
    # stretches the waveform by 10/9 (slower, lower-pitched), 1.1 by 10/11
    # (faster, higher-pitched). Applied only to the train split at preprocess
    # time and folded into the cache key, so the on-disk cache already contains
    # all perturbed copies. Default [1.0] is a no-op (no perturbation).
    speed_perturbations: list[float] = field(default_factory=lambda: [1.0])
    # Tokenizer: null means "download SB's pretrained SentencePiece from HF Hub
    # into ``cache_dir`` (repo: speechbrain/asr-transformer-transformerlm-librispeech)."
    # A local path is only consulted if it contains ``sentencepiece.model``
    # (e.g. a prior training run's ``final/`` dir) and otherwise falls through
    # to the Hub download.
    tokenizer_dir: str | None = None
    cache_dir: str = ""


@dataclass
class TrainConfig:
    output_dir: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    # Target effective batch size = per_device_train_batch_size * world_size *
    # gradient_accumulation_steps. With ``gradient_accumulation_steps: 0`` in
    # the YAML (or unset on the CLI), ``resolve_grad_accum`` computes
    # grad_accum from this and the runtime ``WORLD_SIZE`` so the same config
    # gives the same effective batch on 1 / 2 / 4 / 8 GPUs without manual
    # editing. Set ``gradient_accumulation_steps`` to a positive integer to
    # override and trust whatever effective batch falls out.
    effective_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    num_train_epochs: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_grad_norm: float
    label_smoothing_factor: float
    lr_scheduler_type: str
    logging_steps: int
    save_total_limit: int
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool
    group_by_length: bool
    dataloader_num_workers: int
    # Per-worker batch buffer ahead of the GPU. PyTorch default is 2 (so 8
    # workers × 2 = 16 batches in flight). On networked storage (GPFS / Lustre)
    # the read latency variance dominates step time, and 16-batch buffer is
    # not enough — workers stall on coordinated metadata reads and the GPU
    # drops to 0% util for 1-2 s windows. 4 doubles the buffer; bump higher
    # if you still see step-time spikes after staging the cache to local NVMe.
    dataloader_prefetch_factor: int
    report_to: str
    seed: int
    generation_max_length: int
    generation_num_beams: int
    # Stochastic Weight Averaging. When enabled, maintains a running mean of
    # weights from ``swa_start_frac * num_train_epochs`` onward (per-epoch sampling),
    # and at end of training writes the averaged weights to
    # ``<output_dir>/final-swa/pytorch_model.bin``. Evaluate separately via
    # ``scripts/evaluate.py --checkpoint <output_dir>/final-swa``. Adds one
    # extra full-model copy on GPU (~200 MB for Conformer-S — rounding error on
    # 32 GB V100). Conformer uses LayerNorm, so no post-hoc BN update pass is
    # needed.
    swa_enabled: bool
    swa_start_frac: float


@dataclass
class WandbConfig:
    enabled: bool
    project: str
    entity: str | None
    run_name: str | None
    group: str | None
    tags: list[str]
    notes: str
    watch_model: bool
    log_preds_table: bool
    log_preds_n: int


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    wandb: WandbConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_model_config(raw: dict[str, Any]) -> ModelConfig:
    raw = dict(raw)
    ds_raw = raw.get("downsampler")
    if isinstance(ds_raw, dict):
        raw["downsampler"] = DownsamplerConfig(**ds_raw)
    elif ds_raw is None:
        raw["downsampler"] = DownsamplerConfig()
    return ModelConfig(**raw)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load config from YAML. YAML is the single source of truth for values;
    dataclasses only define the schema. Every field must be present in YAML —
    a missing key will raise a TypeError at dataclass construction.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = Config(
        model=_build_model_config(raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        train=TrainConfig(**raw.get("train", {})),
        wandb=WandbConfig(**raw.get("wandb", {})),
    )
    if overrides:
        # Flat overrides are applied to whichever section owns the key.
        # Nested sub-configs (e.g. ``model.downsampler``) aren't reachable from
        # the flat CLI surface — edit the YAML for those.
        for key, val in overrides.items():
            if val is None:
                continue
            for section in (cfg.model, cfg.data, cfg.train, cfg.wandb):
                if key in {f.name for f in fields(section)}:
                    setattr(section, key, val)
                    break
    return cfg


def resolve_grad_accum(train_cfg: TrainConfig, world_size: int) -> None:
    """Derive ``gradient_accumulation_steps`` from ``effective_batch_size``.

    If ``train_cfg.gradient_accumulation_steps`` is 0, computes
    ``effective_batch_size // (per_device_train_batch_size * world_size)``
    and writes it back. If it's a positive integer, leaves it alone (manual
    override). Raises if the auto-compute doesn't divide evenly or yields a
    value < 1, since silently rounding either way would change effective
    batch behind the user's back.
    """
    if train_cfg.gradient_accumulation_steps > 0:
        return
    if train_cfg.gradient_accumulation_steps < 0:
        raise ValueError(
            f"gradient_accumulation_steps must be >= 0, got {train_cfg.gradient_accumulation_steps}"
        )
    ws = max(1, world_size)
    per_step = train_cfg.per_device_train_batch_size * ws
    if per_step <= 0:
        raise ValueError(
            f"per_device_train_batch_size * world_size = {per_step} (must be > 0)"
        )
    eff = train_cfg.effective_batch_size
    if eff <= 0:
        raise ValueError(f"effective_batch_size must be > 0, got {eff}")
    if eff % per_step != 0:
        raise ValueError(
            f"effective_batch_size={eff} is not divisible by "
            f"per_device_train_batch_size * world_size = "
            f"{train_cfg.per_device_train_batch_size} * {ws} = {per_step}. "
            f"Adjust per_device_train_batch_size or effective_batch_size, "
            f"or set gradient_accumulation_steps explicitly."
        )
    grad_accum = eff // per_step
    if grad_accum < 1:
        raise ValueError(
            f"Derived gradient_accumulation_steps={grad_accum} < 1: "
            f"per_device_train_batch_size * world_size ({per_step}) "
            f"already exceeds effective_batch_size ({eff})."
        )
    train_cfg.gradient_accumulation_steps = grad_accum


def resolve_precision(train_cfg: TrainConfig) -> None:
    """Flip bf16→fp16 on GPUs that don't support bfloat16 natively (e.g. V100 / Volta).

    Mutates ``train_cfg`` in place. Intended to run after CLI overrides but
    before TrainingArguments / autocast dtype is built.

    We gate on ``compute capability >= (8, 0)`` (Ampere+) rather than
    ``torch.cuda.is_bf16_supported()`` because newer PyTorch versions report
    True for V100 on the basis of *emulated* bf16 — which silently falls off
    the fp16 tensor-core path and runs at a fraction of native throughput.
    """
    import torch

    if not torch.cuda.is_available():
        return
    if train_cfg.bf16 and torch.cuda.get_device_capability(0) < (8, 0):
        device_name = torch.cuda.get_device_name(0)
        print(f"[precision] bf16 not natively supported on {device_name}; using fp16.")
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
