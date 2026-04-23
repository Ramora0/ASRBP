# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Trains a Conformer attention encoder-decoder (AED) ASR model from scratch on LibriSpeech. Default architecture: a randomly-initialized `Wav2Vec2ConformerEncoder` with a log-Mel + Conv2d-subsampling frontend, paired with `BartForCausalLM` as the cross-attention decoder, glued via `SpeechEncoderDecoderModel`. Hybrid CTC/AED loss by default (`ctc_enabled: true` in the YAML; total = `(1 - ctc_weight) * AED + ctc_weight * CTC`). Targets a single 32 GB GPU with bf16.

## Common commands

```bash
uv pip install -e .

# Smoke test the whole pipeline without downloads (use before any full run on a new env)
python scripts/smoke_test.py

# Preprocess the dataset once (required before multi-GPU training to avoid racing on save_to_disk)
python scripts/preprocess.py --config configs/conformer_small.yaml

# Train (defaults = full 960h, --subset clean100 for fast iteration)
python scripts/train.py --output_dir outputs/run
python scripts/train.py --subset clean100 --no_wandb --output_dir outputs/smoke

# Evaluate a saved checkpoint — appends to results/wer.json
python scripts/evaluate.py --checkpoint outputs/run/final --split test.clean --num_beams 5
```

There is no separate lint or test suite; `scripts/smoke_test.py` is the closest thing to an integration test.

## Architecture

Library code lives in `conformer_asr/` (installable package); user-facing entrypoints live in `scripts/`. Scripts insert the repo root on `sys.path` so they work without installation.

**Swappable architectures.** The model is built from three independently-pluggable families, each registered in its own sub-package:

- `conformer_asr/downsamplers/` — spectrogram → transformer-input bridges. Default `Conv2dDownsampler` (`type: conv2d`). Must implement `forward(x: (B, T_mel, n_mels)) -> (B, T', hidden)` and `output_lengths(input_lengths) -> output_lengths`; the latter is pure time arithmetic so the encoder can build the post-stem attention mask without a forward pass. For `conv2d`, `num_convs=2` is the standard 4× time stem; extras (`>=3`) are kernel `(3, 1)` stride `(2, 1)` — time-only, mel axis untouched — so `num_convs=3` → 8×, `num_convs=4` → 16×.
- `conformer_asr/encoders/` — full speech encoders. Default `conformer` → `MelConformerEncoder` (Wav2Vec2ConformerEncoder + a pluggable downsampler + pre-stem `InputNormalization` and `SpecAugment` from `encoders/preproc.py`). Must expose `.config.hidden_size`, return a `BaseModelOutput` from `forward`, and provide `_get_feature_vector_attention_mask` for the decoder-side mask — `ConformerAEDWithCTC` and `scripts/evaluate.py` both call it by name. `InputNormalization` maintains running per-bin mean/var until `FreezeInputNormCallback` flips its `frozen` flag at epoch `input_normalize_freeze_epochs`; the stats ride along in the state_dict so eval inherits them. `SpecAugment` is deterministic (fixed mask count, uniform-length, zero-fill after normalization = per-bin mean) and applied at 100 Hz pre-stem rate; `SpecAugWarmupCallback` can gate it off for the first `spec_aug_warmup_steps`.
- `conformer_asr/decoders/` — autoregressive decoders with cross-attention. Default `bart` → `_CompatBartForCausalLM` (works around `SpeechEncoderDecoderModel` + `BartForCausalLM` double-embedding bug).

`build_model` in `conformer_asr/model.py` wires all three via their registries and wraps the result in `SpeechEncoderDecoderModel` (or `ConformerAEDWithCTC` when `ctc_enabled`).

### Adding a new architecture

1. **New downsampler**: subclass `Downsampler` from `conformer_asr/downsamplers/base.py` (implement `forward` + `output_lengths`), add an entry to `DOWNSAMPLERS` in `conformer_asr/downsamplers/__init__.py`, reference it in the YAML via `model.downsampler.type` with any constructor-specific params under `model.downsampler.kwargs`. `build_downsampler` splats `cfg.kwargs` into the constructor alongside the shared `n_mels` / `hidden` / `dropout`, so the registry needs no changes when you add a new knob.
2. **New encoder**: add a builder in `conformer_asr/encoders/`, register it in `ENCODERS` (`encoders/__init__.py`), set `model.encoder_type` in the YAML. The builder owns its config-to-module translation — nothing forces it to use the same `ModelConfig.encoder_*` fields the Conformer builder uses.
3. **New decoder**: same pattern — add a builder in `conformer_asr/decoders/`, register in `DECODERS`, set `model.decoder_type` in the YAML. Builder signature is `(mcfg, *, vocab_size, pad_id, bos_id, eos_id) -> nn.Module`.

The SDPA fast-path patch for `Wav2Vec2ConformerSelfAttention` lives in `conformer_asr/encoders/sdpa_patch.py` and is installed idempotently on first import of the Conformer encoder.

### Config flow

All configuration is a `Config` dataclass tree (`ModelConfig`, `DataConfig`, `TrainConfig`, `WandbConfig`) in `conformer_asr/config.py`. `ModelConfig.downsampler` is a nested `DownsamplerConfig` (`type: str`, `kwargs: dict`). `load_config(path, overrides)` loads YAML then applies flat CLI overrides by looking up which section owns each key — so new CLI flags in scripts work as long as the key name matches a top-level dataclass field. Nested sub-configs (e.g. `model.downsampler`) aren't reachable from the flat CLI surface; edit the YAML for those. Defaults in `configs/conformer_small.yaml` must stay in sync with dataclass defaults.

### Tokenizer

SpeechBrain's pretrained SentencePiece (5K unigram) over whitespace-normalized, case-preserved transcripts (`tokenizer.py`). Fetched on first run from HF Hub (`speechbrain/asr-transformer-transformerlm-librispeech`, file `tokenizer.ckpt`) into `cache_dir`; the `SpeechBrainTokenizer` adapter exposes the subset of the HF `PreTrainedTokenizerFast` API the codebase uses. Special tokens: `<pad>=0, <s>=1, </s>=2, <unk>=3`. `__call__(text)` appends BOS+EOS so the collator's leading-BOS strip stays correct. Vocab-aligned with `sb_lm.py`'s TransformerLM, which enables shallow fusion / decoder warm-start. The data collator in `data.py` **strips the leading BOS from labels** because `SpeechEncoderDecoderModel.shift_tokens_right` re-adds `decoder_start_token_id`; doubling the BOS would break training. `normalize_text()` is applied both at tokenize time and at eval time for WER references — it whitespace-normalizes but **does not change case** (LibriSpeech + SB's SentencePiece are both natively uppercase); do not introduce a different normalization.

### Data pipeline

`load_librispeech` picks one of three fixed subsets (`clean100` / `clean460` / `all960`) and concatenates parts into a train split; validation/test splits come from `cfg.data.eval_split` / `test_split`. Preprocessing computes log-Mel features offline (see `features.py`) and tokenizes labels. Only the **training** split is filtered by duration bounds so evaluation WER is over the full split. `input_length` is preserved for HF's `group_by_length` bucketing. With `speed_perturbations` set to more than one value, the training split is materialized as N contiguous variant rows per clip; `RandomSpeedVariantSampler` in `data.py` + `SpeedAugSeq2SeqTrainer` / `HybridSeq2SeqTrainer` in `scripts/train.py` subsample to one variant per clip per epoch so epoch length stays at N, not `n * N`.

### Cache discipline (important on cluster nodes)

Every entrypoint calls `bootstrap_cache_from_argv()` (the tiny module `bootstrap_cache.py` at repo root) *before* importing any `datasets` / `transformers` / `conformer_asr.*` code, and then `setup_cache_dir(cfg.data.cache_dir)` as a belt-and-braces. This sets `HF_HOME`, `HF_DATASETS_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE` so arrow shards, audio files, and transformers hub artifacts all land on scratch instead of `$HOME`. If you add a new script, import `bootstrap_cache` first — otherwise `datasets` / `transformers` freeze the wrong paths at import time.

### Trainer + CTC

`scripts/train.py` uses `Seq2SeqTrainer` with `predict_with_generate=True` and an inverse-sqrt LR schedule. When `ctc_enabled`, `HybridSeq2SeqTrainer` keeps `labels` in the forward pass (the stock trainer would pop them when a label smoother is active, starving CTC of targets), applies label smoothing to the AED branch only, then re-blends with the raw CTC loss using `model.ctc_weight`. The CTC head reuses `<pad>` as the blank id. `CTCEvalCallback` in `wandb_utils.py` does a cheap greedy-decode WER readout from the CTC logits each eval; `scripts/evaluate.py --ctc_weight > 0` uses the same head for n-best rescoring.

### W&B + evaluation

`init_wandb` (in `wandb_utils.py`) is called *before* the Trainer is constructed, so HF's built-in `WandbCallback` reuses the existing `wandb.run` instead of creating its own. `epoch` is registered as a first-class metric with `train/global_step` as the x-axis. `--no_wandb` also strips `"wandb"` from `report_to` so HF's callback doesn't re-enable it. `scripts/evaluate.py` runs manual beam search via `model.generate` (bf16 autocast on CUDA), computes WER with `evaluate.load("wer")`, **appends** to `results/wer.json`, and force-enables wandb (unless `--no_wandb`) with an `eval:<split>` tag. Evaluation supports n-best rescoring via a SpeechBrain LM (`--lm_weight`) and/or the model's own CTC head (`--ctc_weight`).

## Conventions worth preserving

- The leading-BOS strip in `DataCollatorSpeechSeq2SeqWithPadding` is load-bearing — don't remove the `labels[:, 0] == decoder_start_token_id` check without also removing the tokenizer's `TemplateProcessing` post-processor.
- Only the train split is length-filtered; keep it that way so evaluation WER is over the full split.
- `setup_cache_dir` uses `os.environ.setdefault` — if env vars are already set (e.g. by a SLURM script), those win. Do not switch to hard assignment.
- `results/wer.json` accumulates; do not change it to overwrite.
- `build_model` rebuilds the full architecture at eval time (not `SpeechEncoderDecoderModel.from_pretrained`) because the custom encoder isn't a registered HF model — `from_pretrained` would silently substitute the stock raw-waveform Wav2Vec2Conformer and then try to convolve a waveform kernel over log-Mel features.
- The downsampler's `output_lengths` must stay consistent with its actual `forward` time arithmetic — the encoder's attention mask and CTC's `input_lengths` both rely on it, and a mismatch silently produces wrong masks rather than a clean error.
