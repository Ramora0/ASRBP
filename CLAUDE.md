# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Trains a Conformer attention encoder-decoder (AED) ASR model from scratch on LibriSpeech. Encoder is a randomly-initialized `Wav2Vec2ConformerModel`; decoder is `BartForCausalLM` with cross-attention; glued via `SpeechEncoderDecoderModel`. Pure AED loss (cross-entropy with label smoothing, no CTC branch). Targets a single 32 GB GPU with bf16.

## Architecture

Library code lives in `conformer_asr/` (installable package); user-facing entrypoints live in `scripts/`. Scripts insert the repo root on `sys.path` so they work without installation.

**Config flow.** All configuration is a `Config` dataclass tree (`ModelConfig`, `DataConfig`, `TrainConfig`, `WandbConfig`) in `conformer_asr/config.py`. `load_config(path, overrides)` loads YAML then applies flat CLI overrides by looking up which section owns each key — so new CLI flags in scripts work as long as the key name matches a dataclass field. Defaults in the YAML (`configs/conformer_small.yaml`) must stay in sync with dataclass defaults.

**Model construction.** `build_model` (in `model.py`) wires together `Wav2Vec2ConformerConfig` (relative position embeddings, SpecAugment on) and a `BartConfig` used as a standalone `BartForCausalLM`. Because `BartForCausalLM` still reads `encoder_layers/heads/ffn_dim` internally, those are set to mirror the decoder shape. After construction, pad/bos/eos and `decoder_start_token_id` are copied to `model.config` so `generate()` picks them up.

**Tokenizer.** Byte-level BPE over lowercased, whitespace-normalized transcripts (`tokenizer.py`). Special tokens are `<pad> <s> </s> <unk>`. The post-processor auto-prepends `<s>` and appends `</s>` to any tokenized string. The data collator in `data.py` **strips the leading BOS from labels** because `SpeechEncoderDecoderModel.shift_tokens_right` re-adds `decoder_start_token_id`; doubling the BOS would break training. `normalize_text()` is applied both at tokenizer training time and at eval time for WER references — do not introduce a different normalization.

**Data pipeline.** `load_librispeech` picks one of three fixed subsets (`clean100` / `clean460` / `all960`) and concatenates parts into a train split; validation/test splits come from `cfg.data.eval_split` / `test_split`. Preprocessing filters only the **training** split by duration bounds (so all eval clips are still scored), then maps features through `Wav2Vec2FeatureExtractor` and tokenizes labels. `input_length` is preserved for HF's `group_by_length` bucketing.

**Cache discipline (important on cluster nodes).** Every entrypoint calls `setup_cache_dir(cfg.data.cache_dir)` *before* anything that could trigger a download. It sets `HF_HOME`, `HF_DATASETS_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE` so arrow shards, audio files, and transformers hub artifacts all land on scratch instead of `$HOME`. If you add a new script, call `setup_cache_dir` first — otherwise subprocess workers inherit the wrong paths and re-download into home.

**Trainer + wandb.** `scripts/train.py` uses `Seq2SeqTrainer` with `predict_with_generate=True` and an inverse-sqrt LR schedule. `init_wandb` (in `wandb_utils.py`) is called *before* the Trainer is constructed, so HF's built-in `WandbCallback` reuses the existing `wandb.run` instead of creating its own. `epoch` is registered as a first-class metric with `train/global_step` as the x-axis. `EpochLoggerCallback` mirrors epoch/step to `wandb.summary`; `PredictionsTableCallback` logs a small ref/pred table each eval using a fixed random sample of validation features. `--no_wandb` also strips `"wandb"` from `report_to` so HF's callback doesn't re-enable it.

**Evaluation.** `scripts/evaluate.py` runs manual beam search via `model.generate` (bf16 autocast on CUDA) and computes WER with `evaluate.load("wer")`. Results JSON is **appended** (list of runs), not overwritten. Eval runs get auto-tagged with `eval:<split>` in wandb. Evaluation forces wandb on even if the training config disabled it (unless `--no_wandb` is passed).

## Conventions worth preserving

- The leading-BOS strip in `DataCollatorSpeechSeq2SeqWithPadding` is load-bearing — don't remove the `labels[:, 0] == decoder_start_token_id` check without also removing the tokenizer's `TemplateProcessing` post-processor.
- Only the train split is length-filtered; keep it that way so evaluation WER is over the full split.
- `setup_cache_dir` uses `os.environ.setdefault` — if env vars are already set (e.g. by a SLURM script), those win. Do not switch to hard assignment.
- `results/wer.json` accumulates; do not change it to overwrite.
