# Conformer ASR from Scratch on LibriSpeech

Trains an attention encoder-decoder ASR model from scratch:

- **Encoder**: `Wav2Vec2ConformerModel` (HuggingFace), randomly initialized
- **Decoder**: `BartForCausalLM` with cross-attention
- Glued together with `SpeechEncoderDecoderModel`
- Loss: cross-entropy with label smoothing (pure AED — no CTC branch)
- Inference: beam search via `model.generate`
- Tokenizer: SentencePiece BPE (1000 units) trained on LibriSpeech transcripts
- Sized to fit training on a single 32 GB GPU with bf16

## Setup

```bash
uv pip install -e .
```

## 0. (Recommended) Pre-download LibriSpeech to scratch

LibriSpeech is ~60 GB once extracted. The default cache location is
`/fs/scratch/PAS2836/lees_stuff/hf_cache` (set in `configs/conformer_c4x.yaml`
under `data.cache_dir`). Every script also sets `HF_HOME`, `HF_DATASETS_CACHE`,
`HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` to the same place so sub-processes
(HF workers, transformers hub downloads) don't scatter files into `$HOME`.

Run this once on a login node with internet, before scheduling a GPU job:

```bash
python scripts/download_librispeech.py
# override the cache path if needed:
python scripts/download_librispeech.py --cache_dir /path/to/scratch/hf_cache
```

This fetches all seven splits (train-clean-100/360, train-other-500, validation-clean/other, test-clean/other). Subsequent `train.py` / `evaluate.py` runs hit the cache — useful on GPU nodes without outbound network.

> **How the cache redirect actually works.** `datasets` and `transformers` read
> `HF_HOME` / `HF_DATASETS_CACHE` / `TRANSFORMERS_CACHE` *at import time* and
> freeze the resolved paths. So every entrypoint imports `bootstrap_cache`
> (a tiny module at the project root that has no HF deps) and calls
> `bootstrap_cache_from_argv()` **before** any `datasets` / `transformers` /
> `conformer_asr.*` import. It reads `--cache_dir` / `--config` from `sys.argv`
> and sets the env vars up front. To force a different location outside the
> YAML / CLI, export `HF_HOME_OVERRIDE=/some/path` — it wins over everything.

## 1. Tokenizer

No separate step — `scripts/train.py` fetches SpeechBrain's pretrained SentencePiece
(`speechbrain/asr-transformer-transformerlm-librispeech`, 5K unigram, uppercase)
from HF Hub into `cache_dir` on first run. Override via `data.tokenizer_dir`
in the YAML to point at a local directory containing `sentencepiece.model`
(e.g. a previous run's `final/` dir) if you want to reuse a saved copy.

## 2. Train

Full 960 h run (the default — no `--subset` flag needed):

```bash
python scripts/train.py --output_dir outputs/full
```

Smoke test on the 100 h subset:

```bash
python scripts/train.py \
  --subset clean100 \
  --max_steps 200 \
  --eval_steps 100 \
  --per_device_train_batch_size 4 \
  --output_dir outputs/smoke
```

Defaults come from `configs/conformer_c4x.yaml`. Any CLI flag overrides the YAML.

## 3. Evaluate WER on `test-clean`

```bash
python scripts/evaluate.py \
  --checkpoint outputs/full/checkpoint-best \
  --split test.clean \
  --num_beams 5
# → writes results/wer.json
```

`--split test.other` also works if you want to report the noisy split.

## Weights & Biases

wandb is on by default (`report_to: wandb` in the YAML). Before running anything, log in once:

```bash
wandb login
```

Then train/eval automatically log:

- **train**: loss, learning_rate, grad_norm, **epoch** (as a first-class metric on the x-axis), WER on `validation.clean` at every eval step, a small table of sample predictions per eval
- **eval**: final WER on `test.clean` / `test.other` as a summary metric, plus a table of reference/prediction pairs
- the full resolved config (model/data/train sections) as the run config
- model parameter count, dataset sizes, and tokenizer vocab size as summary fields
- the final model directory as a wandb Artifact

Common overrides:

```bash
python scripts/train.py --wandb_run_name conformer-small-960h --wandb_tags baseline,bf16
python scripts/train.py --no_wandb                        # disable entirely
python scripts/evaluate.py --checkpoint ... --wandb_tags final-eval
```

Set `wandb.watch_model: true` in the YAML to additionally log gradients and weights (bandwidth-heavy; off by default).

## Memory and hyperparameters

The default config (`configs/conformer_c4x.yaml`) targets ~50 M parameters and batches of 8 × 20 s clips in bf16, comfortably under 32 GB. To reduce memory further:

- lower `per_device_train_batch_size` and raise `gradient_accumulation_steps`
- lower `max_audio_seconds` (default 20)
- turn on `gradient_checkpointing: true`

## Layout

```
conformer_asr/   # library code (importable)
scripts/         # entrypoints (preprocess, train, evaluate)
configs/         # YAML hyperparameter configs
```
