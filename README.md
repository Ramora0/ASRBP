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

## 1. Build the tokenizer

```bash
python scripts/prepare_tokenizer.py --subset all960 --vocab_size 1000
# → writes ./tokenizer/
```

Use `--subset clean100` to train the BPE on just `train-clean-100` (faster; same alphabet).

## 2. Train

Smoke test on the 100 h subset:

```bash
python scripts/train.py \
  --subset clean100 \
  --max_steps 200 \
  --eval_steps 100 \
  --per_device_train_batch_size 4 \
  --output_dir outputs/smoke
```

Full 960 h run:

```bash
python scripts/train.py \
  --subset all960 \
  --max_steps 150000 \
  --output_dir outputs/full
```

Defaults come from `configs/conformer_small.yaml`. Any CLI flag overrides the YAML.

## 3. Evaluate WER on `test-clean`

```bash
python scripts/evaluate.py \
  --checkpoint outputs/full/checkpoint-best \
  --split test.clean \
  --num_beams 5
# → writes results/wer.json
```

`--split test.other` also works if you want to report the noisy split.

## Memory and hyperparameters

The default config (`configs/conformer_small.yaml`) targets ~50 M parameters and batches of 8 × 20 s clips in bf16, comfortably under 32 GB. To reduce memory further:

- lower `per_device_train_batch_size` and raise `gradient_accumulation_steps`
- lower `max_audio_seconds` (default 20)
- turn on `gradient_checkpointing: true`

## Layout

```
conformer_asr/   # library code (importable)
scripts/         # entrypoints (prepare_tokenizer, train, evaluate)
configs/         # YAML hyperparameter configs
```
