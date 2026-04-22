"""Evaluate a trained checkpoint on a LibriSpeech split and report WER."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate as hf_evaluate  # noqa: E402
import torch  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import SpeechEncoderDecoderModel, Wav2Vec2FeatureExtractor  # noqa: E402

from conformer_asr.config import load_config  # noqa: E402
from conformer_asr.tokenizer import load_tokenizer, normalize_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/conformer_small.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to saved model directory")
    p.add_argument("--tokenizer_dir", default=None)
    p.add_argument("--split", default="test.clean")
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--output_json", default="results/wer.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    tokenizer_dir = args.tokenizer_dir or cfg.data.tokenizer_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.checkpoint} (device={device})")
    model = SpeechEncoderDecoderModel.from_pretrained(args.checkpoint).to(device).eval()

    tokenizer = load_tokenizer(tokenizer_dir)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=cfg.data.sampling_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    print(f"Loading split {args.split}")
    ds = load_dataset(cfg.data.dataset_id, split=args.split, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=cfg.data.sampling_rate))
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    wer = hf_evaluate.load("wer")
    preds_all: list[str] = []
    refs_all: list[str] = []

    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    use_autocast = device == "cuda"

    for start in tqdm(range(0, len(ds), args.batch_size), desc=f"generate({args.split})"):
        batch = ds[start : start + args.batch_size]
        audios = [a["array"] for a in batch["audio"]]
        inputs = feature_extractor(
            audios,
            sampling_rate=cfg.data.sampling_rate,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
            generated = model.generate(
                input_values,
                attention_mask=attention_mask,
                num_beams=args.num_beams,
                max_length=cfg.train.generation_max_length,
            )
        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        refs = [normalize_text(t) for t in batch["text"]]
        preds_all.extend(preds)
        refs_all.extend(refs)

    score = wer.compute(predictions=preds_all, references=refs_all)
    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_beams": args.num_beams,
        "num_samples": len(preds_all),
        "wer": float(score),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path) as fh:
            existing = json.load(fh)
        if not isinstance(existing, list):
            existing = [existing]
    else:
        existing = []
    existing.append(result)
    with open(out_path, "w") as fh:
        json.dump(existing, fh, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
