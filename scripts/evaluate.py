"""Evaluate a trained checkpoint on a LibriSpeech split and report WER."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- HF cache bootstrap: MUST run before any HF / conformer_asr import. ---
from bootstrap_cache import bootstrap_cache_from_argv  # noqa: E402

_resolved_cache = bootstrap_cache_from_argv()
print(f"HF cache_dir (bootstrapped): {_resolved_cache}")
# -------------------------------------------------------------------------

from conformer_asr.metrics import compute_wer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from conformer_asr.config import autocast_dtype as _autocast_dtype, load_config, resolve_precision  # noqa: E402
from conformer_asr.data import setup_cache_dir  # noqa: E402
from conformer_asr.features import log_mel_spectrogram  # noqa: E402
from conformer_asr.model import build_model  # noqa: E402
from conformer_asr.tokenizer import load_tokenizer, normalize_text  # noqa: E402
from conformer_asr.wandb_utils import init_wandb  # noqa: E402


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
    p.add_argument("--cache_dir", default=None, help="overrides data.cache_dir")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_run_name", dest="run_name", default=None)
    p.add_argument("--wandb_tags", dest="tags", default=None, help="comma-separated")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.cache_dir:
        cfg.data.cache_dir = args.cache_dir
    setup_cache_dir(cfg.data.cache_dir)
    resolve_precision(cfg.train)

    if args.run_name:
        cfg.wandb.run_name = args.run_name
    if args.tags:
        cfg.wandb.tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    # Evaluation runs get tagged so they're easy to filter in the wandb UI.
    cfg.wandb.tags = list(cfg.wandb.tags) + [f"eval:{args.split}"]
    if args.no_wandb:
        cfg.wandb.enabled = False

    tokenizer_dir = args.tokenizer_dir or cfg.data.tokenizer_dir
    tokenizer = load_tokenizer(tokenizer_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Rebuild the architecture via ``build_model`` (which wires in the custom
    # ``MelConformerEncoder``) instead of ``SpeechEncoderDecoderModel.from_pretrained``
    # — the latter would instantiate the stock ``Wav2Vec2ConformerModel``
    # (raw-waveform CNN frontend) because ``MelConformerEncoder`` isn't a
    # registered HF model, and then ``generate()`` would try to convolve a
    # raw-waveform kernel over the log-Mel features.
    print(f"Building model architecture and loading weights from {args.checkpoint} (device={device})")
    model = build_model(cfg.model, tokenizer)
    ckpt_path = Path(args.checkpoint)
    safetensors_file = ckpt_path / "model.safetensors"
    pt_file = ckpt_path / "pytorch_model.bin"
    if safetensors_file.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_file))
    elif pt_file.exists():
        state = torch.load(str(pt_file), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin in {ckpt_path}"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing state_dict keys ({len(missing)}); first few: {missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected state_dict keys ({len(unexpected)}); first few: {unexpected[:5]}")
    model = model.to(device).eval()
    n_mels = cfg.model.n_mels
    n_fft = cfg.model.n_fft
    hop_length = cfg.model.hop_length

    print(f"Loading split {args.split} (cache_dir={cfg.data.cache_dir})")
    ds = load_dataset(
        cfg.data.dataset_id,
        split=args.split,
        trust_remote_code=True,
        cache_dir=cfg.data.cache_dir,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=cfg.data.sampling_rate))
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Force wandb on for evaluate.py even if the training config disabled it,
    # as long as the user didn't pass --no_wandb. Init BEFORE inference so any
    # generation errors are still captured by the run.
    force_report = "wandb" if (cfg.wandb.enabled and not args.no_wandb) else "none"
    cfg.train.report_to = force_report
    wandb_run = init_wandb(
        cfg,
        extra_config={
            "checkpoint": args.checkpoint,
            "split": args.split,
            "num_beams": args.num_beams,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples or len(ds),
        },
        job_type="eval",
    )

    preds_all: list[str] = []
    refs_all: list[str] = []

    autocast_dtype = _autocast_dtype(cfg.train) if device == "cuda" else torch.float32
    use_autocast = device == "cuda" and autocast_dtype != torch.float32

    for start in tqdm(range(0, len(ds), args.batch_size), desc=f"generate({args.split})"):
        batch = ds[start : start + args.batch_size]
        audios = [np.asarray(a["array"], dtype=np.float32) for a in batch["audio"]]
        # Compute log-Mel per sample, then right-pad to batch max along time.
        mels = [
            log_mel_spectrogram(
                a,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                sampling_rate=cfg.data.sampling_rate,
            )
            for a in audios
        ]
        lengths = [m.shape[0] for m in mels]
        t_max = max(lengths)
        input_features = torch.zeros((len(mels), t_max, n_mels), dtype=torch.float32)
        attention_mask = torch.zeros((len(mels), t_max), dtype=torch.long)
        for i, m in enumerate(mels):
            input_features[i, : m.shape[0]] = m
            attention_mask[i, : m.shape[0]] = 1
        input_features = input_features.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
            generated = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                num_beams=args.num_beams,
                max_length=cfg.train.generation_max_length,
            )
        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        refs = [normalize_text(t) for t in batch["text"]]
        preds_all.extend(preds)
        refs_all.extend(refs)

    score = compute_wer(preds_all, refs_all)
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

    if wandb_run is not None:
        import wandb

        key = f"final_wer/{args.split}"
        wandb_run.summary[key] = float(score)
        wandb_run.summary["num_beams"] = args.num_beams
        wandb_run.summary["num_samples"] = len(preds_all)
        wandb.log({key: float(score)})

        if cfg.wandb.log_preds_table:
            n = min(cfg.wandb.log_preds_n, len(preds_all))
            table = wandb.Table(columns=["reference", "prediction"])
            for ref, pred in zip(refs_all[:n], preds_all[:n]):
                table.add_data(ref, pred)
            wandb.log({f"eval/{args.split}/predictions": table})
        wandb.finish()


if __name__ == "__main__":
    main()
