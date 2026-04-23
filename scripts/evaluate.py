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


def _load_ckpt_state(ckpt_path: Path) -> dict[str, torch.Tensor]:
    safetensors_file = ckpt_path / "model.safetensors"
    pt_file = ckpt_path / "pytorch_model.bin"
    if safetensors_file.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_file))
    if pt_file.exists():
        return torch.load(str(pt_file), map_location="cpu")
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {ckpt_path}")


def _average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    # Equal-weight average. Keys must match across all state dicts; floating-
    # point tensors are averaged in fp32 then cast back to the base dtype;
    # integer buffers (e.g. position indices) are left as the first dict's copy.
    base = states[0]
    base_keys = set(base.keys())
    for i, s in enumerate(states[1:], start=1):
        if set(s.keys()) != base_keys:
            missing = base_keys - set(s.keys())
            extra = set(s.keys()) - base_keys
            raise ValueError(
                f"state_dict keys mismatch between ckpt 0 and ckpt {i}: "
                f"missing={list(missing)[:5]}, extra={list(extra)[:5]}"
            )
    averaged: dict[str, torch.Tensor] = {}
    n = len(states)
    for name, base_tensor in base.items():
        if not base_tensor.is_floating_point():
            averaged[name] = base_tensor.clone()
            continue
        acc = base_tensor.detach().float().clone()
        for s in states[1:]:
            acc += s[name].detach().float()
        averaged[name] = (acc / n).to(base_tensor.dtype)
    return averaged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/conformer_small.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to saved model directory")
    p.add_argument(
        "--avg_checkpoints",
        nargs="+",
        default=None,
        help="Optional extra checkpoint dirs to equal-weight average with --checkpoint (SWA-style).",
    )
    p.add_argument("--tokenizer_dir", default=None)
    p.add_argument("--split", default="test.clean")
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3,
                   help="If >0, ban repetition of any n-gram of this size during generation. "
                        "Default 3 stops beam-search loops without hurting natural English.")
    p.add_argument("--repetition_penalty", type=float, default=1.15,
                   help="Soft penalty on previously-emitted tokens. 1.0 = off; default 1.15 "
                        "was the best on a test.clean subset sweep at num_beams=10.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--output_json", default="results/wer.json")
    p.add_argument("--cache_dir", default=None, help="overrides data.cache_dir")
    # --- LM rescoring (n-best). 0.0 disables; otherwise rescores the beam. ---
    p.add_argument(
        "--lm_weight",
        type=float,
        default=0.0,
        help="Weight on SpeechBrain TransformerLM log-prob in n-best rescoring. 0 disables.",
    )
    p.add_argument(
        "--len_penalty",
        type=float,
        default=0.0,
        help="Per-token bonus added to rescored score (β in α·acoustic + lm_weight·lm + β·len).",
    )
    p.add_argument(
        "--lm_repo",
        default="speechbrain/asr-transformer-transformerlm-librispeech",
        help="HF repo id for the rescoring LM; expects lm.ckpt + tokenizer.ckpt.",
    )
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
    tokenizer = load_tokenizer(tokenizer_dir, cache_dir=cfg.data.cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Rebuild the architecture via ``build_model`` (which wires in the custom
    # ``MelConformerEncoder``) instead of ``SpeechEncoderDecoderModel.from_pretrained``
    # — the latter would instantiate the stock ``Wav2Vec2ConformerModel``
    # (raw-waveform CNN frontend) because ``MelConformerEncoder`` isn't a
    # registered HF model, and then ``generate()`` would try to convolve a
    # raw-waveform kernel over the log-Mel features.
    model = build_model(cfg.model, tokenizer)
    ckpt_paths = [Path(args.checkpoint)] + [Path(p) for p in (args.avg_checkpoints or [])]
    if len(ckpt_paths) == 1:
        print(f"Building model architecture and loading weights from {ckpt_paths[0]} (device={device})")
        state = _load_ckpt_state(ckpt_paths[0])
    else:
        print(f"Building model architecture and averaging {len(ckpt_paths)} checkpoints (device={device}):")
        for p in ckpt_paths:
            print(f"  - {p}")
        state = _average_state_dicts([_load_ckpt_state(p) for p in ckpt_paths])
    missing, unexpected = model.load_state_dict(state, strict=False)
    total_keys = len(state)
    loaded = total_keys - len(unexpected)
    print(
        f"[state_dict] loaded {loaded}/{total_keys} tensors "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing:
        print(f"[state_dict] first 10 missing: {missing[:10]}")
    if unexpected:
        print(f"[state_dict] first 10 unexpected: {unexpected[:10]}")
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
            "avg_checkpoints": args.avg_checkpoints,
            "split": args.split,
            "num_beams": args.num_beams,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples or len(ds),
            "lm_weight": args.lm_weight,
            "len_penalty": args.len_penalty,
            "lm_repo": args.lm_repo if args.lm_weight > 0 else None,
        },
        job_type="eval",
    )

    rescore = args.lm_weight > 0
    lm_scorer = None
    if rescore:
        from conformer_asr.sb_lm import SBLMScorer

        print(f"Loading SpeechBrain LM from {args.lm_repo} (device={device})")
        lm_scorer = SBLMScorer.from_hub(
            repo=args.lm_repo, device=device, cache_dir=cfg.data.cache_dir
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
            if rescore:
                # Pull back the full beam + per-sequence acoustic scores so we
                # can rerank with the external LM.
                gen = model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    max_length=cfg.train.generation_max_length,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    repetition_penalty=args.repetition_penalty,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                # sequences: (B*beam, L); sequences_scores: (B*beam,) sum log-probs (length-normalized
                # by HF's default length_penalty=1.0 applied internally — that's baseline already).
                seqs = gen.sequences
                ac_scores = gen.sequences_scores
            else:
                seqs = model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    num_beams=args.num_beams,
                    max_length=cfg.train.generation_max_length,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    repetition_penalty=args.repetition_penalty,
                )
                ac_scores = None

        if rescore:
            bsz = len(audios)
            k = args.num_beams
            hyp_texts = tokenizer.batch_decode(seqs, skip_special_tokens=True)
            # LM expects uppercase (SB's SentencePiece was trained on uppercase LibriSpeech text).
            lm_scores = lm_scorer.score_hypotheses(hyp_texts)  # (B*k,)
            # Token lengths post-detokenize; use word count as a proxy for the length-penalty term
            # since the acoustic score is already length-normalized by HF's default.
            word_lens = torch.tensor(
                [max(1, len(t.split())) for t in hyp_texts], dtype=torch.float32
            )
            total = ac_scores.cpu().float() + args.lm_weight * lm_scores + args.len_penalty * word_lens
            total = total.view(bsz, k)
            best = total.argmax(dim=1)  # (B,)
            preds = [hyp_texts[i * k + best[i].item()] for i in range(bsz)]
        else:
            preds = tokenizer.batch_decode(seqs, skip_special_tokens=True)
        refs = [normalize_text(t) for t in batch["text"]]
        preds_all.extend(preds)
        refs_all.extend(refs)

    score = compute_wer(preds_all, refs_all)
    result = {
        "checkpoint": args.checkpoint,
        "avg_checkpoints": args.avg_checkpoints,
        "split": args.split,
        "num_beams": args.num_beams,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "repetition_penalty": args.repetition_penalty,
        "lm_weight": args.lm_weight,
        "len_penalty": args.len_penalty,
        "lm_repo": args.lm_repo if args.lm_weight > 0 else None,
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

    # Print a few ref/hyp pairs so huge unexpected WERs (98%+) can be
    # eyeballed: garbled predictions vs. a tokenization/case mismatch look
    # very different.
    n_show = min(10, len(preds_all))
    print(f"\nSample predictions (first {n_show}):")
    for i in range(n_show):
        print(f"  REF: {refs_all[i]}")
        print(f"  HYP: {preds_all[i]}\n")

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
