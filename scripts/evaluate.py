"""Evaluate a trained checkpoint on a LibriSpeech split and report WER."""
from __future__ import annotations

import argparse
import json
import os
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
import torch.nn.functional as F  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from conformer_asr.config import autocast_dtype as _autocast_dtype, load_config, resolve_precision  # noqa: E402
from conformer_asr.data import setup_cache_dir  # noqa: E402
from conformer_asr.features import log_mel_spectrogram  # noqa: E402
from conformer_asr.model import build_model  # noqa: E402
from conformer_asr.tokenizer import load_tokenizer, normalize_text  # noqa: E402


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_main() -> bool:
    return _rank() == 0


def _remap_legacy_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate pre-refactor checkpoint key names to current module layout.

    Old training runs saved the Conv2d stem as ``encoder.subsample.*`` and the
    projection as ``encoder.proj.*`` (flat). The refactored encoder nests them
    under ``encoder.downsampler.convs.*`` / ``encoder.downsampler.proj.*``.
    Also re-ties ``decoder.lm_head.weight`` to the decoder embedding when the
    checkpoint only stored the tied copy.
    """
    remapped: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("encoder.subsample."):
            nk = nk.replace("encoder.subsample.", "encoder.downsampler.convs.", 1)
        elif nk.startswith("encoder.proj."):
            nk = nk.replace("encoder.proj.", "encoder.downsampler.proj.", 1)
        remapped[nk] = v
    emb_key = "decoder.model.decoder.embed_tokens.weight"
    head_key = "decoder.lm_head.weight"
    if head_key not in remapped and emb_key in remapped:
        remapped[head_key] = remapped[emb_key]
    return remapped


def _load_ckpt_state(ckpt_path: Path) -> dict[str, torch.Tensor]:
    safetensors_file = ckpt_path / "model.safetensors"
    pt_file = ckpt_path / "pytorch_model.bin"
    if safetensors_file.exists():
        from safetensors.torch import load_file

        raw = load_file(str(safetensors_file))
    elif pt_file.exists():
        raw = torch.load(str(pt_file), map_location="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {ckpt_path}")
    return _remap_legacy_keys(raw)


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
    p.add_argument("--config", default="configs/cnns/c4x.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to saved model directory")
    p.add_argument(
        "--avg_checkpoints",
        nargs="+",
        default=None,
        help="Optional extra checkpoint dirs to equal-weight average with --checkpoint (SWA-style).",
    )
    p.add_argument("--tokenizer_dir", default=None)
    p.add_argument("--split", default="test.clean")
    p.add_argument("--num_beams", type=int, default=10)
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
        default=0.05,
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
    # --- CTC n-best rescoring. 0.0 disables; uses the already-trained CTC head. ---
    p.add_argument(
        "--ctc_weight",
        type=float,
        default=0.0,
        help="Weight on CTC sequence log-prob (length-normalized) in n-best rescoring. "
             "0 disables.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.cache_dir:
        cfg.data.cache_dir = args.cache_dir
    setup_cache_dir(cfg.data.cache_dir)
    resolve_precision(cfg.train)

    # Multi-GPU eval: launch with `torchrun --nproc_per_node=N scripts/evaluate.py ...`.
    # WORLD_SIZE unset (== 1) is the single-GPU path — no process group, no sharding.
    world_size = _world_size()
    rank = _rank()
    local_rank = _local_rank()
    is_main = _is_main()
    distributed = world_size > 1
    if distributed:
        import torch.distributed as dist

        if not dist.is_initialized():
            # NCCL for GPU collectives. `torchrun` sets MASTER_ADDR/PORT automatically.
            dist.init_process_group(backend="nccl")

    tokenizer_dir = args.tokenizer_dir or cfg.data.tokenizer_dir
    tokenizer = load_tokenizer(tokenizer_dir, cache_dir=cfg.data.cache_dir)

    if torch.cuda.is_available():
        # Each rank pinned to its own GPU; required for NCCL collectives (including
        # all_gather_object, which serializes through a CUDA ByteTensor).
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    # Rebuild the architecture via ``build_model`` (which wires in the custom
    # mel-input encoder with its configured downsampler) instead of
    # ``SpeechEncoderDecoderModel.from_pretrained`` — the latter would
    # instantiate the stock ``Wav2Vec2ConformerModel`` (raw-waveform CNN
    # frontend) because our encoder isn't a registered HF model, and then
    # ``generate()`` would try to convolve a raw-waveform kernel over the
    # log-Mel features.
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
    # Shard across ranks AFTER max_samples so the flag means "total samples
    # across all GPUs", δεν per-rank. ``contiguous=True`` preserves original
    # order within each shard (helpful for group_by_length-style batching,
    # though eval doesn't do that).
    full_len = len(ds)
    if distributed:
        ds = ds.shard(num_shards=world_size, index=rank, contiguous=True)
        print(f"[rank {rank}/{world_size}] shard size: {len(ds)} (total: {full_len})")

    ctc_rescore = args.ctc_weight > 0.0
    if ctc_rescore and not hasattr(model, "ctc_head"):
        raise ValueError(
            "--ctc_weight > 0 but the loaded model has no CTC head. "
            "Set model.ctc_enabled=true in the config (and retrain), or pass --ctc_weight 0."
        )
    rescore = args.lm_weight > 0 or ctc_rescore
    lm_scorer = None
    if args.lm_weight > 0:
        # Stub torchaudio before SpeechBrain pulls it in — this environment has
        # torchaudio 2.11 compiled against libcudart.so.13 but torch 2.6+cu124,
        # so the real import fails. The LM path only needs TransformerLM, which
        # doesn't touch any torchaudio symbol.
        import sys as _sys
        if "torchaudio" not in _sys.modules:
            _ta = type(_sys)("torchaudio")
            _ta.__version__ = "0.0.0"
            _sys.modules["torchaudio"] = _ta
        from conformer_asr.sb_lm import SBLMScorer

        print(f"Loading SpeechBrain LM from {args.lm_repo} (device={device})")
        lm_scorer = SBLMScorer.from_hub(
            repo=args.lm_repo, device=device, cache_dir=cfg.data.cache_dir
        )

    preds_all: list[str] = []
    refs_all: list[str] = []
    # BP empirical compression. Stays at 0 for static downsamplers (no
    # ``last_stats``); only meaningful when the encoder uses a BP downsampler.
    bp_n_boundaries = 0
    bp_n_post_frontend = 0
    bp_n_input = 0

    autocast_dtype = _autocast_dtype(cfg.train) if device == "cuda" else torch.float32
    use_autocast = device == "cuda" and autocast_dtype != torch.float32

    for start in tqdm(
        range(0, len(ds), args.batch_size),
        desc=f"generate({args.split})",
        disable=not is_main,
    ):
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

        # Boundary-predictor compression bookkeeping. Tally per batch so the
        # final eval result records the empirical compression rate the model
        # actually produced over this checkpoint+split — independent of the
        # ``prior`` configured at training time.
        bp_stats_fn = getattr(getattr(model, "encoder", None), "downsampler", None)
        bp_stats_fn = getattr(bp_stats_fn, "last_stats", None) if bp_stats_fn is not None else None
        if bp_stats_fn is not None:
            stats = bp_stats_fn()
            if stats is not None:
                bp_n_boundaries += stats["n_boundaries"]
                bp_n_post_frontend += stats["n_post_frontend"]
                bp_n_input += stats["n_input"]

        if rescore:
            bsz = len(audios)
            k = args.num_beams
            hyp_texts = tokenizer.batch_decode(seqs, skip_special_tokens=True)
            # Token lengths post-detokenize; use word count as a proxy for the length-penalty term
            # since the acoustic score is already length-normalized by HF's default.
            word_lens = torch.tensor(
                [max(1, len(t.split())) for t in hyp_texts], dtype=torch.float32
            )
            total = ac_scores.cpu().float() + args.len_penalty * word_lens

            if args.lm_weight > 0:
                # LM expects uppercase (SB's SentencePiece was trained on uppercase LibriSpeech text).
                lm_scores = lm_scorer.score_hypotheses(hyp_texts)  # (B*k,)
                total = total + args.lm_weight * lm_scores

            if ctc_rescore:
                # Extra encoder pass to get CTC logits — generate() doesn't expose them,
                # and the cost is negligible at eval time. One pass per batch, not per beam.
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=autocast_dtype, enabled=use_autocast
                ):
                    enc_out = model.encoder(
                        input_features=input_features,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                enc_hidden = enc_out[0]  # (B, T', H)
                enc_mask = model.encoder._get_feature_vector_attention_mask(
                    enc_hidden.size(1), attention_mask
                )
                # CTC head reads either the encoder hidden states (default) or
                # the post-downsampler tensor (``ctc_input='features'``). Both
                # share the same time dim, so the rest of the rescore path is
                # identical.
                if getattr(model, "ctc_input", "encoder") == "features":
                    ctc_source = model.encoder._features_for_ctc
                else:
                    ctc_source = enc_hidden
                ctc_logits = model.ctc_head(ctc_source)  # (B, T', V)

                B, T_enc, V = ctc_logits.shape
                # (B, T', V) -> (B*k, T', V) by expanding each sample across its k beams.
                expanded_logits = (
                    ctc_logits.unsqueeze(1)
                    .expand(B, k, T_enc, V)
                    .reshape(B * k, T_enc, V)
                )
                input_lens = enc_mask.sum(-1).long()
                expanded_input_lens = (
                    input_lens.unsqueeze(1).expand(B, k).reshape(-1)
                )  # (B*k,)

                # Build CTC targets from beam hypotheses, stripping bos/eos/pad.
                specials = {
                    tokenizer.pad_token_id,
                    tokenizer.bos_token_id,
                    tokenizer.eos_token_id,
                }
                tgts_list: list[list[int]] = [
                    [t for t in h if t not in specials] for h in seqs.tolist()
                ]
                raw_lens = [len(t) for t in tgts_list]
                # Clamp to >=1 for CTC bookkeeping; empty hypotheses will get zero_infinity=True
                # handling but still need a non-zero length to index into the padded tensor.
                tgt_lens = torch.tensor(
                    [max(1, n) for n in raw_lens], dtype=torch.long, device=device
                )
                max_tl = int(tgt_lens.max().item())
                padded = torch.zeros((B * k, max_tl), dtype=torch.long, device=device)
                for i, t in enumerate(tgts_list):
                    if t:
                        padded[i, : len(t)] = torch.tensor(
                            t, dtype=torch.long, device=device
                        )

                # CTC log-sum-exp is unstable in fp16/bf16; fp32 like the training path.
                log_probs = (
                    F.log_softmax(expanded_logits, dim=-1)
                    .transpose(0, 1)
                    .float()
                )  # (T', B*k, V)
                ctc_nll = F.ctc_loss(
                    log_probs,
                    padded,
                    expanded_input_lens,
                    tgt_lens,
                    blank=model.ctc_blank_id,
                    reduction="none",
                    zero_infinity=True,
                )  # (B*k,) — summed -log P over alignments, per hypothesis
                # Length-normalize so the CTC term is on the same per-token scale as
                # HF's length-normalized sequences_scores (default length_penalty=1.0).
                ctc_logp_norm = (-ctc_nll / tgt_lens.float()).cpu()
                total = total + args.ctc_weight * ctc_logp_norm

            total = total.view(bsz, k)
            best = total.argmax(dim=1)  # (B,)
            preds = [hyp_texts[i * k + best[i].item()] for i in range(bsz)]
        else:
            preds = tokenizer.batch_decode(seqs, skip_special_tokens=True)
        refs = [normalize_text(t) for t in batch["text"]]
        preds_all.extend(preds)
        refs_all.extend(refs)

    # Gather per-rank shards onto rank 0 for a single WER computation over the
    # concatenated lists (mean-of-per-shard WERs would be wrong — WER is a
    # global ratio of edit distance to total reference length, not an average).
    # ``all_gather_object`` pickles the lists through a CUDA ByteTensor under
    # NCCL, so every rank must have its device set (done above).
    if distributed:
        import torch.distributed as dist

        gathered_preds: list[list[str]] = [[] for _ in range(world_size)]
        gathered_refs: list[list[str]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered_preds, preds_all)
        dist.all_gather_object(gathered_refs, refs_all)
        # Sum BP tallies across ranks so the empirical compression rate is
        # computed over the full eval set, not per-rank shards.
        bp_tally = torch.tensor(
            [bp_n_boundaries, bp_n_post_frontend, bp_n_input],
            dtype=torch.long,
            device=device,
        )
        dist.all_reduce(bp_tally, op=dist.ReduceOp.SUM)
        if is_main:
            preds_all = [p for shard in gathered_preds for p in shard]
            refs_all = [r for shard in gathered_refs for r in shard]
            bp_n_boundaries = int(bp_tally[0].item())
            bp_n_post_frontend = int(bp_tally[1].item())
            bp_n_input = int(bp_tally[2].item())

    if not is_main:
        dist.destroy_process_group()
        return

    score = compute_wer(preds_all, refs_all)
    result = {
        "checkpoint": args.checkpoint,
        "avg_checkpoints": args.avg_checkpoints,
        "split": args.split,
        "num_beams": args.num_beams,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "repetition_penalty": args.repetition_penalty,
        "lm_weight": args.lm_weight,
        "ctc_weight": args.ctc_weight,
        "len_penalty": args.len_penalty,
        "lm_repo": args.lm_repo if args.lm_weight > 0 else None,
        "num_samples": len(preds_all),
        "wer": float(score),
    }
    if bp_n_post_frontend > 0:
        # Realized boundary rate after the BP downsampler's Conv2d frontend
        # ⇒ direct comparison against the configured ``prior``. ``compression``
        # is its reciprocal (post-frontend frames per output frame), and
        # ``total_compression`` extends to the raw mel rate (input frames per
        # output frame) — the end-to-end multiplier the encoder actually saw.
        result["bp_realized_prior"] = bp_n_boundaries / float(bp_n_post_frontend)
        result["bp_compression"] = bp_n_post_frontend / max(1, bp_n_boundaries)
        if bp_n_input > 0:
            result["bp_total_compression"] = bp_n_input / max(1, bp_n_boundaries)
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

    if distributed:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
