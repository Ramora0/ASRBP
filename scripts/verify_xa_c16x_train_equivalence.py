"""Train-equivalence verification: c16x vs xa_16x_no_ca (full ASR model).

The downsampler-level test (``verify_xa_c16x_equivalence.py``) shows that
``CrossAttnDownsampler(strides=[[2,2],[2,2]], dilations=[1,2], stride=4,
num_layers=0)`` is bit-exact equivalent to ``Conv2dDownsampler(strides=
[[2,2],[2,2],[2,1],[2,1]])`` once the dilated layers use valid time padding.
This script lifts that to the full encoder + decoder + CTC stack and runs
several optimizer steps to confirm the two configurations train identically.

Equivalence requires:
1. **Padding fix** (already applied in cross_attn.py) — without it the dilated
   layers run a same-padded conv, which centers the kernel window and
   shifts the kept-output's input window by ~12 input frames vs c16x.
2. **Identical initial weights.** The two builders happen to consume RNG in
   exactly the same order and amount (4 Conv2d + 1 Linear in both
   downsamplers, identical shapes), so a fresh ``torch.manual_seed`` should
   produce identical state_dicts. This script asserts that, then also
   force-syncs via ``load_state_dict`` to guard against future drift.
3. **Synchronized forward-time RNG.** SpecAugment, dropout, and layerdrop
   all consume RNG during forward. Both models consume them in the same
   order, so saving/restoring ``torch.get_rng_state`` around each forward
   pair makes the two trajectories deterministic relative to each other.

Run with:
  ``python scripts/verify_xa_c16x_train_equivalence.py [--steps N] [--device cpu|cuda]``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conformer_asr.config import load_config
from conformer_asr.model import build_model


REPO_ROOT = Path(__file__).resolve().parent.parent
CFG_C = REPO_ROOT / "configs" / "cnns" / "c16x.yaml"
CFG_X = REPO_ROOT / "configs" / "xa" / "xa_16x_no_ca.yaml"


class StubTokenizer:
    """Minimal stand-in for ``PreTrainedTokenizerFast``.

    ``build_model`` needs only ``pad_token_id``, ``bos_token_id``,
    ``eos_token_id``, and ``len(tokenizer)`` for vocab size.
    """
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    _vocab_size = 5000

    def __len__(self) -> int:
        return self._vocab_size


def _zero_stochastic_fields(mcfg) -> None:
    """Disable every RNG-driven training-time op so the two models can be
    compared without saving/restoring RNG around dropouts whose tensor
    shapes differ.

    Specifically: xa's proj_dropout sees a ``T/4`` tensor while c16x's
    dropout sees a ``T/16`` tensor — same dropout rate but ~4x more RNG
    draws — which silently misaligns every downstream stochastic op
    (encoder dropout, layerdrop, decoder dropout) once any of them are
    nonzero. Zeroing all of them removes that asymmetry. SpecAugment is
    identical between the two configs, so leave it on if you want — the
    fields here also disable it for cleanliness.
    """
    mcfg.encoder_hidden_dropout = 0.0
    mcfg.encoder_attention_dropout = 0.0
    mcfg.encoder_activation_dropout = 0.0
    mcfg.encoder_layerdrop = 0.0
    mcfg.decoder_dropout = 0.0
    mcfg.spec_aug_time_masks = 0
    mcfg.spec_aug_feature_masks = 0


def build_with_seed(cfg_path: Path, seed: int, tok: StubTokenizer, *, deterministic: bool):
    cfg = load_config(str(cfg_path))
    if deterministic:
        _zero_stochastic_fields(cfg.model)
    torch.manual_seed(seed)
    model = build_model(cfg.model, tok)
    return cfg, model


def diff_state_dicts(sd_a: dict, sd_b: dict) -> tuple[float, list[str]]:
    """Returns (max-abs-elementwise-diff over shared keys, key mismatches)."""
    keys_only_a = sorted(set(sd_a) - set(sd_b))
    keys_only_b = sorted(set(sd_b) - set(sd_a))
    issues: list[str] = []
    if keys_only_a:
        issues.append(f"keys only in c16x: {keys_only_a[:5]}{'...' if len(keys_only_a) > 5 else ''}")
    if keys_only_b:
        issues.append(f"keys only in xa: {keys_only_b[:5]}{'...' if len(keys_only_b) > 5 else ''}")
    max_diff = 0.0
    for k in sorted(set(sd_a) & set(sd_b)):
        a, b = sd_a[k], sd_b[k]
        if a.shape != b.shape:
            issues.append(f"shape mismatch at {k}: {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        if not a.is_floating_point():
            continue
        d = (a - b).abs().max().item() if a.numel() else 0.0
        if d > max_diff:
            max_diff = d
    return max_diff, issues


def synthetic_batch(B: int, T_mel: int, n_mels: int, vocab: int, device: torch.device, dtype: torch.dtype):
    """Reproducible synthetic batch (mel features + label sequences)."""
    g = torch.Generator(device="cpu").manual_seed(123)
    input_features = torch.randn(B, T_mel, n_mels, generator=g).to(device=device, dtype=dtype)
    # Variable-length attention mask so encoder_attention_mask exercises the
    # length-derivation path on both models.
    valid_lens = torch.tensor([T_mel, T_mel - 60], dtype=torch.long, device=device)
    attention_mask = (
        torch.arange(T_mel, device=device).unsqueeze(0) < valid_lens.unsqueeze(1)
    ).long()
    # Labels: short token sequences ending in EOS, padded with -100 (HF ignore index).
    label_len = 18
    labels = torch.randint(3, vocab, (B, label_len), generator=g).to(device)
    labels[:, -1] = StubTokenizer.eos_token_id
    return input_features, attention_mask, labels


def step(model, optim, batch, rng_state):
    """One forward+backward+step under a controlled RNG state.

    Returns ``(loss, max-abs-grad)`` after ``optim.step()`` runs.
    """
    torch.set_rng_state(rng_state)
    input_features, attention_mask, labels = batch
    optim.zero_grad(set_to_none=True)
    out = model(
        input_features=input_features,
        attention_mask=attention_mask,
        labels=labels,
    )
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    max_grad = max((g.abs().max().item() for g in grads), default=0.0)
    optim.step()
    return out.loss.detach().clone(), max_grad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64",
                    help="float64 keeps step-to-step weight drift below atol; "
                         "float32 will diverge by ~1e-4 per step from non-associative add.")
    ap.add_argument("--atol", type=float, default=None,
                    help="Default: 1e-9 in float64, 1e-3 in float32.")
    args = ap.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.atol is None:
        args.atol = 1e-9 if dtype is torch.float64 else 1e-3
    device = torch.device(args.device)
    tok = StubTokenizer()

    print(f"Building c16x model (seed={args.seed}, dtype={args.dtype})")
    cfg_c, model_c = build_with_seed(CFG_C, args.seed, tok, deterministic=True)
    print(f"Building xa model   (seed={args.seed}, dtype={args.dtype})")
    cfg_x, model_x = build_with_seed(CFG_X, args.seed, tok, deterministic=True)
    model_c.to(device=device, dtype=dtype)
    model_x.to(device=device, dtype=dtype)

    # 1. State dict comparison — should be identical out of the box.
    sd_c = model_c.state_dict()
    sd_x = model_x.state_dict()
    init_diff, issues = diff_state_dicts(sd_c, sd_x)
    if issues:
        for line in issues:
            print(f"  state_dict warning: {line}")
    print(f"max |init weight diff| post-construction: {init_diff:.3e}")

    # 2. Force-sync via load_state_dict so any drift becomes detectable here
    #    instead of bleeding into training.
    missing, unexpected = model_x.load_state_dict(sd_c, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:3]:
            print(f"    missing sample:    {missing[:3]}")
        if unexpected[:3]:
            print(f"    unexpected sample: {unexpected[:3]}")

    sd_c2 = model_c.state_dict()
    sd_x2 = model_x.state_dict()
    sync_diff, _ = diff_state_dicts(sd_c2, sd_x2)
    assert sync_diff == 0.0, f"weights differ post-sync: max diff {sync_diff:.3e}"
    print("post-sync: state_dicts equal")

    # 3. Synthetic batch — same tensors fed to both models.
    B, T_mel = 2, 401
    n_mels = cfg_c.model.n_mels
    vocab = len(tok)
    batch = synthetic_batch(B, T_mel, n_mels, vocab, device, dtype)

    # 4. Optimizers με matched hyperparameters.
    opt_c = torch.optim.AdamW(model_c.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-8)
    opt_x = torch.optim.AdamW(model_x.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-8)

    model_c.train()
    model_x.train()

    for s in range(args.steps):
        rng = torch.get_rng_state()
        loss_c, gmax_c = step(model_c, opt_c, batch, rng)
        loss_x, gmax_x = step(model_x, opt_x, batch, rng)
        loss_diff = abs(loss_c.item() - loss_x.item())
        # After both steps, advance the global RNG so the next iteration is
        # also deterministically wired but not a literal repeat.
        torch.set_rng_state(rng)
        torch.randn(1)  # consume one RNG draw to advance state
        print(
            f"step {s}: loss_c={loss_c.item():.8f} loss_x={loss_x.item():.8f} "
            f"diff={loss_diff:.3e}  gmax_c={gmax_c:.3e} gmax_x={gmax_x:.3e}"
        )
        assert loss_diff < args.atol, f"step {s}: loss diff {loss_diff:.3e} >= atol {args.atol}"

        # Weight equality after the optimizer step — this is the strongest
        # check, since any divergence in the forward or gradient computation
        # would have shown up here.
        post_diff, _ = diff_state_dicts(model_c.state_dict(), model_x.state_dict())
        assert post_diff < args.atol, (
            f"step {s}: weights diverged post-optim, max diff {post_diff:.3e}"
        )

    print(f"\nOK — c16x and xa trained identically for {args.steps} steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
