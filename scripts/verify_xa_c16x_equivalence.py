"""Verify that CrossAttnDownsampler with num_layers=0 is bit-exact equivalent
to Conv2dDownsampler with the matching strided cascade.

Pairing:
- c16x: Conv2dDownsampler(strides=[[2,2],[2,2],[2,1],[2,1]])
- xa:   CrossAttnDownsampler(strides=[[2,2],[2,2]], dilations=[1, 2],
                             stride=4, num_layers=0)

Equivalence comes from the à trous identity: a stride-2-stride-2 cascade
(c16x's L3+L4) re-parameterizes as dilation-1 + dilation-2 stride-1 convs
followed by a strided pick of 4 — provided the dilated layers use *valid*
time-axis padding (pad=0). Mel side is identical: c16x's L3+L4 use stride-1
mel with pad=1 (dim preserved); xa's dilated layers use stride-1 mel with
pad=1 + dilation-1 (also dim preserved). Layer-by-layer kernel shapes
match: 1->H, H->H, H->H, H->H, all kernel-(3,3).

Test plan: copy the four conv kernels (and biases) and the post-stack
projection from c16x into xa, then check forward outputs and gradients
agree to numerical tolerance.

Run with: ``python scripts/verify_xa_c16x_equivalence.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conformer_asr.downsamplers.conv2d import Conv2dDownsampler
from conformer_asr.downsamplers.cross_attn import CrossAttnDownsampler


N_MELS = 80
HIDDEN = 256
ATOL = 1e-5


def _conv2d_layers(stack: nn.Sequential) -> list[nn.Conv2d]:
    return [m for m in stack if isinstance(m, nn.Conv2d)]


def _copy_weights(src: Conv2dDownsampler, dst: CrossAttnDownsampler) -> None:
    """Copy every learned weight from c16x into xa, layer-for-layer."""
    src_convs = _conv2d_layers(src.convs)
    dst_convs = _conv2d_layers(dst.convs)
    assert len(src_convs) == len(dst_convs) == 4, (
        f"expected 4 conv layers each, got src={len(src_convs)} dst={len(dst_convs)}"
    )
    for i, (sc, dc) in enumerate(zip(src_convs, dst_convs)):
        assert sc.weight.shape == dc.weight.shape, (
            f"conv layer {i} weight shape mismatch: {sc.weight.shape} vs {dc.weight.shape}"
        )
        with torch.no_grad():
            dc.weight.copy_(sc.weight)
            assert (sc.bias is None) == (dc.bias is None)
            if sc.bias is not None:
                dc.bias.copy_(sc.bias)
    with torch.no_grad():
        assert src.proj.weight.shape == dst.proj.weight.shape
        dst.proj.weight.copy_(src.proj.weight)
        dst.proj.bias.copy_(src.proj.bias)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def main() -> int:
    torch.manual_seed(0)

    c16x = Conv2dDownsampler(
        n_mels=N_MELS,
        hidden=HIDDEN,
        strides=[[2, 2], [2, 2], [2, 1], [2, 1]],
        dropout=0.0,
    )
    xa = CrossAttnDownsampler(
        n_mels=N_MELS,
        hidden=HIDDEN,
        dropout=0.0,
        strides=[[2, 2], [2, 2]],
        dilations=[1, 2],
        stride=4,
        num_layers=0,
    )

    _copy_weights(c16x, xa)
    c16x.eval()
    xa.eval()

    # Output-length arithmetic must match before we even look at tensors.
    lens_in = torch.tensor([401, 351, 257], dtype=torch.long)
    lens_c = c16x.output_lengths(lens_in)
    lens_x = xa.output_lengths(lens_in)
    assert torch.equal(lens_c, lens_x), f"length mismatch: c16x={lens_c.tolist()} xa={lens_x.tolist()}"
    print(f"output_lengths({lens_in.tolist()}) -> {lens_c.tolist()} (match)")

    # Forward.
    B, T = 2, 401
    x = torch.randn(B, T, N_MELS, dtype=torch.float64)
    # Cast modules to float64 for tight tolerance — float32 conv accumulators
    # otherwise drift by ~1e-6 per layer, which is honest noise but obscures
    # the equivalence we want to demonstrate.
    c16x = c16x.double()
    xa = xa.double()

    y_c = c16x(x)
    y_x = xa(x, input_lengths=torch.full((B,), T, dtype=torch.long))
    assert y_c.shape == y_x.shape, f"shape mismatch: {y_c.shape} vs {y_x.shape}"
    fwd_diff = _max_abs(y_c, y_x)
    print(f"forward shapes: {tuple(y_c.shape)}; max |y_c - y_x| = {fwd_diff:.3e}")
    assert fwd_diff < ATOL, f"forward outputs differ by {fwd_diff:.3e} (atol={ATOL})"

    # Backward — same scalar loss against same target on both networks.
    target = torch.randn_like(y_c)
    c16x.zero_grad(set_to_none=True)
    xa.zero_grad(set_to_none=True)

    loss_c = ((c16x(x) - target) ** 2).mean()
    loss_x = ((xa(x, input_lengths=torch.full((B,), T, dtype=torch.long)) - target) ** 2).mean()
    loss_diff = abs(loss_c.item() - loss_x.item())
    print(f"loss c16x={loss_c.item():.10f} xa={loss_x.item():.10f}  diff={loss_diff:.3e}")
    assert loss_diff < ATOL, f"loss differs by {loss_diff:.3e}"

    loss_c.backward()
    loss_x.backward()

    src_convs = _conv2d_layers(c16x.convs)
    dst_convs = _conv2d_layers(xa.convs)
    for i, (sc, dc) in enumerate(zip(src_convs, dst_convs)):
        wd = _max_abs(sc.weight.grad, dc.weight.grad)
        bd = _max_abs(sc.bias.grad, dc.bias.grad)
        print(f"conv[{i}] grad diff: weight={wd:.3e}  bias={bd:.3e}")
        assert wd < ATOL, f"conv[{i}] weight grad differs by {wd:.3e}"
        assert bd < ATOL, f"conv[{i}] bias grad differs by {bd:.3e}"

    pwd = _max_abs(c16x.proj.weight.grad, xa.proj.weight.grad)
    pbd = _max_abs(c16x.proj.bias.grad, xa.proj.bias.grad)
    print(f"proj    grad diff: weight={pwd:.3e}  bias={pbd:.3e}")
    assert pwd < ATOL, f"proj weight grad differs by {pwd:.3e}"
    assert pbd < ATOL, f"proj bias grad differs by {pbd:.3e}"

    print("\nOK — c16x and xa (num_layers=0) are bit-exact equivalent on forward and gradients.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
