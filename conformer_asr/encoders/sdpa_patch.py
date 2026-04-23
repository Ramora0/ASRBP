"""SDPA fast-path monkey-patch for ``Wav2Vec2ConformerSelfAttention``.

HF's ``Wav2Vec2ConformerSelfAttention`` has no SDPA dispatch path — it always
materializes the full ``(B, H, T, T)`` attention matrix via eager matmul /
softmax. For ``T ~ 1000`` on V100 that's a serious HBM-bandwidth bottleneck.
We route the rotary (and non-positional) path through
``F.scaled_dot_product_attention`` so the mem-efficient kernel can tile the
attention. The Shaw-relative path is left alone because it needs custom score
composition SDPA can't express.

Applying the patch is idempotent: ``install_sdpa_attention_patch`` only
monkey-patches once per process.
"""
from __future__ import annotations

import math

import torch.nn.functional as F
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerSelfAttention,
)


_ORIG_CONFORMER_SELF_ATTN_FORWARD = Wav2Vec2ConformerSelfAttention.forward
_PATCH_INSTALLED = False


def _sdpa_self_attn_forward(
    self,
    hidden_states,
    attention_mask=None,
    relative_position_embeddings=None,
    output_attentions=False,
):
    # Fall back to eager when the caller wants attention weights (SDPA doesn't
    # surface them) or when we're on the Shaw-relative path.
    if output_attentions or self.position_embeddings_type == "relative":
        return _ORIG_CONFORMER_SELF_ATTN_FORWARD(
            self, hidden_states, attention_mask, relative_position_embeddings, output_attentions
        )

    batch_size, _, _ = hidden_states.size()
    query_key_states = hidden_states
    value_states = hidden_states

    if self.position_embeddings_type == "rotary":
        if relative_position_embeddings is None:
            raise ValueError(
                "`relative_position_embeddings` required when position_embeddings_type == 'rotary'"
            )
        query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)

    query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
    key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
    value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=self.dropout.p if self.training else 0.0,
        scale=1.0 / math.sqrt(self.head_size),
    )
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
    attn_output = self.linear_out(attn_output)
    return attn_output, None


def install_sdpa_attention_patch() -> None:
    """Monkey-patch ``Wav2Vec2ConformerSelfAttention.forward`` at most once."""
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return
    Wav2Vec2ConformerSelfAttention.forward = _sdpa_self_attn_forward
    _PATCH_INSTALLED = True
