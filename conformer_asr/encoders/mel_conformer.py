"""Log-Mel + pluggable-downsampler Conformer encoder.

Takes a ``(B, T_mel, n_mels)`` log-Mel spectrogram (see
``conformer_asr/features.py``), runs it through:

  1. ``InputNormalization`` — per-bin running mean/var, updated for the
     first N epochs then frozen (see ``encoders/preproc.py``).
  2. ``SpecAugment`` — pre-stem deterministic K time + K feature masks
     with zero-fill (see ``encoders/preproc.py``). No-op outside training
     or before ``spec_aug_warmup_steps``.
  3. A swappable ``Downsampler`` (defaults to 2× ``Conv2d(k=3, s=2)``).
  4. The HF ``Wav2Vec2ConformerEncoder`` blocks, optionally interleaved
     with ``CrossAttnBlock`` taps that re-attend the residual stream to
     the downsampler's cached post-CNN feature map (see
     ``cross_attn_layer_indices``). When the index list is empty the
     stock encoder is used as-is.

Slots into ``SpeechEncoderDecoderModel`` by exposing:
  - ``.config.hidden_size`` (inherited from ``Wav2Vec2ConformerConfig``)
  - ``forward(input_features, attention_mask=None, ...)`` returning a
    ``BaseModelOutput`` (the wrapper reads ``encoder_outputs[0]``).
  - ``_get_feature_vector_attention_mask(feat_len, attention_mask, ...)``
    which translates the pre-stem mask through the downsampler's time
    arithmetic so the transformer only attends to valid positions.

The downsampler and spec-augment modules are external state — pass them
into ``__init__`` (normally via ``build_encoder``). The encoder doesn't
care which concrete classes they are, just the interface contracts.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from transformers import Wav2Vec2ConformerConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerEncoder,
    Wav2Vec2ConformerPreTrainedModel,
)

from ..downsamplers.base import Downsampler
from ..downsamplers.cross_attn import CrossAttnBlock, SharedKVProjector
from .preproc import InputNormalization, SpecAugment
from .sdpa_patch import install_sdpa_attention_patch

install_sdpa_attention_patch()


class MelConformerEncoder(Wav2Vec2ConformerPreTrainedModel):
    """Mel-input Conformer encoder with a pluggable ``Downsampler`` stem."""

    def __init__(
        self,
        config: Wav2Vec2ConformerConfig,
        downsampler: Downsampler,
        spec_augment: SpecAugment,
        *,
        cross_attn_layer_indices: Sequence[int] = (),
        cross_attn_num_heads: int = 4,
        cross_attn_dropout: float = 0.0,
    ):
        super().__init__(config)
        self.n_mels = int(config.n_mels)

        self.input_norm = InputNormalization(n_features=self.n_mels)
        self.spec_augment = spec_augment
        self.downsampler = downsampler
        self.encoder = Wav2Vec2ConformerEncoder(config)

        # Interleaved cross-attn taps. Sorted + de-duplicated so the manual
        # encoder loop can iterate the block list in lockstep with the
        # conformer-layer index. Empty list ⇒ stock-encoder fast path.
        idx = sorted({int(i) for i in cross_attn_layer_indices})
        if idx and (idx[0] < 1 or idx[-1] > config.num_hidden_layers):
            raise ValueError(
                f"cross_attn_layer_indices must lie in [1, {config.num_hidden_layers}], "
                f"got {idx}"
            )
        self.cross_attn_layer_indices: tuple[int, ...] = tuple(idx)
        if idx:
            # ``_post_cnn_features`` is set at forward time, not __init__ —
            # gate on the static signal (``post_cnn_lengths`` method) which
            # only the cross_attn downsampler exposes today.
            if not hasattr(self.downsampler, "post_cnn_lengths"):
                raise ValueError(
                    f"Interleaved cross-attention requires a downsampler that "
                    f"exposes a post-CNN K/V cache; "
                    f"{type(self.downsampler).__name__} does not."
                )
            # K/V are computed once per encoder forward by the shared
            # projector and broadcast to every interleaved tap. See
            # ``SharedKVProjector`` for the rationale (~80% per-tap compute
            # lives in K/V projections at T_k = 4 × T_q).
            self.shared_kv_projector = SharedKVProjector(
                hidden=config.hidden_size,
                num_heads=cross_attn_num_heads,
            )
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttnBlock(
                    hidden=config.hidden_size,
                    num_heads=cross_attn_num_heads,
                    dropout=cross_attn_dropout,
                    attn_dropout=cross_attn_dropout,
                )
                for _ in idx
            ])
        else:
            self.shared_kv_projector = None
            self.cross_attn_blocks = nn.ModuleList()

        self.post_init()
        # post_init() recurses ``_init_weights`` over every submodule, which
        # silently overwrites any custom parameter init the downsampler did
        # in its own __init__ (e.g. biasing the boundary predictor toward a
        # target prior). Give the downsampler a chance to restore it.
        self.downsampler.post_parent_init()
        # Same hazard for the cross-attn blocks: ``CrossAttnBlock.__init__``
        # zero-inits ``out_proj`` so the first forward is a residual
        # passthrough, but ``_init_weights`` re-fills the linear with a
        # normal init and erases that. Restore.
        for block in self.cross_attn_blocks:
            nn.init.zeros_(block.out_proj.weight)
            nn.init.zeros_(block.out_proj.bias)

    def _get_feature_vector_attention_mask(
        self,
        feature_vector_length: int,
        attention_mask: torch.LongTensor,
        add_adapter=None,
    ) -> torch.BoolTensor:
        """Build a ``(B, feature_vector_length)`` **bool** mask from the
        pre-stem ``(B, T_mel)`` mask using the downsampler's time arithmetic.

        Bool dtype matches what the parent ``Wav2Vec2ConformerPreTrainedModel``
        returns — ``Wav2Vec2ConformerEncoder.forward`` does ``~attention_mask``
        which only gives the right (logical-NOT) answer on a bool tensor; on a
        long tensor ``~1`` is ``-2`` which is either an out-of-bounds or
        silently-wrapping negative index. ``add_adapter`` is accepted for
        interface compatibility and ignored.
        """
        input_lengths = attention_mask.sum(-1)
        output_lengths = self.downsampler.output_lengths(input_lengths)
        arange = torch.arange(feature_vector_length, device=attention_mask.device)
        return arange[None, :] < output_lengths[:, None]

    def _get_post_cnn_attention_mask(
        self,
        post_cnn_length: int,
        attention_mask: torch.LongTensor,
    ) -> torch.BoolTensor:
        """Bool mask for the **post-CNN / pre-pick** stage of the downsampler.

        Used when an outer CTC head taps the downsampler before its picking
        / pooling stage (``ctc_input='post_cnn'``). The downsampler must
        expose ``post_cnn_lengths(input_lengths)``; only ``CrossAttnDownsampler``
        does so today.
        """
        if not hasattr(self.downsampler, "post_cnn_lengths"):
            raise RuntimeError(
                f"downsampler {type(self.downsampler).__name__} does not "
                "expose 'post_cnn_lengths'; ctc_input='post_cnn' is only "
                "supported with downsamplers that have a non-CNN compression "
                "stage (e.g. cross_attn)."
            )
        input_lengths = attention_mask.sum(-1)
        post_cnn_lens = self.downsampler.post_cnn_lengths(input_lengths)
        arange = torch.arange(post_cnn_length, device=attention_mask.device)
        return arange[None, :] < post_cnn_lens[:, None]

    def forward(
        self,
        input_features: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Per-bin normalization (stats update during training until frozen).
        # 2. Pre-stem SpecAugment (training only, zero-fill after normalization
        #    so masked regions land on the per-bin mean).
        x = self.input_norm(input_features, attention_mask=attention_mask)
        x = self.spec_augment(x, attention_mask=attention_mask)

        # 3. Downsampler stem: (B, T_mel, n_mels) → (B, T', hidden).
        # Dynamic downsamplers (e.g. boundary predictor) need per-sample valid
        # lengths to mask padding before pooling — pass them through; static
        # downsamplers ignore the kwarg.
        input_lengths = attention_mask.sum(-1) if attention_mask is not None else None
        x = self.downsampler(x, input_lengths=input_lengths)
        t_out = x.shape[1]

        # Stash the post-downsampler tensor so an outer wrapper (e.g.
        # ``ConformerAEDWithCTC`` with ``ctc_input="features"``) can attach a
        # head to the conv-stem output without re-running it. Plain attribute
        # (not a buffer/parameter) so it stays out of state_dict.
        self._features_for_ctc = x

        # Downsample the pre-stem attention mask so the Conformer blocks see
        # the correct (B, T') mask for self-attention.
        encoder_attention_mask = None
        if attention_mask is not None:
            encoder_attention_mask = self._get_feature_vector_attention_mask(t_out, attention_mask)

        if not self.cross_attn_blocks:
            return self.encoder(
                x,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return self._encoder_with_xa(
            hidden_states=x,
            attention_mask=encoder_attention_mask,
            pre_stem_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def _encoder_with_xa(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor | None,
        pre_stem_attention_mask: torch.LongTensor | None,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
    ):
        """Manual conformer loop with cross-attn taps injected after each
        index in ``self.cross_attn_layer_indices`` (1-indexed).

        Replicates ``Wav2Vec2ConformerEncoder.forward`` (HF transformers
        v4.x) — input dropout, rotary positional embeddings, per-layer
        layerdrop, final ``layer_norm`` — and drops a ``CrossAttnBlock``
        in between layers when their (1-indexed) position matches.

        The XA Q/K positions are taken from the *original post-CNN
        timeline* so RoPE phases match the true frame-rate gap between a
        picked query at conformer index ``i`` and a key at post-CNN index
        ``j``. K/V is the downsampler's cached ``_post_cnn_features``.
        """
        encoder = self.encoder

        # K/V tensor and its (B, T_k) bool padding mask. The downsampler set
        # ``_post_cnn_features`` on its forward call earlier in this same
        # forward — read it here without re-running the conv stack.
        kv = self.downsampler._post_cnn_features
        kv_mask: torch.Tensor | None = None
        if pre_stem_attention_mask is not None:
            kv_mask = self._get_post_cnn_attention_mask(kv.shape[1], pre_stem_attention_mask)
            kv_mask = kv_mask[:, None, None, :]  # SDPA shape (B, 1, 1, T_k)

        # Q/K position indices in post-CNN coordinates.
        #   - Static downsamplers (cross_attn): Q at ``[0, stride, 2*stride, ...]``.
        #   - Variable selector (bp_xa): downsampler stashes the per-sample
        #     picked indices on ``_kept_indices`` (shape ``(B, T_q)``); we
        #     pass those straight through, so each query's RoPE phase
        #     matches the actual post-CNN frame it was picked from.
        T_k = kv.shape[1]
        T_q = hidden_states.shape[1]
        device = hidden_states.device
        kept_indices = getattr(self.downsampler, "_kept_indices", None)
        if kept_indices is not None:
            q_positions = kept_indices  # (B, T_q)
        else:
            stride = int(getattr(self.downsampler, "stride", 1))
            q_positions = torch.arange(0, T_q * stride, stride, device=device)[:T_q]
        k_positions = torch.arange(T_k, device=device)

        # Shared K/V: project + RoPE-rotate once for the whole encoder pass,
        # then reuse across every interleaved tap below. The K/V source and
        # k_positions are identical across taps, so this is a strict compute
        # win with no modeling-capacity loss on the K/V side (per-tap views
        # are still expressible via each block's own query function).
        K_shared, V_shared = self.shared_kv_projector(kv, k_positions)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # Match HF's mask handling: zero out padded tokens, then build the
        # additive (B, 1, T_q, T_q) extended mask self-attention expects.
        ext_attn_mask: torch.Tensor | None = None
        if attention_mask is not None:
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(
                1, 1, hidden_states.shape[2]
            )
            hidden_states = hidden_states.masked_fill(~expand_attention_mask, 0.0)
            ext_attn_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            ext_attn_mask = ext_attn_mask * torch.finfo(hidden_states.dtype).min
            ext_attn_mask = ext_attn_mask.expand(
                ext_attn_mask.shape[0], 1, ext_attn_mask.shape[-1], ext_attn_mask.shape[-1]
            )

        hidden_states = encoder.dropout(hidden_states)
        rel_pos_emb = (
            encoder.embed_positions(hidden_states) if encoder.embed_positions is not None else None
        )

        # 1-indexed → 0-indexed lookup of "which XA block fires after layer i".
        xa_after = {idx - 1: i for i, idx in enumerate(self.cross_attn_layer_indices)}

        # Reference deltas: how much the conformer layer immediately preceding
        # each XA tap shifts the residual stream. Logged alongside XA's own
        # delta so the per-tap ratio is self-calibrated against the layer's
        # own contribution rather than a hand-picked baseline. ``None``
        # entries mean the reference layer was layer-dropped this step —
        # capturing 0 instead would skew the accumulator's mean toward 0.
        n_taps = len(self.cross_attn_layer_indices)
        ref_layer_delta_rms: list[float | None] = [None] * n_taps

        for i, layer in enumerate(encoder.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            dropout_probability = torch.rand([])
            skip_the_layer = self.training and dropout_probability < encoder.config.layerdrop
            h_before_layer = hidden_states if i in xa_after else None
            if not skip_the_layer:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=ext_attn_mask,
                    relative_position_embeddings=rel_pos_emb,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
            else:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if i in xa_after:
                tap = xa_after[i]
                # Reference: the conformer layer's residual contribution at
                # this depth. Skip when layer-dropped — the "delta" would be
                # zero (layer didn't run), which would bias the running
                # mean toward 0 and produce garbage relative-strength ratios.
                if not skip_the_layer:
                    with torch.no_grad():
                        ref_delta = (hidden_states - h_before_layer).detach().float()
                        ref_layer_delta_rms[tap] = float(ref_delta.pow(2).mean().sqrt().item())
                block = self.cross_attn_blocks[tap]
                hidden_states = block(hidden_states, K_shared, V_shared, q_positions, kv_mask)

        self._last_ref_layer_delta_rms = ref_layer_delta_rms

        hidden_states = encoder.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def last_stats(self) -> dict | None:
        """Per-XA-tap residual-magnitude snapshot from the most-recent
        ``forward``. Returns ``None`` when no interleaved XA blocks are
        configured or before the first forward.

        Trainer's stat accumulator branches on key presence, so the
        ``xa_*`` namespace stays disjoint from BP's
        ``n_boundaries`` / ``aux_loss`` keys.

        Includes a ``ref_*`` reference: the residual delta contributed by
        the conformer layer immediately preceding each tap. Logging XA
        and the layer side by side makes the per-tap ratio self-
        calibrating — XA's contribution as a fraction of the layer's own
        contribution at the same depth.
        """
        if not self.cross_attn_blocks:
            return None
        deltas = [getattr(b, "_last_delta_rms", None) for b in self.cross_attn_blocks]
        q_rms = [getattr(b, "_last_q_rms", None) for b in self.cross_attn_blocks]
        if any(v is None for v in deltas):
            return None
        return {
            "xa_delta_rms": deltas,
            "xa_q_rms": q_rms,
            "xa_ref_layer_delta_rms": list(getattr(self, "_last_ref_layer_delta_rms", [])),
        }
