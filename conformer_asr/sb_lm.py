"""SpeechBrain TransformerLM loader + n-best rescorer for LibriSpeech.

Wraps ``speechbrain/asr-transformer-transformerlm-librispeech`` for use as an
external scorer on detokenized hypothesis strings. Intentionally string-in /
score-out so it composes with any ASR tokenizer (no vocab coupling).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

SB_LM_REPO = "speechbrain/asr-transformer-transformerlm-librispeech"
# Mirrors recipes/LibriSpeech/LM/hparams/transformer.yaml — the config the
# published ``lm.ckpt`` was trained with. Do not change unless swapping the
# checkpoint.
SB_LM_CONFIG = dict(
    vocab=5000,
    d_model=768,
    nhead=12,
    num_encoder_layers=12,
    num_decoder_layers=0,
    d_ffn=3072,
    dropout=0.0,
    normalize_before=False,
)
SB_LM_PAD = 0
SB_LM_BOS = 1
SB_LM_EOS = 2


@dataclass
class SBLMScorer:
    model: torch.nn.Module
    sp: "object"  # sentencepiece.SentencePieceProcessor
    device: torch.device

    @classmethod
    def from_hub(
        cls,
        repo: str = SB_LM_REPO,
        device: str | torch.device = "cpu",
        cache_dir: str | None = None,
    ) -> "SBLMScorer":
        try:
            from speechbrain.lobes.models.transformer.TransformerLM import TransformerLM
        except ImportError as e:
            raise ImportError(
                "speechbrain is required for SBLMScorer. Install with "
                "`uv pip install speechbrain`."
            ) from e
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError(
                "sentencepiece is required for SBLMScorer. Install with "
                "`uv pip install sentencepiece`."
            ) from e
        from huggingface_hub import snapshot_download

        local = Path(
            snapshot_download(
                repo_id=repo,
                cache_dir=cache_dir,
                allow_patterns=["lm.ckpt", "tokenizer.ckpt"],
            )
        )
        model = TransformerLM(activation=torch.nn.GELU, **SB_LM_CONFIG)
        state = torch.load(str(local / "lm.ckpt"), map_location="cpu")
        # SB checkpoints sometimes wrap the state dict under a top-level key.
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[sb_lm] load_state_dict missing={len(missing)} unexpected={len(unexpected)}"
            )
            if missing:
                print(f"[sb_lm] first 5 missing: {missing[:5]}")
            if unexpected:
                print(f"[sb_lm] first 5 unexpected: {unexpected[:5]}")
        model.eval().to(device)
        sp = spm.SentencePieceProcessor()
        sp.load(str(local / "tokenizer.ckpt"))
        return cls(model=model, sp=sp, device=torch.device(device))

    @torch.no_grad()
    def score_hypotheses(
        self,
        texts: Sequence[str],
        *,
        uppercase: bool = True,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Return sum-of-log-probs (``log p(text + eos | bos)``) per hypothesis.

        LibriSpeech transcripts are natively uppercase and SB's SentencePiece
        vocab was trained on uppercase text, so ``uppercase=True`` is the right
        default when hypotheses come from a lowercased ASR pipeline.
        """
        scored: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            ids_list = [
                [SB_LM_BOS]
                + self.sp.encode(t.upper() if uppercase else t, out_type=int)
                + [SB_LM_EOS]
                for t in chunk
            ]
            max_len = max(len(x) for x in ids_list)
            ids = torch.full((len(ids_list), max_len), SB_LM_PAD, dtype=torch.long)
            for i, x in enumerate(ids_list):
                ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
            ids = ids.to(self.device)

            # TransformerLM is teacher-forced: logits[:, t] predicts ids[:, t+1].
            logits = self.model(ids)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            target = ids[:, 1:]
            pred_lp = log_probs[:, :-1, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)
            mask = (target != SB_LM_PAD).float()
            scored.append((pred_lp * mask).sum(dim=1).cpu())
        return torch.cat(scored)
