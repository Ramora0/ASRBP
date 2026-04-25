"""Minimal end-to-end smoke test — no LibriSpeech, no downloads.

Builds the model, wires up Seq2SeqTrainer with a tiny fake in-memory dataset,
and runs two train steps. Designed to isolate the two known interop issues
between ``SpeechEncoderDecoderModel`` + ``BartForCausalLM`` and recent
transformers / Trainer:

1. ``BartDecoder`` rejecting ``input_ids`` + ``inputs_embeds`` combo
   (handled by ``_CompatBartForCausalLM`` in ``conformer_asr/model.py``).
2. ``Trainer.compute_loss`` forwarding ``num_items_in_batch`` into the model
   (handled by ``trainer.model_accepts_loss_kwargs = False``).

Use this on any new environment (local, cluster node) before trying full
training — if this passes, the model / trainer plumbing is sound.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as BLPreTok
from tokenizers.processors import TemplateProcessing
from transformers import (
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from conformer_asr.config import load_config
from conformer_asr.data import DataCollatorSpeechSeq2SeqWithPadding
from conformer_asr.features import log_mel_spectrogram
from conformer_asr.model import build_model


def _make_dummy_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = BLPreTok(add_prefix_space=False)
    tok.add_special_tokens(["<pad>", "<s>", "</s>", "<unk>"])
    tok.add_tokens([chr(c) for c in range(32, 127)])
    tok.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tok.token_to_id("<s>")),
            ("</s>", tok.token_to_id("</s>")),
        ],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )


def _make_fake_dataset(n: int, tokenizer, mcfg) -> Dataset:
    rng = torch.Generator().manual_seed(0)
    records = []
    for _ in range(n):
        audio = torch.randn(16000, generator=rng).numpy()
        mel = log_mel_spectrogram(
            audio,
            n_mels=mcfg.n_mels,
            n_fft=mcfg.n_fft,
            hop_length=mcfg.hop_length,
            sampling_rate=16000,
        )
        ids = tokenizer("hello world").input_ids
        records.append(
            {
                "input_features": mel.tolist(),
                "input_length": int(mel.shape[0]),
                "labels": ids,
            }
        )
    return Dataset.from_list(records)


def main() -> None:
    import transformers

    print(f"transformers: {transformers.__version__}")
    print(f"torch:        {torch.__version__}")

    tokenizer = _make_dummy_tokenizer()

    mcfg = load_config(Path(__file__).resolve().parent.parent / "configs/cnns/c4x.yaml").model
    model = build_model(mcfg, tokenizer)

    train_ds = _make_fake_dataset(4, tokenizer, mcfg)
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        n_mels=mcfg.n_mels,
    )

    with tempfile.TemporaryDirectory() as tmp:
        args = Seq2SeqTrainingArguments(
            output_dir=tmp,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=2,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_cpu=not torch.cuda.is_available(),
            remove_unused_columns=False,
            # label_smoothing_factor > 0 makes Trainer pop `labels` from inputs
            # before calling model.forward — SpeechEncoderDecoderModel then can't
            # derive decoder_input_ids, and BartDecoder rejects the resulting
            # all-None call. Keep this ON here to catch that regression.
            label_smoothing_factor=0.1,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=collator,
        )
        trainer.model_accepts_loss_kwargs = False

        print("trainer.train()...")
        trainer.train()
        print("OK")


if __name__ == "__main__":
    main()
