"""Log-Mel spectrogram extraction for the Conformer AED frontend.

Produces the same shape-of-input as Whisper / SpeechBrain / NeMo / ESPnet:
80-dim log-Mel at 100 Hz (10 ms hop, 25 ms window) from a 16 kHz waveform.
Used both at dataset preprocessing time (``data.preprocess_dataset``) and
at eval time (``scripts/evaluate.py``) so training and inference see
identical features.

The mel filter bank is built once via ``transformers.audio_utils.mel_filter_bank``
(Slaney-style, matching librosa defaults). Magnitudes are clamped to 1e-10
before ``log`` (natural log) to avoid -inf on silent frames.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from transformers.audio_utils import mel_filter_bank


@lru_cache(maxsize=4)
def build_mel_filters(n_mels: int, n_fft: int, sampling_rate: int) -> np.ndarray:
    """Slaney-normalized mel filter bank of shape ``(n_mels, n_fft // 2 + 1)``.

    ``transformers.audio_utils.mel_filter_bank`` returns ``(n_freq, n_mels)``;
    we transpose so ``filters @ magnitudes`` gives ``(n_mels, T)`` directly.
    """
    filters = mel_filter_bank(
        num_frequency_bins=n_fft // 2 + 1,
        num_mel_filters=n_mels,
        min_frequency=0.0,
        max_frequency=sampling_rate / 2.0,
        sampling_rate=sampling_rate,
        norm="slaney",
        mel_scale="slaney",
    )
    return filters.T


def log_mel_spectrogram(
    waveform: np.ndarray | torch.Tensor,
    *,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    sampling_rate: int,
) -> torch.Tensor:
    """Compute a log-Mel spectrogram of shape ``(T_mel, n_mels)``.

    Input can be a 1-D numpy array or torch tensor. Output frame count is
    ``T_wave // hop_length`` (we drop the trailing STFT frame, matching
    Whisper's convention) so 1 s of audio at 16 kHz / hop 160 yields 100 frames.
    """
    if isinstance(waveform, np.ndarray):
        wav = torch.from_numpy(waveform).float()
    else:
        wav = waveform.float()
    if wav.dim() != 1:
        wav = wav.reshape(-1)

    window = torch.hann_window(n_fft, device=wav.device)
    stft = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    # Drop the last frame so T_mel = T_wave // hop_length exactly.
    magnitudes = stft[..., :-1].abs().pow(2)  # (n_freq, T_mel)

    filters = build_mel_filters(n_mels, n_fft, sampling_rate)
    filters_t = torch.as_tensor(filters, dtype=magnitudes.dtype, device=magnitudes.device)
    mel_spec = filters_t @ magnitudes  # (n_mels, T_mel)
    log_mel = torch.clamp(mel_spec, min=1e-10).log()

    return log_mel.transpose(0, 1).contiguous()  # (T_mel, n_mels)
