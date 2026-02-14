"""Shared feature extraction for the SNN experiment.

Provides acoustic features, MFCCs, and pyannote embedding extraction
used by both training data generation and live inference.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.fftpack import dct

SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Build a mel-scale filterbank matrix."""
    fmin = 0.0
    fmax = sr / 2.0
    # Mel scale conversion
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


# Cache the filterbank
_MEL_FB = None


def _get_mel_fb():
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)
    return _MEL_FB


def extract_acoustic(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract 5 core acoustic features from audio chunk.

    Returns: [rms, max_amp, energy, zcr, spectral_centroid]
    """
    if len(audio) < 100:
        return np.zeros(5, dtype=np.float32)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    max_amp = float(np.max(np.abs(audio)))
    energy = float(np.sum(audio ** 2) / len(audio))

    zero_crossings = np.where(np.diff(np.sign(audio)))[0]
    zcr = float(len(zero_crossings) / len(audio))

    try:
        f, t, Sxx = scipy_signal.spectrogram(audio, sr)
        sc = np.sum(f[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-10)
        spectral_centroid = float(np.mean(sc))
    except Exception:
        spectral_centroid = 0.0

    return np.array([rms, max_amp, energy, zcr, spectral_centroid], dtype=np.float32)


def extract_mfccs(audio: np.ndarray, sr: int = SAMPLE_RATE,
                  n_mfcc: int = N_MFCC) -> np.ndarray:
    """Extract MFCCs from audio using scipy (no librosa dependency).

    Returns: [n_mfcc] mean MFCC coefficients across frames.
    """
    if len(audio) < N_FFT:
        return np.zeros(n_mfcc, dtype=np.float32)

    # Pre-emphasis
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Frame the signal
    num_frames = 1 + (len(emphasized) - N_FFT) // HOP_LENGTH
    if num_frames < 1:
        return np.zeros(n_mfcc, dtype=np.float32)

    frames = np.zeros((num_frames, N_FFT))
    for i in range(num_frames):
        start = i * HOP_LENGTH
        frames[i] = emphasized[start:start + N_FFT]

    # Apply Hann window
    frames *= np.hanning(N_FFT)

    # Power spectrum
    mag = np.abs(np.fft.rfft(frames, N_FFT))
    power = (mag ** 2) / N_FFT

    # Mel filterbank
    mel_fb = _get_mel_fb()
    mel_spec = np.dot(power, mel_fb.T)
    mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)
    log_mel = np.log(mel_spec)

    # DCT to get MFCCs
    mfccs = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    # Return mean across frames
    return np.mean(mfccs, axis=0).astype(np.float32)


def extract_features_basic(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract features for real-time use (no embedding model needed).

    Returns: [21] = 5 acoustic + 13 MFCC + 3 reserved zeros
    """
    acoustic = extract_acoustic(audio, sr)
    mfccs = extract_mfccs(audio, sr)
    reserved = np.zeros(3, dtype=np.float32)
    return np.concatenate([acoustic, mfccs, reserved])


def extract_features_full(audio: np.ndarray, sr: int, embedding: np.ndarray,
                          pca_model) -> np.ndarray:
    """Extract full feature vector including PCA-compressed embedding.

    Args:
        audio: Audio samples
        sr: Sample rate
        embedding: Raw pyannote embedding (192-dim)
        pca_model: Fitted sklearn PCA model

    Returns: [37] = 5 acoustic + 13 MFCC + 3 reserved + 16 PCA embedding
    """
    acoustic = extract_acoustic(audio, sr)
    mfccs = extract_mfccs(audio, sr)
    reserved = np.zeros(3, dtype=np.float32)

    # PCA compress embedding
    emb_compressed = pca_model.transform(embedding.reshape(1, -1))[0].astype(np.float32)

    return np.concatenate([acoustic, mfccs, reserved, emb_compressed])


def extract_pyannote_embedding(audio: np.ndarray, sr: int,
                               inference) -> np.ndarray:
    """Extract speaker embedding using pyannote's embedding model.

    Args:
        audio: Audio samples (float32, mono)
        sr: Sample rate
        inference: pyannote Inference object

    Returns: embedding vector (typically 192-dim or 256-dim)
    """
    import torch
    # pyannote expects {"waveform": tensor [1, samples], "sample_rate": int}
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    embedding = inference({"waveform": waveform, "sample_rate": sr})
    return embedding.flatten()
