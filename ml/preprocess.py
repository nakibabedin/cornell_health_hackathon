"""
Audio preprocessing: resample, trim silence, normalize amplitude.
Used for live inference when users upload WebM/Opus recordings from the browser.
"""

import subprocess
import numpy as np
import librosa


def convert_webm_to_wav(
    input_path: str,
    output_path: str,
    target_sr: int = 16000,
) -> str:
    """Convert WebM/Opus (browser recording) to 16kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr),
        "-ac", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
    return output_path


def load_and_preprocess(
    wav_path: str,
    target_sr: int = 16000,
    trim_silence: bool = True,
    top_db: int = 25,
    normalize: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load a WAV file and apply preprocessing.

    Steps:
        1. Load and resample to target_sr
        2. Trim leading/trailing silence
        3. Normalize amplitude to [-1, 1]

    Raises ValueError if audio is too short (< 1s after trimming).
    """
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)

    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=top_db)

    if normalize and np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    duration = len(y) / sr
    if duration < 1.0:
        raise ValueError(
            f"Audio too short after preprocessing: {duration:.2f}s (minimum 1.0s)"
        )

    return y, sr


def validate_audio(
    y: np.ndarray,
    sr: int,
    min_duration: float = 3.0,
    max_duration: float = 30.0,
    min_rms: float = 0.01,
) -> dict:
    """
    Validate audio quality for feature extraction.

    Returns dict with: valid (bool), duration (float), rms (float), errors (list[str]).
    """
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y**2)))
    errors = []

    if duration < min_duration:
        errors.append(f"Recording too short: {duration:.1f}s (minimum {min_duration}s)")
    if duration > max_duration:
        errors.append(f"Recording too long: {duration:.1f}s (maximum {max_duration}s)")
    if rms < min_rms:
        errors.append(f"Recording too quiet: RMS={rms:.4f} (minimum {min_rms})")

    return {
        "valid": len(errors) == 0,
        "duration": round(duration, 2),
        "rms": round(rms, 4),
        "errors": errors,
    }
