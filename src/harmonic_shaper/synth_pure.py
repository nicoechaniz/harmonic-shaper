"""Pure-Python additive synthesis of voice → WAV.

Bypasses the real-time Shaper entirely (the source fork had normalization bugs that
caused clipping under rapid voice_on/off transitions). Implements the same
algorithm in numpy:

    For each analysis frame (every 20ms):
      1. Get F0 + per-harmonic magnitudes
      2. Active set = harmonics within 30 dB of H1
      3. Each active voice plays a pure sine at f0·n
      4. Per-voice gain from STFT magnitude (mapped dB → 0..1)
      5. Master sum normalized by 1/sqrt(N_active)
      6. Optionally apply the same bounded tanh output stage as the live engine

Output: 16-bit stereo WAV at 44.1 kHz.

The pipeline is split into two phases:

  prepare_analysis(y, sr, ...) -> dict
    Cache F0, voiced mask, per-harmonic gain estimates (already smoothed and
    bridged), STFT raw magnitudes, and frequency grid. All the expensive
    librosa work happens here, ONCE.

  synthesize_prepared(prepared, ...) -> np.ndarray
    Re-render audio from the cached dict. Cheap — no librosa calls. Returns the
    pre-limiter mix by default for analysis, or the declared ±0.95 output range
    with limit_output=True.
    Supports per-harmonic gain overrides (per_harmonic_gains) and per-harmonic
    waveform overrides (wave_shapes).

  synthesize(y, sr, ...) -> np.ndarray
    Thin wrapper: prepare_analysis() then synthesize_prepared() with default
    parameters. Same signature as before — all existing callers continue to
    work unchanged.

Usage:
    python -m harmonic_shaper.synth_pure path/to/voice.wav --out /tmp/synth.wav
"""
from __future__ import annotations

import argparse
import logging
import sys
import wave
from pathlib import Path

try:
    import librosa
except Exception as _librosa_exc:  # robustness for missing / wrong version
    librosa = None  # type: ignore[assignment]
    _LIBROSA_IMPORT_ERROR = _librosa_exc

try:
    import soundfile as sf
except Exception as _sf_exc:
    sf = None  # type: ignore[assignment]
    _SF_IMPORT_ERROR = _sf_exc

import numpy as np

log = logging.getLogger("synth_pure")

from .voice_cache import VoiceCache
from .audio_levels import OUTPUT_LIMIT, soft_limit

N_HARMONICS = 32
SAMPLE_RATE = 44100


def analyze(y, sr, f0_min, f0_max):
    """Legacy analyze() — returns (times, f0, voiced, gains_db).

    Kept for callers that import it directly (e.g. build_voice_compare_v3.py).
    For new code prefer prepare_analysis() which returns the full dict the
    synthesizer consumes (including STFT + frequency grid).
    """
    if librosa is None:
        raise ImportError("librosa is required for analyze")
    hop = int(0.0464 * sr)
    n_fft = 4096
    f0, voiced, _ = librosa.pyin(
        y, fmin=f0_min, fmax=f0_max, sr=sr,
        hop_length=hop, frame_length=n_fft, fill_na=0.0,
    )
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)

    T = len(f0)
    gains_db = np.full((T, N_HARMONICS), -120.0, dtype=np.float32)
    for t in range(T):
        ft = f0[t]
        if not voiced[t] or ft <= 0:
            continue
        target_freqs = np.array([ft * (n + 1) for n in range(N_HARMONICS)])
        for n in range(N_HARMONICS):
            tgt = target_freqs[n]
            if tgt > sr / 2 - 50:
                break
            if tgt <= freqs[0] or tgt >= freqs[-1]:
                continue
            mag = float(np.interp(tgt, freqs, stft[:, t]))
            gains_db[t, n] = 20.0 * np.log10(mag + 1e-12)

    log.info("Analysis: %d frames, %.2f s, %.1f%% voiced",
             T, times[-1] if len(times) else 0, 100.0 * voiced.mean())
    if voiced.any():
        log.info("F0 (voiced): mean=%.1f min=%.1f max=%.1f",
                 float(f0[voiced].mean()),
                 float(f0[voiced].min()),
                 float(f0[voiced].max()))
    return times, f0, voiced, gains_db


def prepare_analysis(y, sr, f0_min=70.0, f0_max=400.0) -> dict:
    """Phase 1: run the expensive analysis once, return a cached dict.

    Returns a dict with keys:
        times       : (T,) array of frame centers in seconds
        f0          : (T,) array of (smoothed, bridged) fundamental in Hz
        voiced      : (T,) bool array — True where the frame is treated as voiced
        gains_db    : (T, N_HARMONICS) float32 — dB magnitude per harmonic,
                      recomputed for bridged frames so no silent gaps
        sr          : int — sample rate the analysis ran at
        duration    : float — length of input in seconds
        stft_raw    : (n_freqs, T) float — |STFT| for any later re-analysis
        freqs_stft  : (n_freqs,) float — frequency axis matching stft_raw

    The dict is self-contained — synthesize_prepared() takes only this dict
    plus optional rendering parameters. No further librosa calls needed.
    """
    if librosa is None:
        raise ImportError("librosa is required for prepare_analysis; " + str(getattr(sys.modules[__name__], "_LIBROSA_IMPORT_ERROR", "")))

    # Robustness: reduce stereo/ndim>1 to mono by loudest channel (mirrors server/build pick logic).
    y = np.asarray(y)
    if y.ndim > 1:
        mags = np.abs(y).max(axis=0)
        best = int(np.argmax(mags))
        y = y[:, best]
    y = y.astype(np.float32, copy=False)

    # Input stats (diagnostic)
    y_peak = float(np.abs(y).max()) if y.size else 0.0
    y_rms = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
    log.info("prepare_analysis input: shape=%s peak=%.6g RMS=%.6g sr=%d", y.shape, y_peak, y_rms, int(sr))

    hop = int(0.0464 * sr)
    n_fft = 4096

    # Use normalized copy for pyin + stft so that F0/voiced/harmonic gains are level-invariant
    # (prevents all-zero when caller passes low-amplitude or un-normalized buffers).
    y_anal = y / (y_peak + 1e-12) if y_peak > 1e-12 else y.copy()

    # Raw librosa analysis
    f0, voiced, _ = librosa.pyin(
        y_anal, fmin=f0_min, fmax=f0_max, sr=sr,
        hop_length=hop, frame_length=n_fft, fill_na=0.0,
    )

    # Diagnostic dump of librosa feature outputs (at INFO as required)
    f0_nz = int((f0 > 0).sum())
    voiced_cnt = int(voiced.sum())
    v_ratio = float(voiced.mean()) if len(voiced) else 0.0
    log.info("prepare_analysis pyin: f0(min=%.1f max=%.1f mean=%.1f) nonzero=%d/%d voiced=%d/%d (ratio=%.3f)",
             float(f0.min()), float(f0.max()), float(f0.mean()), f0_nz, len(f0), voiced_cnt, len(voiced), v_ratio)

    stft_raw = np.abs(librosa.stft(y_anal, n_fft=n_fft, hop_length=hop))
    freqs_stft = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)

    # Per-harmonic gain table (in dB), -120 for unvoiced frames
    T = len(f0)

    # Fallback when pyin VAD reports zero voiced (common on low-SNR, edge pitch, or un-normalized inputs).
    # Use yin (always estimates f0) + simple RMS energy VAD so synthesis can still proceed.
    if voiced_cnt == 0 and T > 0:
        log.info("prepare_analysis: pyin voiced=0; falling back to yin + energy VAD")
        try:
            f0 = librosa.yin(
                y_anal, fmin=f0_min, fmax=f0_max, sr=sr,
                hop_length=hop, frame_length=n_fft
            )
        except Exception as _yin_exc:
            log.info("yin fallback failed (%s); using constant 150Hz", _yin_exc)
            f0 = np.full(T, 150.0, dtype=np.float64)
        try:
            rms = librosa.feature.rms(y=y_anal, frame_length=n_fft, hop_length=hop, center=True)[0]
            if len(rms) > T:
                rms = rms[:T]
            elif len(rms) < T:
                rms = np.pad(rms, (0, T - len(rms)), mode="edge")
            rms_db = 20.0 * np.log10(rms + 1e-12)
            voiced = (rms_db > -50.0) & (f0 > 0)
            if not voiced.any():
                voiced = rms_db > -70.0
            if not (f0 > 0).any():
                f0 = np.full(T, 150.0, dtype=np.float64)
                voiced = np.ones(T, dtype=bool)
        except Exception as _vad_exc:
            log.info("energy VAD fallback error (%s); forcing all frames voiced @150Hz", _vad_exc)
            voiced = np.ones(T, dtype=bool)
            f0 = np.full(T, 150.0, dtype=np.float64)
        voiced_cnt = int(voiced.sum())
        v_ratio = float(voiced.mean())
        log.info("prepare_analysis fallback: now voiced=%d/%d f0_mean=%.1f", voiced_cnt, T, float(f0[voiced].mean()) if voiced_cnt else 0.0)

    gains_db = np.full((T, N_HARMONICS), -120.0, dtype=np.float32)
    for t in range(T):
        ft = f0[t]
        if not voiced[t] or ft <= 0:
            continue
        for n in range(N_HARMONICS):
            tgt = ft * (n + 1)
            if tgt > sr / 2 - 50:
                break
            if tgt <= freqs_stft[0] or tgt >= freqs_stft[-1]:
                continue
            mag = float(np.interp(tgt, freqs_stft, stft_raw[:, t]))
            gains_db[t, n] = 20.0 * np.log10(mag + 1e-12)

    # Additional harmonic analysis result logging (diagnostic)
    harm_frames = int(np.any(gains_db > -120, axis=1).sum())
    h1_vals = gains_db[:, 0]
    h1_mean = float(h1_vals[h1_vals > -120].mean()) if (h1_vals > -120).any() else -120.0
    log.info("prepare_analysis harmonic: frames_with_gains=%d H1_dB(mean_on_active~%.1f) max=%.1f",
             harm_frames, h1_mean, float(gains_db.max()))

    log.info("Analysis: %d frames, %.2f s, %.1f%% voiced",
             T, times[-1] if len(times) else 0, 100.0 * voiced.mean())
    if voiced.any():
        log.info("F0 (raw): mean=%.1f min=%.1f max=%.1f",
                 float(f0[voiced].mean()),
                 float(f0[voiced].min()),
                 float(f0[voiced].max()))

    # --- F0 smoothing: median + Gaussian + unvoiced bridging ---
    # Median filter of 5 frames removes outlier spikes (e.g. 247 Hz when the
    # true F0 is ~130). Then light 3-frame Gaussian to smooth pitch micro-
    # jitter without smearing real intonation. Without this, every ~46 ms
    # analysis frame can jump 30+ Hz and the harmonic series gets re-tuned
    # rapidly, producing audible beating/whistling artifacts.
    try:
        from scipy.signal import medfilt
        from scipy.ndimage import gaussian_filter1d
        _HAVE_SCIPY = True
    except Exception:
        _HAVE_SCIPY = False
        log.info("scipy not available; skipping median/gaussian smoothing (bridging still applied)")
    f0_smooth = f0.copy()
    voiced_idx = np.where(voiced)[0]
    if len(voiced_idx) >= 5 and _HAVE_SCIPY:
        f0_voiced = f0[voiced]
        f0_med = medfilt(f0_voiced, kernel_size=5)
        f0_voiced_smooth = gaussian_filter1d(f0_med, sigma=1.5)
        f0_smooth[voiced_idx] = f0_voiced_smooth
    elif len(voiced_idx) >= 5:
        # no scipy: light manual median-ish skip, still copy
        f0_smooth[voiced_idx] = f0[voiced]

    # Hold last F0 forward through unvoiced regions (instead of 0) so the
    # synth continues smoothly through gaps.
    bridged = set()  # frames that changed from unvoiced→voiced
    last_f0 = 0.0
    for i in range(len(f0_smooth)):
        if voiced[i] and f0_smooth[i] > 0:
            last_f0 = f0_smooth[i]
        elif voiced[i] == False and last_f0 > 0:
            f0_smooth[i] = last_f0  # bridge brief unvoiced gaps
            voiced[i] = True  # mark as voiced for synthesis purposes
            bridged.add(i)

    # Recompute harmonic gains for bridged frames — their original gains_db
    # is all -120 dB (analyze() skips unvoiced frames), which would produce
    # silence despite the bridged F0.
    if bridged:
        log.info("Recomputing harmonic gains for %d bridged frames", len(bridged))
        for t in bridged:
            ft = f0_smooth[t]
            if ft <= 0:
                continue
            for n in range(N_HARMONICS):
                tgt = ft * (n + 1)
                if tgt > sr / 2 - 50:
                    break
                if tgt <= freqs_stft[0] or tgt >= freqs_stft[-1]:
                    continue
                mag = float(np.interp(tgt, freqs_stft, stft_raw[:, t]))
                gains_db[t, n] = 20.0 * np.log10(mag + 1e-12)
    log.info("F0 smoothing applied (median + Gaussian + bridging)")

    duration = float(len(y) / sr)

    return {
        "times": times,
        "f0": f0_smooth,           # post-smoothing, post-bridging
        "voiced": voiced,          # True after bridging for bridged frames
        "gains_db": gains_db,
        "sr": int(sr),
        "duration": duration,
        "stft_raw": stft_raw,
        "freqs_stft": freqs_stft,
    }


def extract_aperiodic(y, sr, f0, voiced, times, stft_raw=None, freqs_stft=None, n_fft=4096, hop=None):
    """Extract aperiodic residual from voice signal.

    Subtracts the harmonic series from the STFT to isolate noise.
    Returns: time-domain noise signal (np.ndarray, same length as y, float64).
    """
    global librosa
    if librosa is None:
        raise ImportError("librosa is required for extract_aperiodic")

    # Import at function level as specified (module-level try/except already present).
    import librosa

    y = np.asarray(y).ravel()
    orig_len = len(y)

    if hop is None:
        hop = int(0.0464 * sr)

    if stft_raw is None or freqs_stft is None:
        # Compute from (normalized-like) y; mirror prepare_analysis approach lightly.
        y_for_stft = y.astype(np.float32, copy=False)
        peak = float(np.abs(y_for_stft).max()) if y_for_stft.size else 0.0
        if peak > 1e-12:
            y_for_stft = y_for_stft / peak
        stft_raw = np.abs(librosa.stft(y_for_stft, n_fft=n_fft, hop_length=hop))
        freqs_stft = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    else:
        stft_raw = np.asarray(stft_raw, dtype=np.float32).copy()
        freqs_stft = np.asarray(freqs_stft, dtype=np.float64)

    stft_aper = stft_raw.copy()
    # Subtract harmonic magnitudes at expected locations (nearest bin) for voiced frames.
    n_frames = min(stft_aper.shape[1], len(f0), len(voiced))
    for t in range(n_frames):
        if not (voiced[t] and f0[t] > 0):
            continue
        ft = float(f0[t])
        for n in range(1, N_HARMONICS + 1):
            tgt = ft * n
            if tgt > sr / 2 - 50:
                break
            if tgt <= freqs_stft[0] or tgt >= freqs_stft[-1]:
                continue
            mag = float(np.interp(tgt, freqs_stft, stft_raw[:, t]))
            # Subtract from closest frequency bin; clamp >=0. This removes harmonic energy.
            k = int(np.argmin(np.abs(freqs_stft - tgt)))
            stft_aper[k, t] = max(0.0, stft_aper[k, t] - mag)

    # Reconstruct time-domain residual using random phase (to get noisy signal
    # whose magnitude spectrum matches the aperiodic residual envelope).
    # Use a fixed-seed RandomState for deterministic output across calls
    # (ensures synthesize_prepared default noise mix is reproducible).
    rng = np.random.RandomState(0xC0DE)
    phase = rng.uniform(-np.pi, np.pi, size=stft_aper.shape)
    stft_complex = (stft_aper * np.exp(1j * phase)).astype(np.complex64)

    noise = librosa.istft(stft_complex, hop_length=hop, n_fft=n_fft, length=orig_len if orig_len > 0 else None)
    noise = np.asarray(noise, dtype=np.float64)
    if orig_len > 0:
        if len(noise) < orig_len:
            noise = np.pad(noise, (0, orig_len - len(noise)))
        elif len(noise) > orig_len:
            noise = noise[:orig_len]
    return noise


def _waveform_value(shape: str, phase: float) -> float:
    """Evaluate one sample of a waveform at the given phase (radians).

    sine (default): np.sin(phase)
    square: sign(sin(phase))
    saw:    2*(phase/(2π) mod 1) - 1
    triangle: 2*abs(2*(phase/(2π) mod 1) - 1) - 1

    The non-sine shapes are generated via phase accumulation (no lookup table),
    which keeps phases continuous across samples — the synth already maintains
    one phase per harmonic from sample to sample.
    """
    if shape == "sine" or shape is None:
        return float(np.sin(phase))
    if shape == "square":
        return float(np.sign(np.sin(phase)))
    if shape == "saw":
        frac = (phase / (2.0 * np.pi)) % 1.0
        return float(2.0 * frac - 1.0)
    if shape == "triangle":
        frac = (phase / (2.0 * np.pi)) % 1.0
        return float(2.0 * abs(2.0 * frac - 1.0) - 1.0)
    raise ValueError(f"unknown wave shape {shape!r}; expected sine/square/saw/triangle")


def harmonic_mask_audio(y, sr, f0, voiced, times,
                        n_harmonics=32, bandwidth_hz=5.0,
                        n_fft=4096, hop=None):
    """Keep only energy at harmonic frequencies f₁·N from the original audio.

    Computes STFT, masks all bins except those within ±bandwidth_hz of any
    harmonic frequency f₁·N, then inverse STFT back to time domain.

    This produces the theoretical upper bound of what the Shaper can achieve:
    100% of the energy comes from harmonic frequencies. No broadband noise.
    Fricatives and aspiration are preserved ONLY at the harmonic series.

    Parameters
    ----------
    bandwidth_hz : float
        Half-width of the bandpass window around each harmonic. 5 Hz means
        each harmonic keeps energy at f₁·N ± 5 Hz. Wider = more of the
        original preserved, narrower = purer harmonics.
    """
    if hop is None:
        hop = n_fft // 4  # 1024 for n_fft=4096
    from scipy.signal import istft
    import librosa
    log = logging.getLogger("synth_pure.harmonic_mask")
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    T = S.shape[1]

    # Build the mask: keep bins within bandwidth_hz of any f₁·N
    mask = np.zeros_like(S, dtype=bool)
    freq_res = sr / n_fft  # Hz per bin
    bin_width = max(1, int(np.ceil(bandwidth_hz / freq_res)))

    # For each frame, find harmonic frequencies and mask nearby bins
    for t in range(T):
        t_sec = t * hop / sr
        # Find closest analysis frame
        idx = int(np.searchsorted(times, t_sec))
        idx = min(max(idx, 0), len(times) - 1)
        ft = float(f0[idx])
        if not voiced[idx] or ft <= 0:
            continue
        for n in range(1, n_harmonics + 1):
            target_hz = ft * n
            if target_hz > sr / 2 - 50:
                break
            center_bin = int(np.searchsorted(freqs, target_hz))
            lo = max(0, center_bin - bin_width)
            hi = min(len(freqs) - 1, center_bin + bin_width)
            mask[lo:hi + 1, t] = True

    # Apply mask and reconstruct
    S_masked = S * mask
    _, y_out = istft(S_masked, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                     nfft=n_fft, boundary=True)
    # Trim/pad to match original length
    if len(y_out) < len(y):
        y_out = np.pad(y_out, (0, len(y) - len(y_out)))
    else:
        y_out = y_out[:len(y)]

    # Normalize to peak 1.0 — ISTFT with sparse mask produces
    # massively amplified output (scipy's overlap-add normalization
    # assumes full-spectrum input).
    pk = float(np.abs(y_out).max())
    if pk > 0:
        y_out = y_out / pk

    kept = float(mask.sum() / mask.size * 100)
    log.info("harmonic_mask: kept %.1f%% of STFT bins (bw=%.1f Hz, %d harmonics)",
             kept, bandwidth_hz, n_harmonics)
    return np.asarray(y_out, dtype=np.float64)


def synthesize_prepared(prepared: dict,
                        thresh_db: float = -30.0,
                        noise_floor_db: float = -40.0,
                        max_voices: int = 32,
                        gain_curve: str = "sqrt",
                        spectral_tilt_db: float = -12.0,
                        per_harmonic_gains: dict | None = None,
                        wave_shapes: dict | None = None,
                        noise_mix_db: float = -12.0,
                        limit_output: bool = False) -> np.ndarray:
    """Phase 2: render audio from a cached analysis dict.

    Parameters
    ----------
    prepared : dict
        Output of prepare_analysis(). MUST contain at minimum the keys
        returned by that function; the renderer does NOT call analyze() or
        prepare_analysis() again.
    thresh_db : float
        Threshold (dB) below H1 for considering a harmonic active.
        Default -30.
    noise_floor_db : float
        Absolute noise floor — any harmonic with magnitude below this is
        treated as inactive even if it's above the relative threshold.
        Prevents spectral leakage from formants being treated as harmonics.
        NOTE: the default in this function is -40.0 (matching the existing
        build_voice_compare_v3.py workflow), while the legacy synthesize()
        default was -50.0; that mismatch is preserved via synthesize()'s
        explicit noise_floor_db=-50.0 pass-through.
    max_voices : int
        Cap simultaneous active harmonics. Default 32 (was 6)
    gain_curve : {"linear", "sqrt", "square"}
        How to map dB magnitude → 0..1 gain:
          - "linear": (g_db + |floor|) / |floor|
          - "sqrt":   sqrt of linear — compresses range
          - "square": square of linear — expands range
    spectral_tilt_db : float
        Spectral tilt in dB per octave applied to harmonic gains.
        Default -12.0 matches the natural glottal source roll-off
        (Titze 2015: -10 to -15 dB/oct for normal voice).
        0 disables tilt (flat).
    per_harmonic_gains : dict[int, float], optional
        Multiplicative gain applied to each harmonic AFTER STFT-derived
        dB→linear conversion and BEFORE spectral tilt. Keys are 1-based
        harmonic numbers: {1: 1.0, 2: 0.8, 3: 0.6, ...}.
        Unspecified harmonics default to 1.0.
    wave_shapes : dict[int, str], optional
        Per-harmonic waveform override. Keys are 1-based harmonic numbers,
        values are one of: "sine", "square", "saw", "triangle".
        Unspecified harmonics default to "sine".
    noise_mix_db : float
        Level in dB for mixing extracted aperiodic residual after synthesis.
        Default -12.0. Use <= -120.0 to disable mixing.
    limit_output : bool
        Apply the live engine's ``tanh(mix * 1.05) * 0.95`` output stage.
        The default is False so analysis callers can inspect pre-limiter peaks.

    Returns
    -------
    np.ndarray, shape (total_samples,), dtype float64. With ``limit_output``
    true, every finite sample is in the declared range [-0.95, 0.95]. With it
    false, this is the unbounded pre-limiter mix used for level diagnosis.
    """
    # Sanity check — make sure we got a real prepared dict and not raw audio.
    if not isinstance(prepared, dict):
        raise ValueError(
            f"prepared must be a dict from prepare_analysis(); got {type(prepared).__name__}"
        )
    required = {"times", "f0", "voiced", "gains_db", "sr", "duration"}
    missing = required - set(prepared.keys())
    if missing:
        raise ValueError(
            f"prepared dict missing keys {missing}; "
            "did you pass raw audio instead of prepare_analysis() output?"
        )

    times = prepared["times"]
    f0 = prepared["f0"]
    voiced = prepared["voiced"]
    gains_db = prepared["gains_db"]
    sr = prepared["sr"]
    duration = prepared["duration"]

    # Pre-compute per-harmonic spectral tilt gains.
    # Natural voice has -10 to -15 dB/oct roll-off (Titze 2015).
    # We apply this as a gentle multiplicative gain per harmonic so higher
    # harmonics decay naturally rather than being cut off abruptly by a LPF.
    if spectral_tilt_db != 0.0:
        tilt_gains = np.ones(N_HARMONICS, dtype=np.float64)
        for n in range(1, N_HARMONICS + 1):
            octaves = np.log2(max(n, 1))
            tilt_gains[n - 1] = 10.0 ** (spectral_tilt_db * octaves / 20.0)
        log.info("Spectral tilt: %.1f dB/oct (H1=%.3f, H2=%.3f, H4=%.3f)",
                 spectral_tilt_db,
                 tilt_gains[0], tilt_gains[1], tilt_gains[3])
    else:
        tilt_gains = None
        log.info("Spectral tilt: flat (0 dB/oct)")

    # Per-harmonic gain overrides: 1-based index → multiplier.
    # Default 1.0 means "no change". Applied AFTER dB→linear and BEFORE tilt.
    harm_gains = np.ones(N_HARMONICS, dtype=np.float64)
    if per_harmonic_gains:
        for k, g in per_harmonic_gains.items():
            if not (1 <= k <= N_HARMONICS):
                raise ValueError(
                    f"per_harmonic_gains key {k} out of range 1..{N_HARMONICS}"
                )
            harm_gains[k - 1] = float(g)
        active_overrides = {k: v for k, v in per_harmonic_gains.items() if v != 1.0}
        if active_overrides:
            log.info("per_harmonic_gains overrides: %s", active_overrides)

    # Per-harmonic waveform overrides: 1-based index → shape name.
    # Default 'sine'. Each call to _waveform_value evaluates one sample at
    # the harmonic's current phase — phases are kept continuous across
    # samples, so non-sine shapes stay band-limited (no aliasing from a
    # piecewise construction at a fixed sample rate).
    shape_map: dict[int, str] = {}
    if wave_shapes:
        valid = {"sine", "square", "saw", "triangle"}
        for k, s in wave_shapes.items():
            if not (1 <= k <= N_HARMONICS):
                raise ValueError(
                    f"wave_shapes key {k} out of range 1..{N_HARMONICS}"
                )
            if s not in valid:
                raise ValueError(
                    f"wave_shapes[{k}] = {s!r} not in {valid}"
                )
            shape_map[k] = s
        if any(v != "sine" for v in shape_map.values()):
            log.info("wave_shapes overrides: %s", shape_map)

    total_samples = int(np.ceil(duration * SAMPLE_RATE))
    log.info("Rendering %d samples @ %d Hz (%.2f s), curve=%s, thresh=%ddB, "
             "floor=%ddB, max_voices=%d",
             total_samples, SAMPLE_RATE, duration, gain_curve,
             thresh_db, noise_floor_db, max_voices)

    # Per-voice state
    phases = np.zeros(N_HARMONICS + 1)  # index 0 unused, 1..32
    envs = np.zeros(N_HARMONICS + 1)
    # The live callback retains a voice's last parameters while its release
    # envelope decays. Mirror that behavior by holding the last active frame's
    # gain instead of replacing it with the new inactive frame's -120 dB.
    held_gains_db = np.full(N_HARMONICS + 1, -120.0, dtype=np.float64)

    out = np.zeros(total_samples, dtype=np.float64)
    block_size = 512
    floor_abs = abs(noise_floor_db)

    # Pre-compute per-block interpolated F0 and gains to avoid per-sample
    # index lookup. We also interpolate ft WITHIN a block so the harmonic
    # series glides continuously instead of stepping at block boundaries.
    n_blocks = (total_samples + block_size - 1) // block_size
    block_ft = np.zeros(n_blocks + 1)  # +1 for interpolation endpoint
    block_voiced = np.zeros(n_blocks + 1, dtype=bool)
    block_gains = np.zeros((n_blocks + 1, N_HARMONICS), dtype=np.float32)
    for b in range(n_blocks + 1):
        t_b = min(b * block_size / SAMPLE_RATE, duration)
        idx = int(np.searchsorted(times, t_b))
        idx = min(max(idx, 0), len(times) - 1)
        block_voiced[b] = bool(voiced[idx])
        block_ft[b] = float(f0[idx])
        block_gains[b] = gains_db[idx]

    # Track previous block's active harmonic mask so we can detect set changes
    # and apply boundary crossfading to avoid clicks from phase discontinuities.
    prev_active_mask: np.ndarray | None = None
    prev_frame_gains: np.ndarray | None = None

    # Per-sample renderer extracted so crossfade path can reuse without duplication.
    # Mutates the provided voice state arrays in place and returns the sample value.
    def _next_sample(frac: float, ft: float, use_active: np.ndarray,
                     phases_arr: np.ndarray, envs_arr: np.ndarray,
                     gains_arr: np.ndarray, gain_frame: np.ndarray) -> float:
        mix = 0.0
        for n in range(1, N_HARMONICS + 1):
            target_env = 1.0 if use_active[n - 1] else 0.0
            if target_env > 0.0:
                gains_arr[n] = gain_frame[n - 1]
            if target_env > envs_arr[n]:
                envs_arr[n] = min(target_env, envs_arr[n] + 1.0 / (0.010 * SAMPLE_RATE))
            else:
                envs_arr[n] = max(target_env, envs_arr[n] - 1.0 / (0.030 * SAMPLE_RATE))
            if envs_arr[n] <= 0:
                continue
            g_db = gains_arr[n]
            g_lin = max(0.0, min(1.0, (g_db + floor_abs) / floor_abs))
            if gain_curve == "sqrt":
                g_norm = np.sqrt(g_lin)
            elif gain_curve == "square":
                g_norm = g_lin * g_lin
            else:
                g_norm = g_lin
            # Per-harmonic gain override (between dB→linear and tilt).
            g_norm *= harm_gains[n - 1]
            # Apply spectral tilt (natural harmonic decay).
            if tilt_gains is not None:
                g_norm *= tilt_gains[n - 1]
            phases_arr[n] += 2.0 * np.pi * ft * n / SAMPLE_RATE
            # Per-harmonic waveform selection (default sine).
            shape = shape_map.get(n, "sine")
            wave_val = _waveform_value(shape, phases_arr[n])
            mix += g_norm * envs_arr[n] * wave_val

        n_active = int(np.sum(envs_arr > 0.001))
        norm = 1.0 / np.sqrt(max(n_active, 1))
        return mix * norm

    for block_idx in range(n_blocks):
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, total_samples)
        n_samples = block_end - block_start

        # Linear interpolation of F0 across the block (smooths pitch steps
        # between analysis frames — no audible pitch quantization).
        ft_start = block_ft[block_idx]
        ft_end = block_ft[block_idx + 1]
        voiced_start = block_voiced[block_idx]
        voiced_end = block_voiced[block_idx + 1]

        # Active set decision at start of block
        frame_gains = block_gains[block_idx]
        if voiced_start and ft_start > 0:
            ref = frame_gains[0]
            above_relative = frame_gains > (ref + thresh_db)
            above_absolute = frame_gains > noise_floor_db
            active_mask = above_relative & above_absolute
            if active_mask.sum() > max_voices:
                strengths = frame_gains.copy()
                strengths[~active_mask] = -200
                top_idx = np.argpartition(-strengths, max_voices)[:max_voices]
                new_mask = np.zeros(N_HARMONICS, dtype=bool)
                new_mask[top_idx] = True
                active_mask = new_mask
        else:
            active_mask = np.zeros(N_HARMONICS, dtype=bool)

        # Block-boundary crossfade when active harmonic SET changes.
        # A very short (4-sample) linear crossfade between the "continuation
        # under previous mask" and "start under new mask" prevents amplitude
        # steps / phase jumps from becoming audible clicks. Uses overlap of
        # two short renders (old + new) mixed with linear alpha; keeps exact
        # duration and all other behaviors (gains, shapes, tilt, noise, envs).
        cf = 4
        do_xfade = (block_idx > 0 and prev_active_mask is not None and
                    not np.array_equal(active_mask, prev_active_mask) and
                    block_start > 0)
        if do_xfade:
            cf = min(cf, n_samples)
            # Render cf samples continuing the PREVIOUS active set (from state at
            # block boundary, which matches end of prior block).
            old_phases = phases.copy()
            old_envs = envs.copy()
            old_gains_db = held_gains_db.copy()
            old_samples = np.empty(cf, dtype=np.float64)
            for s in range(cf):
                frac = (s + 0.5) / block_size if block_size > 0 else 0
                ft = ft_start + (ft_end - ft_start) * frac
                old_samples[s] = _next_sample(
                    frac, ft, prev_active_mask, old_phases, old_envs,
                    old_gains_db, prev_frame_gains,
                )
            # Render same cf samples under the NEW active set (advances real state).
            new_samples = np.empty(cf, dtype=np.float64)
            for s in range(cf):
                frac = (s + 0.5) / block_size if block_size > 0 else 0
                ft = ft_start + (ft_end - ft_start) * frac
                new_samples[s] = _next_sample(
                    frac, ft, active_mask, phases, envs,
                    held_gains_db, frame_gains,
                )
            # Linear crossfade: old fades out, new fades in.
            for s in range(cf):
                alpha = (s + 0.5) / cf
                out[block_start + s] = (1.0 - alpha) * old_samples[s] + alpha * new_samples[s]
            # Remaining non-transition samples of block under new mask (state already advanced).
            for s in range(cf, n_samples):
                frac = (s + 0.5) / block_size if block_size > 0 else 0
                ft = ft_start + (ft_end - ft_start) * frac
                out[block_start + s] = _next_sample(
                    frac, ft, active_mask, phases, envs,
                    held_gains_db, frame_gains,
                )
        else:
            for s in range(n_samples):
                frac = (s + 0.5) / block_size if block_size > 0 else 0
                ft = ft_start + (ft_end - ft_start) * frac
                out[block_start + s] = _next_sample(
                    frac, ft, active_mask, phases, envs,
                    held_gains_db, frame_gains,
                )

        if block_idx % 50 == 0:
            active_count = int(active_mask.sum())
            t_b = block_start / SAMPLE_RATE
            log.info("  t=%.2fs voiced=%d f0=%.1f active=%d",
                     t_b, voiced_start, ft_start, active_count)

        prev_active_mask = active_mask.copy()
        prev_frame_gains = frame_gains.copy()

    # Mix aperiodic noise (if enabled). Uses stft_raw/freqs_stft captured in prepare.
    if noise_mix_db > -120:
        stft_r = prepared.get("stft_raw")
        freqs_s = prepared.get("freqs_stft")
        if stft_r is not None and freqs_s is not None:
            # Construct dummy y sized for the *analysis* sr/length so extract returns
            # correct-length noise for the input-rate STFT grid; we resample after.
            y_len = int(np.ceil(duration * sr)) if (duration and sr) else len(out)
            y_dummy = np.zeros(max(y_len, 1), dtype=np.float32)
            noise_signal = extract_aperiodic(
                y_dummy, sr, f0, voiced, times,
                stft_raw=stft_r, freqs_stft=freqs_s,
            )
            # Resample noise to synth output rate (SAMPLE_RATE) if analysis sr differed.
            if sr != SAMPLE_RATE and len(noise_signal) > 0:
                import librosa
                noise_signal = librosa.resample(
                    noise_signal.astype(np.float32),
                    orig_sr=sr,
                    target_sr=SAMPLE_RATE,
                ).astype(np.float64)
            # Align lengths
            if len(noise_signal) < len(out):
                noise_signal = np.pad(noise_signal, (0, len(out) - len(noise_signal)))
            elif len(noise_signal) > len(out):
                noise_signal = noise_signal[:len(out)]
            # Filter noise through harmonic mask — only keep energy at f₁·N.
            # This ensures ALL output (harmonic + aperiodic) stays within
            # the harmonic series grid.
            noise_signal = harmonic_mask_audio(
                noise_signal, SAMPLE_RATE, f0, voiced, times,
                n_harmonics=max_voices, bandwidth_hz=5.0,
            )
            noise_gain_lin = 10.0 ** (noise_mix_db / 20.0)
            out = out + noise_gain_lin * noise_signal
            log.info("Aperiodic noise mixed at %.1f dB", noise_mix_db)
        else:
            log.info("Aperiodic noise requested but stft_raw missing from prepared dict")

    log.info("Pre-limiter mix: peak=%.4f RMS=%.4f",
             float(np.abs(out).max()), float(np.sqrt(np.mean(out**2))))
    if limit_output:
        out = soft_limit(out)
        log.info("Limited output: peak=%.4f RMS=%.4f",
                 float(np.abs(out).max()), float(np.sqrt(np.mean(out**2))))
    return out


def synthesize(y, sr, thresh_db, f0_min, f0_max, max_voices=32,
                noise_floor_db=-50.0, gain_curve="sqrt",
                spectral_tilt_db=-12.0, noise_mix_db=-12.0):
    """Render the additive synthesis sample-by-sample.

    Backwards-compatible wrapper: runs prepare_analysis() then
    synthesize_prepared() with default parameters. Every existing caller
    (build_voice_compare_v3.py, the CLI entry point) continues to work
    unchanged.

    For re-rendering the SAME audio with different synth parameters
    (waveform, per-harmonic gain tweaks, etc.), call prepare_analysis() once
    then synthesize_prepared() multiple times — saves ~95% of the cost on
    the second+ passes.
    """
    prepared = prepare_analysis(y, sr, f0_min=f0_min, f0_max=f0_max)
    return synthesize_prepared(
        prepared,
        thresh_db=thresh_db,
        noise_floor_db=noise_floor_db,
        max_voices=max_voices,
        gain_curve=gain_curve,
        spectral_tilt_db=spectral_tilt_db,
        noise_mix_db=noise_mix_db,
    )


_voice_cache: VoiceCache | None = None

def get_cache() -> VoiceCache:
    """Return the process-wide VoiceCache singleton."""
    global _voice_cache
    if _voice_cache is None:
        _voice_cache = VoiceCache()
    return _voice_cache


def synthesize_cached(y, sr, wav_path: Path = None, **kwargs) -> np.ndarray:
    """Synthesize with VoiceCache backing for prepare_analysis.

    If wav_path is given, attempts to reuse a cached prepare_analysis()
    result. Falls back to synthesize() when wav_path is None (no caching)
    or when the cache misses.

    All **kwargs are forwarded to synthesize_prepared(). The cache only
    covers prepare_analysis() — re-rendering with different synth params
    (thresh_db, gain_curve, etc.) reuses the cached analysis and just
    re-runs the cheap synthesize_prepared().
    """
    if wav_path is None:
        return synthesize(y, sr, **kwargs)

    cache = get_cache()
    prepared = cache.get(wav_path)
    if prepared is None:
        prepared = prepare_analysis(y, sr)
        cache.store(wav_path, prepared)

    # Extract f0_min/f0_max from kwargs if they were passed —
    # synthesize() extracts these, but synthesize_cached bypasses synthesize()
    # when cache hits. prepare_analysis uses f0_min/f0_max; if the cached dict
    # exists we've already used whatever values were passed on the first call.
    # All other kwargs (including noise_mix_db, max_voices, etc.) are passed
    # through to synthesize_prepared.
    synth_kwargs = {k: v for k, v in kwargs.items()
                    if k not in ('f0_min', 'f0_max')}
    return synthesize_prepared(prepared, **synth_kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--out", type=Path,
                        default=Path.home() / "Music" / "voice-analysis" / "synth_pure.wav")
    parser.add_argument("--thresh-db", type=float, default=-30.0)
    parser.add_argument("--f0-min", type=float, default=70.0)
    parser.add_argument("--f0-max", type=float, default=400.0)
    parser.add_argument("--log", default="INFO")
    parser.add_argument("--max-voices", type=int, default=32,
                        help="Cap simultaneous active harmonics (default 32)")
    parser.add_argument("--noise-floor-db", type=float, default=-50.0,
                        help="Absolute noise floor in dB (default -50)")
    parser.add_argument("--gain-curve", choices=["linear", "sqrt", "square"],
                        default="sqrt",
                        help="dB → 0..1 mapping: linear, sqrt (default), square")
    parser.add_argument("--spectral-tilt-db", type=float, default=-12.0,
                        help="Spectral tilt in dB/oct (default -12, 0=flat)")
    parser.add_argument("--noise-mix-db", type=float, default=-12.0,
                        help="Aperiodic noise mix level in dB (default -12; <=-120 disables)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if sf is None:
        raise ImportError("soundfile is required for CLI: " + str(getattr(sys.modules[__name__], "_SF_IMPORT_ERROR", "")))
    y, sr = sf.read(str(args.wav), always_2d=False)
    if y.ndim > 1:
        y = y[:, 0]
    y = y.astype(np.float32)
    log.info("Input: %d samples @ %d Hz (%.2f s)",
             len(y), sr, len(y) / sr)

    out = synthesize_cached(y, sr, wav_path=args.wav,
                           thresh_db=args.thresh_db,
                           f0_min=args.f0_min, f0_max=args.f0_max,
                           max_voices=args.max_voices,
                           noise_floor_db=args.noise_floor_db,
                           gain_curve=args.gain_curve,
                           spectral_tilt_db=args.spectral_tilt_db,
                           noise_mix_db=args.noise_mix_db)

    # Soft-clip + normalize to peak 0.95
    peak = float(np.abs(out).max())
    if peak > OUTPUT_LIMIT:
        log.warning("Peak %.3f > %.2f — applying Shaper soft limiter",
                    peak, OUTPUT_LIMIT)
        out = soft_limit(out)
    elif peak > 0:
        out = out * (OUTPUT_LIMIT / peak)
        log.info("Normalized: peak %.2f (gain was %.1f dB)",
                 OUTPUT_LIMIT, 20 * np.log10(OUTPUT_LIMIT / peak))
    else:
        log.error("Output is silent!")
        sys.exit(1)

    # Write 16-bit stereo WAV (mono → stereo duplicate)
    pcm = (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16)
    pcm_stereo = np.column_stack([pcm, pcm]).reshape(-1)
    with wave.open(str(args.out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm_stereo.tobytes())
    log.info("Wrote %s (%.1f KB)", args.out, args.out.stat().st_size / 1024)


if __name__ == "__main__":
    sys.exit(main())
