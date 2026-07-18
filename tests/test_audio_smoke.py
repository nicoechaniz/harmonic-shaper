"""Deterministic, hardware-free Shaper level and transition evidence.

Run ``pytest tests/test_audio_smoke.py -q -s`` to print the characterized
peak/RMS/full-scale measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import harmonic_shaper.audio_engine as audio_engine_module
from harmonic_shaper.audio_engine import AudioEngine
from harmonic_shaper.audio_levels import OUTPUT_LIMIT, soft_limit
from harmonic_shaper.state import VoiceParameterStore
from harmonic_shaper.synth_pure import (
    N_HARMONICS,
    SAMPLE_RATE,
    synthesize_prepared,
)


BLOCK_SIZE = 512
FUNDAMENTAL_HZ = 110.0
TRANSITION_HISTORY = [32] * 12 + [1] * 4 + [32] * 12 + [1] * 20


@dataclass(frozen=True)
class AudioMetrics:
    peak: float
    rms: float
    at_or_over_full_scale: int


def _metrics(samples: np.ndarray) -> AudioMetrics:
    values = np.asarray(samples, dtype=np.float64)
    return AudioMetrics(
        peak=float(np.max(np.abs(values))),
        rms=float(np.sqrt(np.mean(np.square(values)))),
        at_or_over_full_scale=int(np.count_nonzero(np.abs(values) >= 1.0)),
    )


def _emit_metrics(scenario: str, stage: str, samples: np.ndarray) -> AudioMetrics:
    result = _metrics(samples)
    print(
        f"{scenario:18s} stage={stage:11s} samples={samples.shape[0]:5d} "
        f"peak={result.peak:.9f} RMS={result.rms:.9f} "
        f"at/over-full-scale={result.at_or_over_full_scale}"
    )
    return result


def _prepared(active_voice_counts: list[int]) -> dict:
    """Build one analysis row per renderer block with deterministic gains."""

    block_count = len(active_voice_counts)
    gains_db = np.full(
        (block_count + 1, N_HARMONICS), -120.0, dtype=np.float32
    )
    for block_index, active_count in enumerate(active_voice_counts):
        gains_db[block_index, :active_count] = 0.0
    gains_db[-1, :active_voice_counts[-1]] = 0.0

    return {
        "times": np.arange(block_count + 1, dtype=np.float64)
        * BLOCK_SIZE
        / SAMPLE_RATE,
        "f0": np.full(block_count + 1, FUNDAMENTAL_HZ, dtype=np.float64),
        "voiced": np.ones(block_count + 1, dtype=bool),
        "gains_db": gains_db,
        "sr": SAMPLE_RATE,
        "duration": block_count * BLOCK_SIZE / SAMPLE_RATE,
    }


def _render_reference(active_voice_counts: list[int], *, limited: bool) -> np.ndarray:
    return synthesize_prepared(
        _prepared(active_voice_counts),
        noise_floor_db=-50.0,
        spectral_tilt_db=0.0,
        noise_mix_db=-120.0,
        limit_output=limited,
    )


@pytest.mark.parametrize(
    ("scenario", "active_voice_counts", "expect_pre_limiter_overload"),
    [
        ("pure-one-voice", [1] * 16, False),
        ("pure-dense-32", [32] * 16, True),
    ],
)
def test_pure_prepared_smoke_levels(
    scenario: str,
    active_voice_counts: list[int],
    expect_pre_limiter_overload: bool,
) -> None:
    requested_samples = len(active_voice_counts) * BLOCK_SIZE
    raw = _render_reference(active_voice_counts, limited=False)
    limited = _render_reference(active_voice_counts, limited=True)

    assert raw.shape == (requested_samples,)
    assert limited.shape == (requested_samples,)
    assert np.isfinite(raw).all()
    assert np.isfinite(limited).all()
    assert float(np.max(np.abs(limited))) <= OUTPUT_LIMIT
    np.testing.assert_array_equal(limited, soft_limit(raw))

    raw_metrics = _emit_metrics(scenario, "pre-limiter", raw)
    limited_metrics = _emit_metrics(scenario, "output", limited)
    assert limited_metrics.at_or_over_full_scale == 0
    if expect_pre_limiter_overload:
        assert raw_metrics.peak > 4.0
        assert raw_metrics.at_or_over_full_scale > 0
    else:
        assert 0.99 < raw_metrics.peak < 1.0
        assert raw_metrics.at_or_over_full_scale == 0


@pytest.fixture(scope="module")
def transition_reference() -> tuple[np.ndarray, np.ndarray]:
    raw = _render_reference(TRANSITION_HISTORY, limited=False)
    limited = _render_reference(TRANSITION_HISTORY, limited=True)
    return raw, limited


def test_pure_rapid_active_count_transition_keeps_release_gains(
    transition_reference: tuple[np.ndarray, np.ndarray],
) -> None:
    raw, limited = transition_reference
    requested_samples = len(TRANSITION_HISTORY) * BLOCK_SIZE

    assert raw.shape == (requested_samples,)
    assert limited.shape == (requested_samples,)
    assert np.isfinite(raw).all()
    assert np.isfinite(limited).all()
    assert float(np.max(np.abs(limited))) <= OUTPUT_LIMIT

    raw_metrics = _emit_metrics("pure-transition", "pre-limiter", raw)
    limited_metrics = _emit_metrics("pure-transition", "output", limited)
    assert 4.1 < raw_metrics.peak < 4.5
    assert raw_metrics.at_or_over_full_scale > 0
    assert limited_metrics.at_or_over_full_scale == 0

    # Blocks 12..15 drop from 32 voices to one. The other 31 voices must retain
    # their last gain while their 30 ms release envelopes decay. Before the
    # held-gain correction this window peaked at one and contained no tail sum.
    release_window = raw[12 * BLOCK_SIZE : 16 * BLOCK_SIZE]
    release_metrics = _emit_metrics(
        "pure-release-tail", "pre-limiter", release_window
    )
    assert release_metrics.peak > 3.0
    assert release_metrics.at_or_over_full_scale > 0


def _activate_voices(store: VoiceParameterStore, harmonic_ns: range) -> None:
    for harmonic_n in harmonic_ns:
        store.set_attack(harmonic_n, 0.010)
        store.set_release(harmonic_n, 0.030)
        store.set_pan(harmonic_n, 0.0)
        store.set_phase(harmonic_n, 0.0)
        store.set_shape(harmonic_n, 0.0)
        store.voice_on(
            harmonic_n,
            voice_id=harmonic_n,
            freq=FUNDAMENTAL_HZ * harmonic_n,
            gain=1.0,
        )


def _deactivate_voices(store: VoiceParameterStore, harmonic_ns: range) -> None:
    for harmonic_n in harmonic_ns:
        store.voice_off(voice_id=harmonic_n)


def _callback_blocks(engine: AudioEngine, block_count: int) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for _ in range(block_count):
        outdata = np.empty((BLOCK_SIZE, 2), dtype=np.float32)
        engine._audio_callback(outdata, BLOCK_SIZE, time_info=None, status=None)
        blocks.append(outdata.copy())
    return np.concatenate(blocks)


def test_live_callback_transition_matches_reference_without_hardware(
    transition_reference: tuple[np.ndarray, np.ndarray],
) -> None:
    reference_raw, _ = transition_reference
    store = VoiceParameterStore()
    store.set_master_gain(1.0)
    store.set_sidechain_amount(0.0)
    store.set_lfo_amount(0.0)
    engine = AudioEngine(store, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    pre_limiter_blocks: list[np.ndarray] = []
    engine.attach_recorder(pre_limiter_blocks)

    _activate_voices(store, range(1, N_HARMONICS + 1))
    output_sections = [_callback_blocks(engine, 12)]
    _deactivate_voices(store, range(2, N_HARMONICS + 1))
    output_sections.append(_callback_blocks(engine, 4))
    _activate_voices(store, range(2, N_HARMONICS + 1))
    output_sections.append(_callback_blocks(engine, 12))
    _deactivate_voices(store, range(2, N_HARMONICS + 1))
    output_sections.append(_callback_blocks(engine, 20))

    output = np.concatenate(output_sections)
    pre_limiter = np.concatenate(pre_limiter_blocks)
    requested_samples = len(TRANSITION_HISTORY) * BLOCK_SIZE
    assert output.shape == (requested_samples, 2)
    assert pre_limiter.shape == (requested_samples, 2)
    assert np.isfinite(output).all()
    assert np.isfinite(pre_limiter).all()
    assert float(np.max(np.abs(output))) <= OUTPUT_LIMIT
    assert set(engine._voice_state) == {1}

    raw_metrics = _emit_metrics("live-transition", "pre-limiter", pre_limiter)
    output_metrics = _emit_metrics("live-transition", "output", output)
    assert raw_metrics.at_or_over_full_scale > 0
    assert output_metrics.at_or_over_full_scale == 0

    # Both live voices are centered, so equal-power panning makes each channel
    # the mono reference times 1/sqrt(2). Callback envelopes are block-rate,
    # while the reference envelopes are sample-rate; their aggregate transition
    # metrics should nevertheless remain close.
    pan_scale = 1.0 / np.sqrt(2.0)
    reference_metrics = _metrics(reference_raw * pan_scale)
    assert raw_metrics.peak == pytest.approx(reference_metrics.peak, rel=0.05)
    assert raw_metrics.rms == pytest.approx(reference_metrics.rms, rel=0.05)

    # Away from an envelope edge, the renderers are sample-identical after
    # accounting for their one-sample oscillator convention and float32 output.
    start = 2 * BLOCK_SIZE
    stop = 12 * BLOCK_SIZE
    expected_left = reference_raw[start : stop - 1] * pan_scale
    actual_left = pre_limiter[start + 1 : stop, 0]
    np.testing.assert_allclose(actual_left, expected_left, rtol=0.0, atol=1e-6)


def test_audio_engine_only_requires_sounddevice_when_starting(monkeypatch) -> None:
    monkeypatch.setattr(audio_engine_module, "HAS_SOUNDDEVICE", False)
    monkeypatch.setattr(
        audio_engine_module, "SOUNDDEVICE_IMPORT_ERROR", ImportError("test sentinel")
    )
    engine = AudioEngine(VoiceParameterStore())

    with pytest.raises(ImportError, match="test sentinel"):
        engine.start()
