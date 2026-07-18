"""Headless NumPy render regression for the extracted reference synth."""

from __future__ import annotations

import numpy as np

from harmonic_shaper.synth_pure import N_HARMONICS, SAMPLE_RATE, synthesize_prepared


def test_single_voice_render_is_finite_and_bounded() -> None:
    sample_count = 2_048
    duration = sample_count / SAMPLE_RATE
    gains_db = np.full((2, N_HARMONICS), -120.0, dtype=np.float32)
    gains_db[:, 0] = 0.0
    prepared = {
        "times": np.array([0.0, duration], dtype=np.float64),
        "f0": np.array([220.0, 220.0], dtype=np.float64),
        "voiced": np.array([True, True]),
        "gains_db": gains_db,
        "sr": SAMPLE_RATE,
        "duration": duration,
    }

    rendered = synthesize_prepared(
        prepared,
        noise_floor_db=-50.0,
        spectral_tilt_db=0.0,
        noise_mix_db=-120.0,
    )

    assert rendered.shape == (sample_count,)
    assert np.isfinite(rendered).all()
    assert float(np.max(np.abs(rendered))) <= 1.0
    assert float(np.max(np.abs(rendered))) > 0.0
