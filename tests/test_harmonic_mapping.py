"""Golden vectors and pure mapping tests for the native harmonic source."""

from __future__ import annotations

import pytest

from harmonic_shaper.harmonic_mapping import (
    DEFAULT_ANCHOR_MIDI,
    DEFAULT_F1_HZ,
    HARMONIC_MAP,
    beacon_frequency,
    get_harmonic_for_key,
    map_midi_note,
    playable_frequency,
    velocity_to_gain,
)


# Authority-derived golden vectors (f1=40.40, anchor=24) for the 12 classes
# at the anchor octave, computed from NaturalHarmony harmonics.py.
_CHROMATIC_AT_ANCHOR = [
    # (pc, midi_note, harmonic_n, playable_hz)
    (0, 24, 1, 40.40),
    (1, 25, 17, 42.925),
    (2, 26, 9, 45.45),
    (3, 27, 19, 47.975),
    (4, 28, 5, 50.50),
    (5, 29, 21, 53.025),
    (6, 30, 11, 55.55),
    (7, 31, 3, 60.60),
    (8, 32, 13, 65.65),
    (9, 33, 27, 68.175),
    (10, 34, 7, 70.70),
    (11, 35, 15, 75.75),
]

# Octave-aware C series (direct harmonic landings on powers of 2)
_C_OCTAVES = [
    # (midi_note, harmonic_n, playable_hz)
    (24, 1, 40.40),
    (36, 2, 80.80),
    (48, 4, 161.60),
    (60, 8, 323.20),
    (72, 16, 646.40),
    (84, 32, 1292.80),
]


class TestHarmonicMapTable:
    def test_twelve_chromatic_classes(self) -> None:
        assert set(HARMONIC_MAP.keys()) == set(range(12))

    def test_prototype_character(self) -> None:
        assert HARMONIC_MAP[0] == 1
        assert HARMONIC_MAP[4] == 5
        assert HARMONIC_MAP[7] == 3
        assert HARMONIC_MAP[10] == 7


class TestGoldenChromaticVectors:
    @pytest.mark.parametrize("pc,midi,n,freq", _CHROMATIC_AT_ANCHOR)
    def test_chromatic_class_at_anchor(
        self, pc: int, midi: int, n: int, freq: float
    ) -> None:
        m = map_midi_note(midi, f1=DEFAULT_F1_HZ, anchor_midi=DEFAULT_ANCHOR_MIDI)
        assert m.midi_note == midi
        assert m.harmonic_n == n
        assert m.harmonic_n == HARMONIC_MAP[pc]
        assert m.frequency_hz == pytest.approx(freq, rel=1e-9, abs=1e-9)
        assert m.beacon_freq_hz == pytest.approx(beacon_frequency(DEFAULT_F1_HZ, n))
        assert m.store_band(32) == n  # all prototypes ≤ 27

    @pytest.mark.parametrize("midi,n,freq", _C_OCTAVES)
    def test_octave_aware_c_series(self, midi: int, n: int, freq: float) -> None:
        m = map_midi_note(midi, f1=DEFAULT_F1_HZ, anchor_midi=DEFAULT_ANCHOR_MIDI)
        assert m.harmonic_n == n
        assert m.frequency_hz == pytest.approx(freq, rel=1e-9, abs=1e-9)
        assert m.source == "direct"
        assert get_harmonic_for_key(midi, DEFAULT_ANCHOR_MIDI) == n


class TestPlayableAndBeacon:
    def test_beacon_is_f1_times_n(self) -> None:
        assert beacon_frequency(40.40, 5) == pytest.approx(202.0)

    def test_playable_preserves_ratio_within_octave(self) -> None:
        f1 = 40.40
        fund = playable_frequency(f1, 1, 60)
        fifth = playable_frequency(f1, 3, 67)
        ratio = fifth / fund
        assert 1.4 <= ratio <= 1.6

    def test_map_matches_playable_helpers(self) -> None:
        for midi in (40, 55, 62, 71):
            m = map_midi_note(midi, f1=40.40, anchor_midi=24)
            n = get_harmonic_for_key(midi, 24)
            assert m.harmonic_n == n
            assert m.frequency_hz == pytest.approx(playable_frequency(40.40, n, midi))


class TestVelocityToGain:
    def test_bounds(self) -> None:
        assert velocity_to_gain(0) == 0.0
        assert velocity_to_gain(-1) == 0.0
        assert velocity_to_gain(127) == pytest.approx(1.0)
        assert velocity_to_gain(64) == pytest.approx(64 / 127.0)

    def test_custom_bounds(self) -> None:
        g = velocity_to_gain(127, min_gain=0.2, max_gain=0.8)
        assert g == pytest.approx(0.8)
        g0 = velocity_to_gain(0, min_gain=0.2, max_gain=0.8)
        assert g0 == 0.0
        mid = velocity_to_gain(64, min_gain=0.0, max_gain=0.5)
        assert mid == pytest.approx((64 / 127.0) * 0.5)
        assert 0.0 <= mid <= 0.5


class TestStoreBand:
    def test_in_range_uses_harmonic_n(self) -> None:
        m = map_midi_note(36, f1=40.40, anchor_midi=24)  # n=2
        assert m.store_band(32) == 2

    def test_out_of_range_falls_back_to_prototype(self) -> None:
        # Force a high direct harmonic if possible; C8-ish with low anchor
        m = map_midi_note(108, f1=40.40, anchor_midi=24)
        assert m.harmonic_n > 32
        assert m.store_band(32) == HARMONIC_MAP[108 % 12]
        assert 1 <= m.store_band(32) <= 32


class TestNoNaturalHarmonyImport:
    def test_module_has_no_runtime_nh_imports(self) -> None:
        import sys

        import harmonic_shaper.harmonic_mapping as hm

        # Runtime deps are stdlib only (math + dataclasses).
        assert hm.math.__name__ == "math"
        forbidden = ("NaturalHarmony", "harmonic_beacon")
        for value in vars(hm).values():
            module_name = getattr(value, "__module__", "") or ""
            assert not any(f in module_name for f in forbidden)
        assert not any(
            key.startswith("NaturalHarmony") or key.startswith("harmonic_beacon")
            for key in sys.modules
        )
