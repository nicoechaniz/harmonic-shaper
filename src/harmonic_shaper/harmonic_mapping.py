"""Native MIDI-note → natural-harmonic mapping for the Harmonic Shaper.

Dependency-free extract of the portable musical core from NaturalHarmony:

- ``harmonic_beacon/harmonics.py`` — ``HARMONIC_MAP``, ``get_harmonic_for_key``,
  ``beacon_frequency``, ``playable_frequency``
- ``harmonic_beacon/key_mapper.py`` — chromatic prototype character +
  octave-aware playable pitch (without NH config/UI coupling)

This module never imports NaturalHarmony.  The stable public entry point is
:func:`map_midi_note`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ── Chromatic natural-harmonic prototypes (12 pitch classes) ─────────────────
# MIDI key offset 0–11 relative to C → harmonic number n.
# Preserves NaturalHarmony's 12-key musical character.
HARMONIC_MAP: dict[int, int] = {
    0: 1,    # C  → Fundamental (1/1)
    1: 17,   # C# → Minor Second (17/16)
    2: 9,    # D  → Major Second (9/8)
    3: 19,   # Eb → Harmonic minor 3rd (19/16)
    4: 5,    # E  → Major Third (5/4)
    5: 21,   # F  → Narrow Fourth (21/16)
    6: 11,   # F# → Mystic Tritone (11/8)
    7: 3,    # G  → Perfect Fifth (3/2)
    8: 13,   # Ab → Harmonic minor 6th (13/8)
    9: 27,   # A  → Major Sixth (27/16)
    10: 7,   # Bb → Harmonic Seventh (7/4)
    11: 15,  # B  → Major Seventh (15/8)
}

INTERVAL_NAMES: dict[int, str] = {
    0: "Fundamental",
    1: "Minor Second",
    2: "Major Second",
    3: "Harmonic m3",
    4: "Major Third",
    5: "Narrow Fourth",
    6: "Mystic Tritone",
    7: "Perfect Fifth",
    8: "Harmonic m6",
    9: "Major Sixth",
    10: "Harmonic Seventh",
    11: "Major Seventh",
}

# Defaults aligned with Shaper / NaturalHarmony practice
DEFAULT_F1_HZ = 40.40
DEFAULT_ANCHOR_MIDI = 24  # C1 — MIDI note representing f₁
DEFAULT_CENTS_THRESHOLD = 25.0

MIDI_A4 = 69
FREQ_A4 = 440.0

# Velocity → gain bounds for native note source
VELOCITY_GAIN_MIN = 0.0
VELOCITY_GAIN_MAX = 1.0


@dataclass(frozen=True)
class NoteMapping:
    """Result of mapping one MIDI note to a Shaper harmonic voice.

    Attributes:
        midi_note: Pressed MIDI note (0–127).
        harmonic_n: Harmonic index n ≥ 1 from the hybrid key mapper.
        frequency_hz: Octave-aware playable frequency (Hz).
        beacon_freq_hz: Raw beacon frequency f₁ × n (Hz), before octave adapt.
        source: ``\"direct\"`` if the key lands near a pure harmonic, else
            ``\"interval\"`` (12-key table fallback).
        anchor_midi: Anchor MIDI note used for this mapping.
        f1: Fundamental frequency used for this mapping.
        cents_error: Signed cents deviation from the nearest pure harmonic
            (meaningful for diagnostics; zero-ish for direct hits).
    """

    midi_note: int
    harmonic_n: int
    frequency_hz: float
    beacon_freq_hz: float
    source: str
    anchor_midi: int
    f1: float
    cents_error: float = 0.0

    def store_band(self, max_bands: int = 32) -> int:
        """Band index for :class:`~harmonic_shaper.state.VoiceParameterStore`.

        Uses ``harmonic_n`` when it falls inside the lattice ``1..max_bands``.
        Out-of-range harmonics fall back to the chromatic prototype so control
        surfaces and polyphony bookkeeping stay within the lattice.
        """
        if 1 <= self.harmonic_n <= max_bands:
            return self.harmonic_n
        return HARMONIC_MAP[self.midi_note % 12]


def get_standard_frequency(midi_note: float) -> float:
    """Equal-tempered frequency for a (possibly fractional) MIDI note, A4=440 Hz."""
    return FREQ_A4 * (2.0 ** ((float(midi_note) - MIDI_A4) / 12.0))


def get_harmonic_for_key(
    midi_note: int,
    anchor_note: int = DEFAULT_ANCHOR_MIDI,
    cents_threshold: float = DEFAULT_CENTS_THRESHOLD,
) -> int:
    """Hybrid 88-key harmonic index (direct landing, else 12-key table).

    1. If the key is within ``cents_threshold`` of a pure harmonic of the
       anchor, use that harmonic (left-aligned floor of ``2^(semitones/12)``).
    2. Otherwise fall back to :data:`HARMONIC_MAP` for the pitch class so each
       chromatic position keeps its natural-harmonic character.
    """
    semitones = int(midi_note) - int(anchor_note)
    n_exact = 2 ** (semitones / 12)
    n_nearest = max(1, int(n_exact))

    if n_nearest > 0:
        perfect_semitones = 12 * math.log2(n_nearest)
        cents_error = abs(semitones - perfect_semitones) * 100
    else:
        cents_error = float("inf")

    if cents_error <= cents_threshold:
        return n_nearest
    return HARMONIC_MAP[int(midi_note) % 12]


def get_harmonic_info(
    midi_note: int,
    anchor_note: int = DEFAULT_ANCHOR_MIDI,
    cents_threshold: float = DEFAULT_CENTS_THRESHOLD,
) -> dict:
    """Detailed harmonic diagnostics for a key (mirrors NH authority shape)."""
    semitones = int(midi_note) - int(anchor_note)
    n_exact = 2 ** (semitones / 12)
    n_nearest = max(1, int(n_exact))

    if n_nearest > 0:
        perfect_semitones = 12 * math.log2(n_nearest)
        cents_error = (semitones - perfect_semitones) * 100
    else:
        cents_error = 0.0

    is_direct = abs(cents_error) <= cents_threshold
    if is_direct:
        n_used = n_nearest
        source = "direct"
    else:
        n_used = HARMONIC_MAP[int(midi_note) % 12]
        source = "interval"

    return {
        "midi_note": int(midi_note),
        "harmonic": n_used,
        "n_exact": n_exact,
        "n_nearest": n_nearest,
        "cents_error": cents_error,
        "semitones_from_anchor": semitones,
        "source": source,
    }


def beacon_frequency(f1: float, n: int) -> float:
    """Raw harmonic (beacon) frequency: f₁ × n."""
    return float(f1) * int(n)


def playable_frequency(f1: float, n: int, target_note: int) -> float:
    """Octave-aware playable frequency for a pressed key.

    Starts from the raw beacon voice ``f1 * n``, then shifts by whole octaves
    so the result sits near the 12-TET pitch expected for ``target_note``.
    """
    raw_freq = beacon_frequency(f1, n)
    target_freq = get_standard_frequency(target_note)
    if raw_freq <= 0 or target_freq <= 0:
        return 0.0
    ratio = target_freq / raw_freq
    octave_shift = round(math.log2(ratio))
    return raw_freq * (2.0 ** octave_shift)


def velocity_to_gain(
    velocity: int,
    *,
    min_gain: float = VELOCITY_GAIN_MIN,
    max_gain: float = VELOCITY_GAIN_MAX,
) -> float:
    """Map MIDI velocity to bounded gain in ``[min_gain, max_gain]``.

    Velocity ≤ 0 yields 0.0 (callers treat this as note-off).  Values are
    clamped to MIDI 0–127 before scaling, then to ``[0, 1]``.
    """
    if velocity <= 0:
        return 0.0
    linear = max(0, min(127, int(velocity))) / 127.0
    lo = max(0.0, float(min_gain))
    hi = min(1.0, float(max_gain))
    if hi < lo:
        lo, hi = hi, lo
    return max(0.0, min(1.0, lo + linear * (hi - lo)))


def map_midi_note(
    midi_note: int,
    f1: float = DEFAULT_F1_HZ,
    anchor_midi: int = DEFAULT_ANCHOR_MIDI,
    *,
    cents_threshold: float = DEFAULT_CENTS_THRESHOLD,
) -> NoteMapping:
    """Stable mapping API: MIDI note + f₁ / anchor → harmonic voice params.

    Parameters
    ----------
    midi_note:
        MIDI note number (0–127).
    f1:
        Fundamental frequency in Hz (Shaper lattice root).
    anchor_midi:
        MIDI note that represents f₁ (default C1 = 24).
    cents_threshold:
        Direct-harmonic landing tolerance in cents.

    Returns
    -------
    NoteMapping
        Frozen mapping with ``harmonic_n``, octave-aware ``frequency_hz``,
        and source metadata.
    """
    note = max(0, min(127, int(midi_note)))
    info = get_harmonic_info(
        note, anchor_note=int(anchor_midi), cents_threshold=cents_threshold
    )
    n = int(info["harmonic"])
    f1_v = float(f1)
    beacon = beacon_frequency(f1_v, n)
    playable = playable_frequency(f1_v, n, note)
    return NoteMapping(
        midi_note=note,
        harmonic_n=n,
        frequency_hz=playable,
        beacon_freq_hz=beacon,
        source=str(info["source"]),
        anchor_midi=int(anchor_midi),
        f1=f1_v,
        cents_error=float(info["cents_error"]),
    )
