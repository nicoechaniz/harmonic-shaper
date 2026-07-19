# Native MIDI Harmonic Banks

## Purpose

The default native keyboard mode is a direct playable projection of the active
harmonic series. It does not derive pitch from 12-tone equal temperament and
does not octave-shift partials to resemble a piano.

For an active fundamental `f1`, adjacent keys select adjacent integer partials:

```text
first key -> n=1 -> f1 * 1
next key  -> n=2 -> f1 * 2
...
key 32    -> n=32 -> f1 * 32
```

The MIDI note number selects the partial index; velocity controls gain only.
Whenever the runtime updates `f1`, the next key press uses the new active
series.

## Banks

The default configuration defines two non-overlapping 32-key banks:

| Bank | MIDI notes | Behavior |
|---|---:|---|
| Momentary | `24..55` | A press starts the partial and a release stops it. |
| Toggle | `72..103` | A press starts and sustains the partial; releasing the key does nothing; pressing that same key again stops it. |

The defaults were calibrated from a generic three-octave keyboard using its
lowest and highest transpose positions. They are MIDI-number configuration,
not a device-name dependency.

Notes outside configured banks are ignored deliberately. This prevents a
controller in an intermediate transpose position from accidentally falling
back to a tempered or pitch-class mapping.

## Configuration

Defaults live in `harmonic_shaper.config`:

```text
NATIVE_MIDI_MAPPING_MODE = "sequential_banks"
NATIVE_MIDI_MOMENTARY_START = 24
NATIVE_MIDI_TOGGLE_START = 72
NATIVE_MIDI_BANK_SIZE = 32
```

A different controller can override its bank starts at launch:

```bash
python -m harmonic_shaper \
  --native-midi-momentary-start 36 \
  --native-midi-toggle-start 84
```

`--native-midi-mode legacy_hybrid` explicitly enables the previous
NaturalHarmony-derived mapping for compatibility. It is not the default
playable-series behavior.

## Safety and lifecycle

- Toggle-bank note-off messages are ignored by design; otherwise ordinary key
  release would cancel sustain immediately.
- Momentary and toggle voices retain separate ownership maps, so a note-off
  cannot silence another key's current owner.
- `panic` releases both banks.
- Overlapping configured banks are rejected at startup.

## Verification

The behavior is covered by `tests/test_native_midi.py`:

- direct `f1*n` mapping with no 12-TET adaptation;
- adjacent momentary keys selecting adjacent partials;
- toggle sustain, second-press release, and ignored toggle note-off;
- ignored notes outside banks;
- panic and configuration-overlap safety.
