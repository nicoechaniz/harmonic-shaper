"""Native MIDI note-on/off lifecycle and headless VoiceParameterStore probe."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from harmonic_shaper import config
from harmonic_shaper.harmonic_mapping import map_midi_note, velocity_to_gain
from harmonic_shaper.main import build_parser
from harmonic_shaper.midi_control import NativeNoteHandler
from harmonic_shaper.state import VoiceParameterStore


def _active(store: VoiceParameterStore) -> dict[int, object]:
    return store.get_snapshot()


def _freqs(store: VoiceParameterStore) -> dict[int, float]:
    return {n: v.freq for n, v in store.get_snapshot().items()}


class TestCLINativeDefaults:
    def test_native_flags_default_on_path(self) -> None:
        args = build_parser().parse_args([])
        assert args.no_native_midi is False
        assert args.slave is False
        assert args.f1 == config.DEFAULT_F1
        assert args.anchor == config.DEFAULT_ANCHOR_MIDI

    def test_no_native_and_f1_anchor(self) -> None:
        args = build_parser().parse_args(
            ["--no-native-midi", "--f1", "55.0", "--anchor", "36"]
        )
        assert args.no_native_midi is True
        assert args.f1 == 55.0
        assert args.anchor == 36


class TestNativeNoteLifecycle:
    def test_note_on_activates_mapped_voice(self) -> None:
        store = VoiceParameterStore()
        store.update_f1(40.40)
        handler = NativeNoteHandler(store, anchor_midi=24, enabled=True)

        vid = handler.note_on(60, 100)  # C4 → n=8, playable 323.2
        assert vid is not None
        snap = _active(store)
        assert 8 in snap
        assert snap[8].active is True
        assert snap[8].voice_id == vid
        assert snap[8].freq == pytest.approx(323.2)
        assert snap[8].gain == pytest.approx(velocity_to_gain(100))

    def test_note_off_releases_originating_voice(self) -> None:
        store = VoiceParameterStore()
        store.update_f1(40.40)
        handler = NativeNoteHandler(store, anchor_midi=24)

        vid = handler.note_on(60, 80)
        assert _active(store)
        released = handler.note_off(60)
        assert released == vid
        assert _active(store) == {}
        assert handler.held_notes == {}

    def test_zero_velocity_is_note_off(self) -> None:
        store = VoiceParameterStore()
        store.update_f1(40.40)
        handler = NativeNoteHandler(store, anchor_midi=24)

        handler.note_on(64, 90)
        assert _active(store)
        handler.note_on(64, 0)  # zero-velocity note-on
        assert _active(store) == {}
        assert 64 not in handler.held_notes

    def test_velocity_bounds(self) -> None:
        store = VoiceParameterStore()
        store.update_f1(40.40)
        handler = NativeNoteHandler(
            store,
            anchor_midi=24,
            velocity_gain_min=0.1,
            velocity_gain_max=0.9,
        )
        handler.note_on(48, 127)
        g = store.get_snapshot()[4].gain  # C3 → n=4
        assert g == pytest.approx(0.9)
        handler.note_off(48)
        handler.note_on(48, 1)
        g_lo = store.get_snapshot()[4].gain
        assert 0.1 <= g_lo <= 0.9

    def test_overlapping_different_notes(self) -> None:
        store = VoiceParameterStore()
        store.update_f1(40.40)
        handler = NativeNoteHandler(store, anchor_midi=24)

        v1 = handler.note_on(60, 100)  # C4 → band 8
        v2 = handler.note_on(64, 100)  # E4 → band 5 (prototype)
        m60 = map_midi_note(60, 40.40, 24)
        m64 = map_midi_note(64, 40.40, 24)
        snap = _active(store)
        assert m60.store_band() in snap
        assert m64.store_band() in snap
        assert len(snap) == 2

        # Release C4 only — E4 must remain
        handler.note_off(60)
        snap = _active(store)
        assert m60.store_band() not in snap or (
            snap.get(m60.store_band()) is not None
            and snap[m60.store_band()].voice_id == v2
            if m60.store_band() == m64.store_band()
            else True
        )
        # When bands differ (expected for C4 vs E4), only E remains
        if m60.store_band() != m64.store_band():
            assert m64.store_band() in snap
            assert snap[m64.store_band()].voice_id == v2
            assert m60.store_band() not in snap

        handler.note_off(64)
        assert _active(store) == {}

    def test_overlapping_same_note_retrigger(self) -> None:
        store = VoiceParameterStore()
        store.update_f1(40.40)
        handler = NativeNoteHandler(store, anchor_midi=24)

        v1 = handler.note_on(60, 50)
        v2 = handler.note_on(60, 100)  # re-trigger
        assert v1 != v2
        snap = _active(store)
        assert len(snap) == 1
        assert list(snap.values())[0].voice_id == v2
        assert list(snap.values())[0].gain == pytest.approx(velocity_to_gain(100))

        handler.note_off(60)
        assert _active(store) == {}

    def test_concurrent_same_pitch_class_different_octaves(self) -> None:
        """C2 (n=2) and C3 (n=4) must not release each other."""
        store = VoiceParameterStore()
        store.update_f1(40.40)
        handler = NativeNoteHandler(store, anchor_midi=24)

        v_low = handler.note_on(36, 80)
        v_high = handler.note_on(48, 90)
        snap = _active(store)
        assert 2 in snap and 4 in snap
        assert snap[2].freq == pytest.approx(80.80)
        assert snap[4].freq == pytest.approx(161.60)

        handler.note_off(36)
        snap = _active(store)
        assert 2 not in snap
        assert 4 in snap
        assert snap[4].voice_id == v_high

        handler.note_off(48)
        assert _active(store) == {}
        assert v_low != v_high

    def test_co_band_collision_superseded_note_off_keeps_current_owner(self) -> None:
        """Same-band collision: note-off for a superseded note must not silence the owner.

        With f1=40.4 and anchor=24, MIDI notes 0 and 12 both map to store band 1.
        The store retains only the most recent active voice per band; note-off must
        target the originating voice_id only so the current band owner stays live.
        """
        f1, anchor = 40.4, 24
        m0 = map_midi_note(0, f1=f1, anchor_midi=anchor)
        m12 = map_midi_note(12, f1=f1, anchor_midi=anchor)
        assert m0.store_band() == 1
        assert m12.store_band() == 1
        assert m0.store_band() == m12.store_band()

        store = VoiceParameterStore()
        store.update_f1(f1)
        handler = NativeNoteHandler(store, anchor_midi=anchor, enabled=True)

        v0 = handler.note_on(0, 100)
        assert v0 == 1_000_000
        snap = _active(store)
        assert 1 in snap
        assert snap[1].active is True
        assert snap[1].voice_id == v0

        v12 = handler.note_on(12, 100)
        assert v12 == 1_000_001
        assert v12 != v0
        snap = _active(store)
        assert len(snap) == 1
        assert snap[1].active is True
        assert snap[1].voice_id == v12  # replaces active store-band owner

        # Superseded same-band note-off must NOT deactivate the current owner
        released = handler.note_off(0)
        assert released == v0
        snap = _active(store)
        assert 1 in snap
        assert snap[1].active is True
        assert snap[1].voice_id == v12
        assert handler.held_notes == {12: v12}

        # Final active voice releases only on its own note-off
        released_final = handler.note_off(12)
        assert released_final == v12
        assert _active(store) == {}
        assert handler.held_notes == {}

    def test_disabled_handler_ignores_notes(self) -> None:
        store = VoiceParameterStore()
        handler = NativeNoteHandler(store, enabled=False)
        assert handler.note_on(60, 100) is None
        assert _active(store) == {}

    def test_handle_message_note_on_off(self) -> None:
        store = VoiceParameterStore()
        store.update_f1(40.40)
        handler = NativeNoteHandler(store, anchor_midi=24)
        handler.handle_message(SimpleNamespace(type="note_on", note=36, velocity=100))
        assert 2 in _active(store)
        handler.handle_message(SimpleNamespace(type="note_off", note=36, velocity=0))
        assert _active(store) == {}

    def test_panic_clears_held(self) -> None:
        store = VoiceParameterStore()
        store.update_f1(40.40)
        handler = NativeNoteHandler(store, anchor_midi=24)
        handler.note_on(36, 100)
        handler.note_on(48, 100)
        handler.panic()
        assert handler.held_notes == {}
        assert _active(store) == {}

    def test_f1_from_store_live(self) -> None:
        store = VoiceParameterStore()
        store.update_f1(40.40)
        handler = NativeNoteHandler(store, anchor_midi=24)
        expected_a = map_midi_note(24, f1=40.40, anchor_midi=24)
        handler.note_on(24, 100)
        assert store.get_snapshot()[expected_a.store_band()].freq == pytest.approx(
            expected_a.frequency_hz
        )
        handler.note_off(24)
        store.update_f1(55.0)
        expected_b = map_midi_note(24, f1=store.f1, anchor_midi=24)
        handler.note_on(24, 100)
        assert store.get_snapshot()[expected_b.store_band()].freq == pytest.approx(
            expected_b.frequency_hz
        )
        assert expected_b.frequency_hz != expected_a.frequency_hz


class TestHeadlessProbe:
    """Feed fake MIDI through the native path; assert store snapshots.

    No audio hardware, no MIDI ports — pure in-process probe.
    """

    def test_headless_sequence(self) -> None:
        store = VoiceParameterStore()
        store.update_f1(config.DEFAULT_F1)
        handler = NativeNoteHandler(
            store,
            anchor_midi=config.DEFAULT_ANCHOR_MIDI,
            enabled=True,
        )

        report: list[str] = []

        def snap_line(label: str) -> None:
            s = store.get_snapshot()
            parts = [
                f"n={n} freq={v.freq:.3f} gain={v.gain:.3f} vid={v.voice_id}"
                for n, v in sorted(s.items())
            ]
            line = f"{label}: active={len(s)} [{'; '.join(parts)}]"
            report.append(line)

        # Chromatic walk at anchor octave
        notes = list(range(24, 36))
        for note in notes:
            handler.note_on(note, 100)
        snap_line("after_12_note_on")
        assert len(_active(store)) == 12

        freqs = _freqs(store)
        for pc, midi, n, freq in [
            (0, 24, 1, 40.40),
            (4, 28, 5, 50.50),
            (7, 31, 3, 60.60),
        ]:
            assert n in freqs
            assert freqs[n] == pytest.approx(freq)

        # Release half
        for note in notes[:6]:
            handler.note_off(note)
        snap_line("after_6_note_off")
        assert len(_active(store)) == 6

        # Zero-velocity release of one remaining
        still = notes[6]
        handler.note_on(still, 0)
        snap_line("after_zero_vel")
        assert still not in handler.held_notes

        # Overlap polyphony: hold two C octaves
        handler.panic()
        store.panic()
        handler.note_on(36, 80)
        handler.note_on(48, 90)
        snap_line("poly_c2_c3")
        assert _freqs(store)[2] == pytest.approx(80.80)
        assert _freqs(store)[4] == pytest.approx(161.60)

        handler.note_off(36)
        snap_line("release_c2_keep_c3")
        assert 2 not in _active(store)
        assert 4 in _active(store)

        handler.note_off(48)
        snap_line("all_released")
        assert _active(store) == {}

        # Print probe report for the build task report (captured by pytest -s)
        print("\n--- headless native MIDI probe ---")
        for line in report:
            print(line)
        print("--- end probe ---\n")
