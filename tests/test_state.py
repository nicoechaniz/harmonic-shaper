"""VoiceParameterStore behavior and concurrency smoke tests."""

from __future__ import annotations

import threading

import numpy as np

from harmonic_shaper import config
from harmonic_shaper.audio_engine import AudioEngine
from harmonic_shaper.osc_receiver import ShaperOSCReceiver
from harmonic_shaper.state import VoiceParameterStore


def test_store_clamps_contract_parameters() -> None:
    store = VoiceParameterStore()
    store.set_gain(1, 2.0)
    store.set_pan(1, -3.0)
    store.set_phase(1, 450.0)
    store.set_master_gain(-1.0)

    voice = store.get_all_snapshot()[1]
    assert voice.gain == 1.0
    assert voice.pan == -1.0
    assert round(voice.phase, 6) == round(3.141592653589793 / 2.0, 6)
    assert store.get_master_gain() == 0.0


def test_harmonic_envelope_owns_lifecycle_and_tracks_active_series() -> None:
    store = VoiceParameterStore()
    store.update_f1(50.0)

    store.set_harmonic_envelope(3, 0.6)
    voice = store.get_snapshot()[3]
    assert voice.active is True
    assert voice.gain == 0.6
    assert voice.freq == 150.0

    store.set_vsrate(1.5)
    assert store.get_snapshot()[3].freq == 225.0
    store.update_f1(40.0)
    assert store.get_snapshot()[3].freq == 180.0

    store.set_harmonic_envelope(3, 0.0)
    assert 3 not in store.get_snapshot()


def test_harmonic_envelope_release_preserves_different_voice_owner() -> None:
    store = VoiceParameterStore()
    store.set_harmonic_envelope(4, 0.5)
    store.voice_on(4, voice_id=42, freq=161.6, gain=0.7)

    store.set_harmonic_envelope(4, 0.0)
    voice = store.get_snapshot()[4]
    assert voice.active is True
    assert voice.voice_id == 42
    assert voice.gain == 0.7


def test_store_thread_safety_smoke() -> None:
    store = VoiceParameterStore()
    # Exercise re-entrant reads from the state-change callback too.
    store._on_change = lambda: store.to_dict()
    start = threading.Barrier(5)
    failures: list[BaseException] = []

    def writer(offset: int) -> None:
        try:
            start.wait()
            for index in range(1_500):
                n = ((index + offset) % config.N_BANDS) + 1
                store.voice_on(n, offset * 10_000 + index, store.f1 * n, gain=index / 1_499)
                store.set_pan(n, ((index % 201) - 100) / 100.0)
                store.set_phase(n, index * 7.0)
                if index % 3 == 0:
                    store.voice_off(offset * 10_000 + index)
        except BaseException as exc:  # capture worker failures for the main thread
            failures.append(exc)

    def reader() -> None:
        try:
            start.wait()
            for _ in range(2_000):
                snapshot = store.get_all_snapshot()
                state = store.to_dict()
                assert all(1 <= n <= config.N_BANDS for n in snapshot)
                assert isinstance(state["voices"], dict)
        except BaseException as exc:
            failures.append(exc)

    threads = [
        threading.Thread(target=writer, args=(0,)),
        threading.Thread(target=writer, args=(1,)),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not any(thread.is_alive() for thread in threads)
    assert failures == []
    assert len(store.get_all_snapshot()) <= config.N_BANDS


# ─── partial_ceiling ─────────────────────────────────────────────────────


def test_partial_ceiling_default_and_clamp() -> None:
    store = VoiceParameterStore()
    assert store.get_partial_ceiling() == config.N_BANDS
    assert store.to_dict()["partial_ceiling"] == config.N_BANDS

    store.set_partial_ceiling(5)
    assert store.get_partial_ceiling() == 5

    store.set_partial_ceiling(0)
    assert store.get_partial_ceiling() == 1

    store.set_partial_ceiling(100)
    assert store.get_partial_ceiling() == config.N_BANDS


def test_partial_ceiling_level_mapping() -> None:
    """OSC /digital/ceiling 0.0 → n_max=1, 1.0 → n_max=32."""
    assert VoiceParameterStore.level_to_partial_ceiling(0.0) == 1
    assert VoiceParameterStore.level_to_partial_ceiling(1.0) == 32
    assert VoiceParameterStore.level_to_partial_ceiling(0.5) == 1 + round(0.5 * 31)

    store = VoiceParameterStore()
    store.set_partial_ceiling_from_level(0.0)
    assert store.get_partial_ceiling() == 1
    store.set_partial_ceiling_from_level(1.0)
    assert store.get_partial_ceiling() == 32
    store.set_partial_ceiling_from_level(-0.5)
    assert store.get_partial_ceiling() == 1
    store.set_partial_ceiling_from_level(1.5)
    assert store.get_partial_ceiling() == 32


def test_osc_ceiling_handler_maps_level() -> None:
    store = VoiceParameterStore()
    receiver = object.__new__(ShaperOSCReceiver)
    receiver._store = store

    receiver._on_ceiling("/digital/ceiling", 0.0)
    assert store.get_partial_ceiling() == 1
    receiver._on_ceiling("/digital/ceiling", 1.0)
    assert store.get_partial_ceiling() == 32
    receiver._on_ceiling("/digital/ceiling", 0.5)
    assert store.get_partial_ceiling() == VoiceParameterStore.level_to_partial_ceiling(0.5)


def test_panic_clears_voices_but_not_ceiling() -> None:
    store = VoiceParameterStore()
    store.set_partial_ceiling(8)
    for n in range(1, 9):
        store.voice_on(n, voice_id=n, freq=40.0 * n, gain=0.5)

    assert store.get_snapshot()  # non-empty before panic
    store.panic()

    assert store.get_partial_ceiling() == 8
    assert store.get_snapshot() == {}
    for voice in store.get_all_snapshot().values():
        assert voice.active is False


def _drive_audio_block(engine: AudioEngine, frames: int = 256) -> np.ndarray:
    """Run one audio callback without opening a PortAudio stream."""
    out = np.zeros((frames, 2), dtype=np.float32)
    engine._audio_callback(out, frames, None, None)
    return out


def test_ceiling_drop_releases_high_partials_and_raise_restores() -> None:
    """Ceiling 5 → n=6..32 release (env→0); ceiling 32 → available again."""
    store = VoiceParameterStore()
    store.set_global_attack(0.001)
    store.set_global_release(0.001)
    engine = AudioEngine(store, sample_rate=44_100, block_size=256)

    # Activate a low and several high partials.
    for n in (1, 5, 6, 12, 32):
        store.voice_on(n, voice_id=n, freq=40.0 * n, gain=0.8)
        store.set_attack(n, 0.001)
        store.set_release(n, 0.001)

    # Warm up so envelopes reach ~1.0 under full ceiling.
    for _ in range(20):
        _drive_audio_block(engine)

    for n in (1, 5, 6, 12, 32):
        assert n in engine._voice_state
        assert engine._voice_state[n]["env"] > 0.9

    # Drop ceiling: high partials must enter release (target_env=0 path).
    store.set_partial_ceiling(5)
    for _ in range(40):
        _drive_audio_block(engine)

    assert engine._voice_state[1]["env"] > 0.9
    assert engine._voice_state[5]["env"] > 0.9
    # High partials release fully and may be pruned once env hits 0.
    for n in (6, 12, 32):
        if n in engine._voice_state:
            assert engine._voice_state[n]["env"] < 0.05
        # Store still holds them active so raising the ceiling re-enables them.
        assert store.get_all_snapshot()[n].active is True

    # Raise ceiling: previously masked voices become available and attack again.
    store.set_partial_ceiling(32)
    for _ in range(40):
        _drive_audio_block(engine)

    for n in (1, 5, 6, 12, 32):
        assert n in engine._voice_state
        assert engine._voice_state[n]["env"] > 0.9

