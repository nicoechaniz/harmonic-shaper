"""VoiceParameterStore behavior and concurrency smoke tests."""

from __future__ import annotations

import threading

from harmonic_shaper import config
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
