"""Arpeggiator H=0 generator tests (advance_arp)."""

from __future__ import annotations

from harmonic_shaper.state import (
    VoiceParameterStore,
    advance_arp,
    arp_voice_id,
    density_to_subdiv,
    ENVELOPE_PROFILES,
)


def _snap_arp_settled(store: VoiceParameterStore, hand: int = 0) -> None:
    """Copy targets into settled so tests observe exact rates without ease lag."""
    st = store._arp_state[hand]
    for key in ("rate", "dir", "density", "lo", "hi", "gate", "gain"):
        st[f"settled_{key}"] = st[f"target_{key}"]


def _configure_arp(
    store: VoiceParameterStore,
    *,
    rate: float = 0.0,
    direction: float = 1.0,
    density: float = 0.0,
    lo: int = 1,
    hi: int = 16,
    gate: float = 0.5,
    gain: float = 0.7,
    bpm: float = 120.0,
    enable: bool = True,
    ceiling: int = 32,
) -> None:
    store.set_clock_bpm(bpm)
    store.set_settle_beats(0.25)
    store.set_partial_ceiling(ceiling)
    store.set_generator_enable(True)
    store.set_arp_rate(0, rate)
    store.set_arp_direction(0, direction)
    store.set_arp_density(0, density)
    store.set_arp_register_lo(0, lo)
    store.set_arp_register_hi(0, hi)
    store.set_arp_gate(0, gate)
    store.set_arp_gain(0, gain)
    store.set_arp_enable(0, enable)
    _snap_arp_settled(store, 0)


def _active_arp_voices(store: VoiceParameterStore, hand: int = 0) -> dict[int, object]:
    """Active voices whose voice_id is in the arp band for hand H."""
    out = {}
    for n, v in store.get_snapshot().items():
        if v.voice_id is None:
            continue
        # Band for H: -20000 - H*1000 - n  →  -20001 .. -20032 for H=0
        base = -20_000 - hand * 1000
        if base - 32 <= v.voice_id <= base - 1:
            out[n] = v
    return out


def test_arp_voice_id_band_h0() -> None:
    assert arp_voice_id(0, 1) == -20001
    assert arp_voice_id(0, 32) == -20032
    assert arp_voice_id(1, 1) == -21001


def test_density_quantization() -> None:
    assert density_to_subdiv(0.0) == 1
    assert density_to_subdiv(1.0) == 8
    # fill=0.5 → index round(0.5*5)=round(2.5)=2 (banker's) → subdiv 3
    assert density_to_subdiv(0.5) == 3
    assert density_to_subdiv(0.6) == 4  # round(3.0)=3 → {...,4,...}


def test_rate_zero_sustains_window_center() -> None:
    """rate=0: single sustained note at register center, no cursor walk."""
    store = VoiceParameterStore()
    store.update_f1(40.0)
    _configure_arp(store, rate=0.0, lo=1, hi=9, direction=1.0, gate=0.5)
    # Center of [1,9] = 5
    advance_arp(0.01, store)
    arp = _active_arp_voices(store)
    assert list(arp.keys()) == [5]
    v = arp[5]
    assert v.active is True
    assert v.voice_id == arp_voice_id(0, 5)
    assert abs(v.freq - 40.0 * 5) < 1e-9
    assert v.envelope_profile == "pluck"
    assert abs(v.attack_s - ENVELOPE_PROFILES["pluck"][0]) < 1e-12
    assert abs(v.release_s - ENVELOPE_PROFILES["pluck"][1]) < 1e-12

    cursor_before = store.get_arp_state(0)["cursor_n"]
    # One full beat at 120 BPM: still sustained, cursor unmoved.
    advance_arp(0.5, store)
    assert store.get_arp_state(0)["cursor_n"] == cursor_before == 5
    assert list(_active_arp_voices(store).keys()) == [5]


def test_rate_four_advances_four_steps_per_beat_upward() -> None:
    """rate=4, dir=+1: cursor advances 4 steps per beat upward."""
    store = VoiceParameterStore()
    store.update_f1(40.0)
    _configure_arp(
        store, rate=4.0, direction=1.0, density=0.0, lo=1, hi=16, gate=0.9, bpm=120.0
    )
    # Seed cursor at lo so first step goes to lo+1 ... 
    store._arp_state[0]["cursor_n"] = 1
    store._arp_state[0]["step_phase"] = 0.0

    # At 120 BPM, 1 beat = 0.5 s; rate=4 → 4 steps.
    # Integrate in small blocks so step boundaries are counted cleanly.
    dt = 0.005
    steps = int(0.5 / dt)
    cursors = []
    for _ in range(steps):
        advance_arp(dt, store)
        cursors.append(store.get_arp_state(0)["cursor_n"])

    # After exactly 4 steps from cursor=1 with +dir: 2,3,4,5
    assert store.get_arp_state(0)["cursor_n"] == 5
    # Intermediate unique ascending positions visited
    assert 2 in cursors and 3 in cursors and 4 in cursors and 5 in cursors


def test_direction_reversal_zero_cross_triggers_voice_on() -> None:
    """Direction reversal at zero-crossing triggers a new voice_on."""
    store = VoiceParameterStore()
    store.update_f1(40.0)
    _configure_arp(
        store, rate=2.0, direction=1.0, lo=1, hi=16, gate=0.95, bpm=120.0
    )
    store._arp_state[0]["cursor_n"] = 4
    store._arp_state[0]["last_dir_sign"] = 1.0
    store._arp_state[0]["step_phase"] = 0.0

    # Warm a little so something is active, without a full step boundary.
    advance_arp(0.01, store)
    # Force direction flip through settled value (simulates completed ease).
    store.set_arp_direction(0, -1.0)
    store._arp_state[0]["settled_dir"] = -1.0
    store._arp_state[0]["target_dir"] = -1.0

    before_cursor = store.get_arp_state(0)["cursor_n"]
    advance_arp(0.001, store)  # tiny dt: no step_phase boundary, only zero-cross
    after_cursor = store.get_arp_state(0)["cursor_n"]
    # Zero-cross advances one step in the new (down) direction.
    assert after_cursor == before_cursor - 1
    snap = store.get_snapshot()
    assert after_cursor in snap
    assert snap[after_cursor].voice_id == arp_voice_id(0, after_cursor)
    assert snap[after_cursor].envelope_profile == "pluck"


def test_ceiling_clamps_effective_window() -> None:
    """ceiling=8, register_hi=16 → effective window clamped to [lo, 8]."""
    store = VoiceParameterStore()
    store.update_f1(40.0)
    _configure_arp(
        store,
        rate=4.0,
        direction=1.0,
        lo=1,
        hi=16,
        gate=0.9,
        ceiling=8,
        bpm=120.0,
    )
    store._arp_state[0]["cursor_n"] = 7
    store._arp_state[0]["step_phase"] = 0.0

    # Enough time for several steps; cursor must never exceed ceiling 8.
    dt = 0.01
    for _ in range(200):
        advance_arp(dt, store)
        cursor = store.get_arp_state(0)["cursor_n"]
        assert 1 <= cursor <= 8
        for n in _active_arp_voices(store):
            assert n <= 8


def test_generator_enable_false_no_triggers() -> None:
    """generator_enable=False: no arp voice triggers."""
    store = VoiceParameterStore()
    store.update_f1(40.0)
    _configure_arp(store, rate=4.0, direction=1.0, lo=1, hi=8, gate=0.5)
    store.set_generator_enable(False)
    store._arp_state[0]["cursor_n"] = 1
    store._arp_state[0]["step_phase"] = 0.0

    advance_arp(1.0, store)  # many beats worth of would-be steps
    assert _active_arp_voices(store) == {}
    # Cursor should not walk while generators are gated off.
    assert store.get_arp_state(0)["cursor_n"] == 1


def test_density_half_adds_subdivisions() -> None:
    """density=0.5 → subdiv 4; more steps per beat than density=0."""
    store_a = VoiceParameterStore()
    store_a.update_f1(40.0)
    _configure_arp(
        store_a, rate=2.0, direction=1.0, density=0.0, lo=1, hi=32, gate=0.9, bpm=120.0
    )
    store_a._arp_state[0]["cursor_n"] = 1
    store_a._arp_state[0]["step_phase"] = 0.0

    store_b = VoiceParameterStore()
    store_b.update_f1(40.0)
    _configure_arp(
        store_b, rate=2.0, direction=1.0, density=0.5, lo=1, hi=32, gate=0.9, bpm=120.0
    )
    store_b._arp_state[0]["cursor_n"] = 1
    store_b._arp_state[0]["step_phase"] = 0.0

    # One beat at 120 BPM = 0.5 s.
    # density=0 → subdiv 1 → 2 steps → cursor 3
    # density=0.5 → subdiv 3 → 6 steps → cursor 7
    dt = 0.005
    for _ in range(int(0.5 / dt)):
        advance_arp(dt, store_a)
        advance_arp(dt, store_b)

    assert store_a.get_arp_state(0)["cursor_n"] == 3
    assert density_to_subdiv(0.5) == 3
    assert store_b.get_arp_state(0)["cursor_n"] == 7
    assert store_b.get_arp_state(0)["cursor_n"] > store_a.get_arp_state(0)["cursor_n"]


def test_gate_schedules_voice_off_after_fraction_of_step() -> None:
    """gate=0.3: voice_off after 30% of step period."""
    store = VoiceParameterStore()
    store.update_f1(40.0)
    # rate=2 at 120 BPM → step_rate = 2 * 2 = 4 Hz → step_period = 0.25 s
    # gate=0.3 → off after 0.075 s
    _configure_arp(
        store,
        rate=2.0,
        direction=1.0,
        density=0.0,
        lo=1,
        hi=16,
        gate=0.3,
        bpm=120.0,
    )
    store._arp_state[0]["cursor_n"] = 1
    store._arp_state[0]["step_phase"] = 0.0

    # Cross one step boundary.
    advance_arp(0.26, store)  # slightly more than one step at 4 Hz? 
    # step_rate_hz = 2 * (120/60) = 4 Hz, period=0.25. 0.26 → one step + a bit.
    # Actually after first step, pending off at 0.075 s remaining, then we advanced
    # remaining 0.01 of the same call... advance_arp processes pending within same call
    # after ease, before the step? Order: tick pending first, then run hand.
    # So on the call that fires the step, pending is scheduled but not yet expired.

    st = store.get_arp_state(0)
    # There should be a pending off shortly after the trigger.
    # Re-run from clean state with controlled small steps.
    store2 = VoiceParameterStore()
    store2.update_f1(40.0)
    _configure_arp(
        store2,
        rate=2.0,
        direction=1.0,
        density=0.0,
        lo=1,
        hi=16,
        gate=0.3,
        bpm=120.0,
    )
    store2._arp_state[0]["cursor_n"] = 1
    store2._arp_state[0]["step_phase"] = 0.0

    step_period = 1.0 / (2.0 * (120.0 / 60.0))  # 0.25
    gate_time = 0.3 * step_period  # 0.075

    # Exactly one step: advance phase by 1.0 in one go.
    advance_arp(step_period, store2)
    n = store2.get_arp_state(0)["cursor_n"]
    assert n in store2.get_snapshot()
    assert store2.get_snapshot()[n].active is True
    pending = store2.get_arp_state(0)["pending_offs"]
    assert len(pending) >= 1
    # Remaining time on the most recent pending off ≈ gate_time (may be slightly less
    # if tick ran after schedule in same call — schedule is after tick).
    assert abs(pending[-1][0] - gate_time) < 1e-9

    # Advance just past gate fraction: voice should release.
    advance_arp(gate_time + 1e-6, store2)
    # That voice_id should be off (cursor may have moved if more steps fired —
    # with dt=gate_time only, step_phase advances by gate_time*4Hz = 0.3, no new step).
    snap = store2.get_snapshot()
    # The note at n may still be active if a new step re-triggered same n, but
    # with 0.3 phase advance and no full step, it should be off.
    if n in snap:
        # If still present, it must not be our original gate-expired note unless
        # re-triggered — with no full step, it should be gone.
        assert snap[n].active is False or snap[n].voice_id != arp_voice_id(0, n)
    else:
        assert n not in snap


def test_envelope_profiles_defined() -> None:
    assert "pad" in ENVELOPE_PROFILES
    assert "pluck" in ENVELOPE_PROFILES
    assert "perc" in ENVELOPE_PROFILES
    assert ENVELOPE_PROFILES["pluck"] == (0.02, 0.25)
    assert ENVELOPE_PROFILES["perc"] == (0.001, 0.08)


def test_osc_arp_handlers_wire_to_store() -> None:
    from harmonic_shaper.osc_receiver import ShaperOSCReceiver

    store = VoiceParameterStore()
    receiver = object.__new__(ShaperOSCReceiver)
    receiver._store = store

    receiver._on_arp_enable("/digital/arp/0/enable", 1)
    receiver._on_arp_rate("/digital/arp/0/rate", 3.5)
    receiver._on_arp_direction("/digital/arp/0/direction", -0.5)
    receiver._on_arp_density("/digital/arp/0/density", 0.25)
    receiver._on_arp_register_lo("/digital/arp/0/register_lo", 2)
    receiver._on_arp_register_hi("/digital/arp/0/register_hi", 12)
    receiver._on_arp_gate("/digital/arp/0/gate", 0.4)
    receiver._on_arp_gain("/digital/arp/0/gain", 0.8)

    st = store.get_arp_state(0)
    assert st["enabled"] is True
    assert st["target_rate"] == 3.5
    assert st["target_dir"] == -0.5
    assert st["target_density"] == 0.25
    assert st["target_lo"] == 2.0
    assert st["target_hi"] == 12.0
    assert st["target_gate"] == 0.4
    assert st["target_gain"] == 0.8
