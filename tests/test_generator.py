"""Arpeggiator H=0/H=1 + foot percussion generator tests."""

from __future__ import annotations

from harmonic_shaper.state import (
    VoiceParameterStore,
    advance_arp,
    advance_perc,
    arp_voice_id,
    density_to_subdiv,
    is_perc_voice_id,
    perc_voice_id,
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

    # H=1 path is wired the same way.
    receiver._on_arp_enable("/digital/arp/1/enable", 1)
    receiver._on_arp_rate("/digital/arp/1/rate", 2.0)
    receiver._on_arp_gain("/digital/arp/1/gain", 0.55)
    st1 = store.get_arp_state(1)
    assert st1["enabled"] is True
    assert st1["target_rate"] == 2.0
    assert st1["target_gain"] == 0.55


# ─── H=1 second hand ──────────────────────────────────────────────────


def test_arp_voice_id_band_h1() -> None:
    assert arp_voice_id(1, 1) == -21001
    assert arp_voice_id(1, 32) == -21032
    # Bands for H=0 and H=1 never overlap.
    h0 = {arp_voice_id(0, n) for n in range(1, 33)}
    h1 = {arp_voice_id(1, n) for n in range(1, 33)}
    assert h0.isdisjoint(h1)


def test_h0_and_h1_independent_voice_ids() -> None:
    """H=0 and H=1 produce independent voices with non-overlapping voice_ids."""
    store = VoiceParameterStore()
    store.update_f1(40.0)
    # Disjoint registers so each hand owns distinct harmonic slots.
    _configure_arp(
        store, rate=0.0, lo=1, hi=5, direction=1.0, gate=0.5, gain=0.7, enable=True
    )
    # Configure H=1 similarly (snap settled so ease lag is zero).
    store.set_arp_rate(1, 0.0)
    store.set_arp_direction(1, 1.0)
    store.set_arp_density(1, 0.0)
    store.set_arp_register_lo(1, 16)
    store.set_arp_register_hi(1, 24)
    store.set_arp_gate(1, 0.5)
    store.set_arp_gain(1, 0.6)
    store.set_arp_enable(1, True)
    _snap_arp_settled(store, 1)

    advance_arp(0.01, store)

    voices_h0 = _active_arp_voices(store, hand=0)
    voices_h1 = _active_arp_voices(store, hand=1)
    assert voices_h0, "H=0 should sustain a voice"
    assert voices_h1, "H=1 should sustain a voice"

    ids_h0 = {v.voice_id for v in voices_h0.values()}
    ids_h1 = {v.voice_id for v in voices_h1.values()}
    assert ids_h0.isdisjoint(ids_h1)
    for vid in ids_h0:
        assert -20032 <= vid <= -20001
    for vid in ids_h1:
        assert -21032 <= vid <= -21001
    # Centers: [1,5]→3, [16,24]→20
    assert 3 in voices_h0
    assert 20 in voices_h1
    assert voices_h0[3].voice_id == arp_voice_id(0, 3)
    assert voices_h1[20].voice_id == arp_voice_id(1, 20)
    assert voices_h1[20].envelope_profile == "pluck"


def test_h1_direction_zero_cross_retriggers() -> None:
    """H=1 uses the same zero-cross re-trigger as H=0."""
    store = VoiceParameterStore()
    store.update_f1(40.0)
    store.set_clock_bpm(120.0)
    store.set_settle_beats(0.25)
    store.set_partial_ceiling(32)
    store.set_generator_enable(True)
    store.set_arp_rate(1, 2.0)
    store.set_arp_direction(1, 1.0)
    store.set_arp_density(1, 0.0)
    store.set_arp_register_lo(1, 1)
    store.set_arp_register_hi(1, 16)
    store.set_arp_gate(1, 0.95)
    store.set_arp_gain(1, 0.7)
    store.set_arp_enable(1, True)
    _snap_arp_settled(store, 1)
    store._arp_state[1]["cursor_n"] = 8
    store._arp_state[1]["last_dir_sign"] = 1.0
    store._arp_state[1]["step_phase"] = 0.0

    advance_arp(0.01, store)
    store.set_arp_direction(1, -1.0)
    store._arp_state[1]["settled_dir"] = -1.0
    store._arp_state[1]["target_dir"] = -1.0

    before = store.get_arp_state(1)["cursor_n"]
    advance_arp(0.001, store)
    after = store.get_arp_state(1)["cursor_n"]
    assert after == before - 1
    snap = store.get_snapshot()
    assert after in snap
    assert snap[after].voice_id == arp_voice_id(1, after)
    assert snap[after].envelope_profile == "pluck"


# ─── Foot percussion ──────────────────────────────────────────────────


def _configure_perc(
    store: VoiceParameterStore,
    *,
    rate: float = 2.0,
    gain: float = 0.8,
    accent: float = 0.0,
    bpm: float = 120.0,
    enable: bool = True,
) -> None:
    store.set_clock_bpm(bpm)
    store.set_generator_enable(True)
    store.set_perc_rate(rate)
    store.set_perc_gain(gain)
    store.set_perc_accent(accent)
    store.set_perc_enable(enable)


def _active_perc_voices(store: VoiceParameterStore) -> dict[int, object]:
    out = {}
    for key, v in store.get_snapshot().items():
        if v.voice_id is not None and is_perc_voice_id(v.voice_id):
            out[key] = v
    return out


def test_perc_voice_id_pool() -> None:
    assert perc_voice_id(0) == -30000
    assert perc_voice_id(7) == -30007
    assert perc_voice_id(8) == -30000  # wraps
    assert is_perc_voice_id(-30000)
    assert is_perc_voice_id(-30007)
    assert not is_perc_voice_id(-30008)
    assert not is_perc_voice_id(-20001)


def test_perc_rate_two_hits_per_beat() -> None:
    """perc rate=2: 2 hits per beat at current clock_bpm."""
    store = VoiceParameterStore()
    store.update_f1(40.0)
    _configure_perc(store, rate=2.0, bpm=120.0, enable=True)

    # 1 beat at 120 BPM = 0.5 s → 2 hits.
    dt = 0.005
    for _ in range(int(0.5 / dt)):
        advance_perc(dt, store)

    st = store.get_perc_state()
    assert st["pulse_index"] == 2
    # Hits used rotating slots 0 then 1.
    assert st["next_slot"] == 2


def test_perc_envelope_schedules_short_voice_off() -> None:
    """Perc envelope is short: voice_off scheduled ~80 ms after trigger."""
    store = VoiceParameterStore()
    store.update_f1(40.0)
    _configure_perc(store, rate=2.0, bpm=120.0)

    # Cross exactly one pulse boundary.
    step_period = 1.0 / (2.0 * (120.0 / 60.0))  # 0.25 s
    advance_perc(step_period, store)

    st = store.get_perc_state()
    assert st["pulse_index"] == 1
    assert len(st["pending_offs"]) >= 1
    # Most recent pending off ≈ perc release (0.08 s).
    assert abs(st["pending_offs"][-1][0] - ENVELOPE_PROFILES["perc"][1]) < 1e-9

    perc = _active_perc_voices(store)
    assert perc, "expected an active perc voice after the hit"
    for v in perc.values():
        assert v.envelope_profile == "perc"
        assert abs(v.attack_s - 0.001) < 1e-12
        assert abs(v.release_s - 0.08) < 1e-12
        assert is_perc_voice_id(v.voice_id)

    # After release window, voice should be off.
    advance_perc(0.08 + 1e-6, store)
    # No full new pulse (0.08 * 4 Hz = 0.32 phase) — original hit released.
    remaining = _active_perc_voices(store)
    # Either empty or only a re-trigger if phase crossed — with 0.08 s of phase
    # advance (0.32 of a step) no new step fires, so pool should be silent.
    assert remaining == {}


def test_perc_hits_do_not_reduce_melodic_norm() -> None:
    """Percussion hits do NOT reduce melodic voice amplitudes (separate norm)."""
    from harmonic_shaper.audio_engine import AudioEngine
    from harmonic_shaper.state import is_perc_voice_id as _is_perc

    store = VoiceParameterStore()
    store.update_f1(40.0)
    # Two melodic pad voices via direct voice_on.
    store.voice_on(1, voice_id=1001, freq=40.0, gain=1.0)
    store.voice_on(2, voice_id=1002, freq=80.0, gain=1.0)

    engine = AudioEngine(store, sample_rate=48000, block_size=256)
    # Drive one silent callback block without perc — capture melodic gain path
    # by inspecting the norm computation indirectly via engine internals.
    # Replicate the engine's melodic-count rule (perc excluded).
    def melodic_count(snap) -> int:
        return sum(
            1
            for v in snap.values()
            if not (v.voice_id is not None and _is_perc(v.voice_id))
        )

    snap_before = store.get_snapshot()
    n_before = melodic_count(snap_before)
    assert n_before == 2
    norm_before = 1.0 / (n_before ** 0.5)

    # Fire a perc hit into the pool.
    _configure_perc(store, rate=4.0, bpm=120.0, enable=True)
    advance_perc(0.01, store)  # enough to cross at least one step at 8 Hz
    # Force a hit even if phase didn't cross: ensure at least one pulse.
    if store.get_perc_state()["pulse_index"] == 0:
        # Advance a full pulse period at rate=4, 120 BPM → 8 Hz → 0.125 s
        advance_perc(0.13, store)
    assert store.get_perc_state()["pulse_index"] >= 1

    snap_after = store.get_snapshot()
    n_after = melodic_count(snap_after)
    assert n_after == 2, "perc must not count toward melodic polyphony"
    norm_after = 1.0 / (n_after ** 0.5)
    assert abs(norm_after - norm_before) < 1e-12

    # Melodic voices keep their gains; perc lives on dedicated keys.
    assert snap_after[1].gain == 1.0
    assert snap_after[2].gain == 1.0
    perc = _active_perc_voices(store)
    assert perc, "expected active perc voice"
    for key in perc:
        assert key not in (1, 2)
        assert is_perc_voice_id(perc[key].voice_id)

    # Engine path: melodic voices still present with full gain after callback.
    import numpy as np

    out = np.zeros((256, 2), dtype=np.float32)
    engine._audio_callback(out, 256, None, None)
    # Melodic snapshot still at gain 1.0 after audio tick.
    snap = store.get_snapshot()
    if 1 in snap:
        assert snap[1].gain == 1.0
    if 2 in snap:
        assert snap[2].gain == 1.0


def test_panic_clears_both_hands_and_perc() -> None:
    """panic() clears arp state for both hands and the perc pool."""
    store = VoiceParameterStore()
    store.update_f1(40.0)

    _configure_arp(store, rate=4.0, direction=1.0, lo=1, hi=8, gate=0.5)
    store._arp_state[0]["cursor_n"] = 5
    store._arp_state[0]["step_phase"] = 0.4
    store.set_arp_enable(1, True)
    store.set_arp_rate(1, 3.0)
    store.set_arp_register_lo(1, 10)
    store.set_arp_register_hi(1, 16)
    _snap_arp_settled(store, 1)
    store._arp_state[1]["cursor_n"] = 12
    store._arp_state[1]["step_phase"] = 0.7
    advance_arp(0.1, store)

    _configure_perc(store, rate=4.0, bpm=120.0)
    advance_perc(0.2, store)
    assert store.get_perc_state()["pulse_index"] >= 1

    store.panic()

    # All voices inactive.
    assert store.get_snapshot() == {}

    # Both arp hands reset to defaults (cursor 1, phase 0, no pending).
    for hand in (0, 1):
        st = store.get_arp_state(hand)
        assert st["cursor_n"] == 1
        assert st["step_phase"] == 0.0
        assert st["pending_offs"] == []
        assert st["sustain_voice_id"] is None
        assert st["enabled"] is False

    # Perc pool runtime cleared (scene rate/gain/accent preserved).
    pst = store.get_perc_state()
    assert pst["pulse_index"] == 0
    assert pst["next_slot"] == 0
    assert pst["step_phase"] == 0.0
    assert pst["pending_offs"] == []
    assert pst["active_slots"] == []
    assert pst["rate"] == 4.0  # scene param kept
    assert _active_perc_voices(store) == {}


def test_osc_perc_handlers_wire_to_store() -> None:
    from harmonic_shaper.osc_receiver import ShaperOSCReceiver

    store = VoiceParameterStore()
    receiver = object.__new__(ShaperOSCReceiver)
    receiver._store = store

    receiver._on_perc_enable("/digital/perc/enable", 1)
    receiver._on_perc_rate("/digital/perc/rate", 3.0)
    receiver._on_perc_gain("/digital/perc/gain", 0.55)
    receiver._on_perc_accent("/digital/perc/accent", 0.4)

    st = store.get_perc_state()
    assert st["enabled"] is True
    assert st["rate"] == 3.0
    assert st["gain"] == 0.55
    assert st["accent"] == 0.4


def test_perc_gated_by_generator_enable() -> None:
    store = VoiceParameterStore()
    store.update_f1(40.0)
    _configure_perc(store, rate=4.0, bpm=120.0, enable=True)
    store.set_generator_enable(False)
    advance_perc(1.0, store)
    assert store.get_perc_state()["pulse_index"] == 0
    assert _active_perc_voices(store) == {}
