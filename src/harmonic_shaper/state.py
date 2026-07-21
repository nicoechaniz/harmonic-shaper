"""Thread-safe voice parameter store for the Shaper.

Adapted from NaturalHarmony/harmonic_shaper/state.py:
- Raised polyphony from 5 to MAX_VOICES (32).
- Voice identity is `harmonic_n` (1..32). Multiple voice_ids can map to the
  same harmonic_n (layering) — we keep the most recent active one.
- f1 is exposed as a first-class store attribute so the OSC receiver can
  push updates without touching VoiceParams.
"""

import math
import threading
from copy import copy
from dataclasses import dataclass
from typing import Callable, Optional

from . import config


# Envelope profiles applied at voice_on / scheduled voice_off.
# ``pad`` keeps the existing linear AR defaults (zero regression).
ENVELOPE_PROFILES: dict[str, tuple[float, float]] = {
    "pad": (config.DEFAULT_VOICE_ATTACK_S, config.DEFAULT_VOICE_RELEASE_S),
    "pluck": (0.02, 0.25),
    "perc": (0.001, 0.08),
}

# Density fill 0..1 → subdivision factor, quantized for musical steps.
_ARP_DENSITY_SUBDIVS: tuple[int, ...] = (1, 2, 3, 4, 6, 8)

# Arpeggiator voice_id band: -20000 - H*1000 - n  (H hand, n harmonic 1..32).
_ARP_VOICE_ID_BASE = -20_000
_ARP_HANDS = (0, 1)  # H=0 and H=1; H=2.. reserved
_ARP_RATE_EPS = 0.01
_ARP_DIR_EPS = 1e-6

# Foot percussion pool: voice_id -30000 .. -30007 (8 rotating slots).
_PERC_VOICE_ID_BASE = -30_000
_PERC_POOL_SIZE = 8
_PERC_RELEASE_S = ENVELOPE_PROFILES["perc"][1]  # schedule voice_off after release
# Timbre MVP: cycle low partials n ∈ {1, 2, 3}.
_PERC_PARTIALS: tuple[int, ...] = (1, 2, 3)


def arp_voice_id(hand: int, harmonic_n: int) -> int:
    """Dedicated voice_id for arpeggiator hand H, partial n."""
    return _ARP_VOICE_ID_BASE - int(hand) * 1000 - int(harmonic_n)


def perc_voice_id(slot: int) -> int:
    """Dedicated voice_id for percussion pool slot k (0..7) → -30000..-30007."""
    k = int(slot) % _PERC_POOL_SIZE
    return _PERC_VOICE_ID_BASE - k


def is_perc_voice_id(voice_id: Optional[int]) -> bool:
    """True if voice_id belongs to the foot-percussion pool."""
    if voice_id is None:
        return False
    return _PERC_VOICE_ID_BASE - (_PERC_POOL_SIZE - 1) <= int(voice_id) <= _PERC_VOICE_ID_BASE


def density_to_subdiv(fill: float) -> int:
    """Map density fill 0..1 to a quantized subdivision in {1,2,3,4,6,8}."""
    fill = max(0.0, min(1.0, float(fill)))
    idx = int(round(fill * (len(_ARP_DENSITY_SUBDIVS) - 1)))
    return _ARP_DENSITY_SUBDIVS[idx]


def _default_arp_state(hand: int = 0) -> dict:
    """Fresh per-hand arpeggiator runtime + target/settled params."""
    del hand  # reserved for multi-hand defaults later
    return {
        "enabled": False,
        "cursor_n": 1,
        "step_phase": 0.0,       # 0..1 within one step
        "last_dir_sign": 1.0,    # last non-zero direction sign
        # targets (OSC/REST write surface)
        "target_rate": 0.0,
        "target_dir": 0.0,
        "target_density": 0.0,
        "target_lo": 1.0,
        "target_hi": float(config.N_BANDS),
        "target_gate": 0.5,
        "target_gain": 0.6,
        # settled (eased toward targets over settle_beats)
        "settled_rate": 0.0,
        "settled_dir": 0.0,
        "settled_density": 0.0,
        "settled_lo": 1.0,
        "settled_hi": float(config.N_BANDS),
        "settled_gate": 0.5,
        "settled_gain": 0.6,
        # scheduled voice_off: list of (remaining_s, voice_id)
        "pending_offs": [],
        # rate≈0 sustain bookkeeping
        "sustain_voice_id": None,
        "sustain_n": None,
    }


def _default_perc_state() -> dict:
    """Fresh foot-percussion runtime state."""
    return {
        "enabled": False,
        "rate": 0.0,          # pulses_per_beat 0..8
        "gain": 0.7,
        "accent": 0.0,        # 0..1 downbeat accent
        "step_phase": 0.0,    # 0..1 within one pulse
        "pulse_index": 0,     # counts pulses; 0 ≡ downbeat when rate≥1
        "next_slot": 0,       # rotating oldest-first pool index 0..7
        # scheduled voice_off: list of (remaining_s, voice_id)
        "pending_offs": [],
        # active pool slots for diagnostics (voice_id list, oldest first)
        "active_slots": [],
    }


@dataclass
class VoiceParams:
    """Parameters for a single Shaper voice (one pure sine)."""
    harmonic_n: int = 0
    freq: float = 0.0          # Hz — set by beacon broadcast
    gain: float = config.DEFAULT_VOICE_GAIN
    pan: float = config.DEFAULT_VOICE_PAN       # -1..+1
    phase: float = config.DEFAULT_VOICE_PHASE_DEG  # radians
    attack_s: float = config.DEFAULT_VOICE_ATTACK_S
    release_s: float = config.DEFAULT_VOICE_RELEASE_S
    shape: float = config.DEFAULT_VOICE_SHAPE   # 0=pure sine, 1=rich
    lfo_gain: float = config.DEFAULT_LFO_GAIN   # LFO→gain mod amount
    lfo_pan: float = config.DEFAULT_LFO_PAN
    lfo_phase: float = config.DEFAULT_LFO_PHASE
    active: bool = False
    voice_id: Optional[int] = None
    # Named AR preset: "pad" (default), "pluck" (arp), "perc" (later).
    envelope_profile: str = "pad"

    def copy(self) -> "VoiceParams":
        return copy(self)

    def apply_envelope_profile(self, profile: str = None) -> None:
        """Set attack_s/release_s from a named profile (pad/pluck/perc)."""
        name = profile if profile is not None else self.envelope_profile
        if name not in ENVELOPE_PROFILES:
            name = "pad"
        self.envelope_profile = name
        attack, release = ENVELOPE_PROFILES[name]
        self.attack_s = attack
        self.release_s = release

    def to_dict(self) -> dict:
        return {
            "harmonic_n": self.harmonic_n,
            "freq": round(self.freq, 3),
            "gain": round(self.gain, 4),
            "pan": round(self.pan, 4),
            "phase_deg": round(math.degrees(self.phase) % 360, 1),
            "attack_s": self.attack_s,
            "release_s": self.release_s,
            "shape": self.shape,
            "lfo_gain": self.lfo_gain,
            "lfo_pan": self.lfo_pan,
            "lfo_phase": self.lfo_phase,
            "active": self.active,
            "envelope_profile": self.envelope_profile,
        }


class VoiceParameterStore:
    """Thread-safe store for per-harmonic Shaper parameters.

    Keyed by harmonic_n (1..32). The beacon populates voice_on/off/freq;
    control surfaces (MIDI/OSC/Web) set gain/pan/phase.
    """

    # Negative IDs are reserved for native OSC envelope control. This keeps
    # release ownership separate from MIDI and external beacon voice IDs.
    _ENVELOPE_VOICE_ID_BASE = -10_000

    def __init__(self, on_change: Optional[Callable[[], None]] = None):
        self._lock = threading.RLock()
        self._voices: dict[int, VoiceParams] = {}
        self._active_history: list[int] = []  # chronological, for note stealing
        self.f1: float = config.DEFAULT_F1
        self._base_f1: float = config.DEFAULT_F1    # before vsrate
        self._vsrate: float = 1.0
        self._master_gain: float = config.DEFAULT_SHAPER_MASTER
        self._global_attack_s: float = config.DEFAULT_VOICE_ATTACK_S
        self._global_release_s: float = config.DEFAULT_VOICE_RELEASE_S
        # Side-chain
        self._beacon_level: float = 0.0          # updated via OSC /beacon/level
        self._sidechain_amount: float = config.DEFAULT_SIDECHAIN_AMOUNT
        # LFO
        self._lfo_rate_divisor: int = config.DEFAULT_LFO_RATE_DIVISOR
        self._lfo_waveform: str = config.DEFAULT_LFO_WAVEFORM
        self._lfo_amount: float = config.DEFAULT_LFO_AMOUNT
        self._lfo_phase: float = 0.0             # 0..1, advances in audio callback
        self._strum_period_s: float = config.DEFAULT_STRUM_PERIOD_S
        self._strum_times: list[float] = []       # recent strum timestamps
        # Scene mask over the fixed 1..32 harmonic grid (not voice state).
        # High partials above the ceiling enter natural release in the audio
        # callback; panic does not reset this value.
        self._partial_ceiling: int = config.N_BANDS
        # Musical clock (Shaper-owned). Weaver may converge clock_bpm from body
        # tempo; settle_beats is the ease time constant in local beats, not ms.
        self._clock_bpm: float = 90.0
        self._settle_beats: float = 1.0
        self._generator_enable: bool = True
        self._beat_phase: float = 0.0  # 0..1, advances at bpm/60 Hz
        # Arpeggiator per hand H (H=0, H=1). Settled params ease over settle_beats.
        self._arp_state: dict[int, dict] = {
            h: _default_arp_state(h) for h in _ARP_HANDS
        }
        # Foot percussion pool (voice_id -30000..-30007); separate from melodic norm.
        self._perc_enabled: bool = False
        self._perc_rate: float = 0.0
        self._perc_gain: float = 0.7
        self._perc_accent: float = 0.0
        self._perc_state: dict = _default_perc_state()
        self._on_change = on_change
        self._panic_callback: Optional[Callable[[], None]] = None

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _notify(self) -> None:
        if self._on_change:
            try:
                self._on_change()
            except Exception:
                pass

    def _ensure(self, n: int) -> None:
        if n not in self._voices:
            v = VoiceParams(harmonic_n=n)
            v.attack_s = self._global_attack_s
            v.release_s = self._global_release_s
            self._voices[n] = v

    # ─── Beacon-driven lifecycle ──────────────────────────────────────────

    def voice_on(
        self,
        harmonic_n: int,
        voice_id: int,
        freq: float,
        gain: float = None,
        envelope_profile: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._ensure(harmonic_n)
            v = self._voices[harmonic_n]
            v.voice_id = voice_id
            v.freq = freq
            v.active = True
            if gain is not None:
                v.gain = max(0.0, min(1.0, float(gain)))
            else:
                v.gain = config.DEFAULT_VOICE_GAIN
            if envelope_profile is not None:
                v.apply_envelope_profile(envelope_profile)

            if harmonic_n in self._active_history:
                self._active_history.remove(harmonic_n)
            self._active_history.append(harmonic_n)

            # Note stealing: drop oldest if over limit
            while len(self._active_history) > config.MAX_VOICES:
                oldest_n = self._active_history.pop(0)
                self._voices[oldest_n].active = False
        self._notify()

    def voice_off(self, voice_id: int) -> None:
        """Deactivate the voice bound to ``voice_id``.

        If the voice has a named envelope profile, re-apply its release time so
        the audio callback ramps out with the profile's release (pluck/perc).
        """
        with self._lock:
            for n, v in self._voices.items():
                if v.voice_id == voice_id:
                    # Re-apply profile so release_s matches the note's envelope.
                    if v.envelope_profile in ENVELOPE_PROFILES:
                        v.apply_envelope_profile(v.envelope_profile)
                    v.active = False
                    if n in self._active_history:
                        self._active_history.remove(n)
                    break
        self._notify()

    def voice_freq(self, voice_id: int, freq: float) -> None:
        with self._lock:
            for v in self._voices.values():
                if v.voice_id == voice_id:
                    v.freq = freq
                    break
        self._notify()

    def update_f1(self, f1: float) -> None:
        with self._lock:
            self._base_f1 = max(config.F1_MIN, min(config.F1_MAX, float(f1)))
            self.f1 = self._base_f1 * self._vsrate
            for harmonic_n, voice in self._voices.items():
                if voice.voice_id == self._envelope_voice_id(harmonic_n):
                    voice.freq = self.f1 * harmonic_n
        self._notify()

    def set_vsrate(self, rate: float) -> None:
        with self._lock:
            self._vsrate = max(0.1, min(4.0, float(rate)))
            self.f1 = self._base_f1 * self._vsrate
            for harmonic_n, voice in self._voices.items():
                if voice.voice_id == self._envelope_voice_id(harmonic_n):
                    voice.freq = self.f1 * harmonic_n
        self._notify()

    # ─── Parameter control ────────────────────────────────────────────────

    @classmethod
    def _envelope_voice_id(cls, harmonic_n: int) -> int:
        return cls._ENVELOPE_VOICE_ID_BASE - harmonic_n

    def set_harmonic_envelope(self, harmonic_n: int, gain: float) -> None:
        """Set a source-owned native envelope for one series harmonic.

        A positive envelope owns and activates its harmonic at ``f1*n``. Zero
        releases only the matching envelope voice, preserving a differently
        owned MIDI or beacon voice on the same harmonic.
        """

        envelope_gain = max(0.0, min(1.0, float(gain)))
        voice_id = self._envelope_voice_id(harmonic_n)
        with self._lock:
            self._ensure(harmonic_n)
            voice = self._voices[harmonic_n]
            if envelope_gain == 0.0:
                if voice.voice_id == voice_id:
                    voice.active = False
                    voice.gain = 0.0
                    if harmonic_n in self._active_history:
                        self._active_history.remove(harmonic_n)
            else:
                voice.voice_id = voice_id
                voice.freq = self.f1 * harmonic_n
                voice.gain = envelope_gain
                voice.active = True
                if harmonic_n in self._active_history:
                    self._active_history.remove(harmonic_n)
                self._active_history.append(harmonic_n)
                while len(self._active_history) > config.MAX_VOICES:
                    oldest_n = self._active_history.pop(0)
                    self._voices[oldest_n].active = False
        self._notify()

    def set_gain(self, harmonic_n: int, gain: float) -> None:
        with self._lock:
            self._ensure(harmonic_n)
            self._voices[harmonic_n].gain = max(0.0, min(1.0, float(gain)))
        self._notify()

    def set_pan(self, harmonic_n: int, pan: float) -> None:
        with self._lock:
            self._ensure(harmonic_n)
            self._voices[harmonic_n].pan = max(-1.0, min(1.0, float(pan)))
        self._notify()

    def set_phase(self, harmonic_n: int, phase_deg: float) -> None:
        with self._lock:
            self._ensure(harmonic_n)
            self._voices[harmonic_n].phase = math.radians(float(phase_deg) % 360)
        self._notify()

    def set_attack(self, harmonic_n: int, attack_s: float) -> None:
        with self._lock:
            self._ensure(harmonic_n)
            self._voices[harmonic_n].attack_s = max(0.0, min(5.0, float(attack_s)))
        self._notify()

    def set_release(self, harmonic_n: int, release_s: float) -> None:
        with self._lock:
            self._ensure(harmonic_n)
            self._voices[harmonic_n].release_s = max(0.0, min(5.0, float(release_s)))
        self._notify()

    def set_params(self, harmonic_n: int, **kwargs) -> None:
        with self._lock:
            self._ensure(harmonic_n)
            v = self._voices[harmonic_n]
            if "gain" in kwargs:
                v.gain = max(0.0, min(1.0, float(kwargs["gain"])))
            if "pan" in kwargs:
                v.pan = max(-1.0, min(1.0, float(kwargs["pan"])))
            if "phase_deg" in kwargs:
                v.phase = math.radians(float(kwargs["phase_deg"]) % 360)
            if "attack_s" in kwargs:
                v.attack_s = max(0.0, min(5.0, float(kwargs["attack_s"])))
            if "release_s" in kwargs:
                v.release_s = max(0.0, min(5.0, float(kwargs["release_s"])))
        self._notify()

    def set_master_gain(self, gain: float) -> None:
        with self._lock:
            self._master_gain = max(0.0, min(1.0, float(gain)))
        self._notify()

    def get_master_gain(self) -> float:
        with self._lock:
            return self._master_gain

    # ─── Partial ceiling (scene mask over 1..N_BANDS) ─────────────────

    @staticmethod
    def level_to_partial_ceiling(level: float) -> int:
        """Map a continuous 0..1 level to integer ceiling n_max in 1..N_BANDS.

        Wire formula: ``n_max = 1 + round(level * 31)`` with level clamped to
        0..1.  OSC ``/digital/ceiling`` and REST global ``ceiling`` both use
        this mapping.
        """
        level = max(0.0, min(1.0, float(level)))
        return int(1 + round(level * (config.N_BANDS - 1)))

    def set_partial_ceiling(self, n_max: int) -> None:
        """Set the highest playable harmonic index (1..N_BANDS)."""
        with self._lock:
            self._partial_ceiling = max(1, min(config.N_BANDS, int(n_max)))
        self._notify()

    def set_partial_ceiling_from_level(self, level: float) -> None:
        """Set ceiling from a 0..1 level (OSC/REST continuous control)."""
        self.set_partial_ceiling(self.level_to_partial_ceiling(level))

    def get_partial_ceiling(self) -> int:
        with self._lock:
            return self._partial_ceiling

    # ─── Musical clock + settle (local-beat timebase) ─────────────────

    def set_clock_bpm(self, bpm: float) -> None:
        """Set musical clock tempo in BPM (clamped 20..240). Default 90."""
        with self._lock:
            self._clock_bpm = max(20.0, min(240.0, float(bpm)))
        self._notify()

    def get_clock_bpm(self) -> float:
        with self._lock:
            return self._clock_bpm

    def set_settle_beats(self, beats: float) -> None:
        """Set ease time constant in local beats (clamped 0.25..4.0). Default 1.0."""
        with self._lock:
            self._settle_beats = max(0.25, min(4.0, float(beats)))
        self._notify()

    def get_settle_beats(self) -> float:
        with self._lock:
            return self._settle_beats

    def set_generator_enable(self, enabled) -> None:
        """Enable/disable generators (arpegio/perc). Accepts bool or int 0|1."""
        with self._lock:
            if isinstance(enabled, bool):
                self._generator_enable = enabled
            else:
                self._generator_enable = bool(int(enabled))
        self._notify()

    def get_generator_enable(self) -> bool:
        with self._lock:
            return self._generator_enable

    # ─── Arpeggiator (H hands; H=0 and H=1) ───────────────────────────

    def _ensure_arp(self, hand: int) -> dict:
        h = int(hand)
        if h not in self._arp_state:
            self._arp_state[h] = _default_arp_state(h)
        return self._arp_state[h]

    def get_arp_state(self, hand: int = 0) -> dict:
        """Copy of arp runtime state for tests / diagnostics (not audio hot path)."""
        with self._lock:
            st = self._ensure_arp(hand)
            out = dict(st)
            out["pending_offs"] = list(st["pending_offs"])
            return out

    def set_arp_enable(self, hand: int, enabled) -> None:
        """Enable/disable arpeggiator hand H. Accepts bool or int 0|1."""
        with self._lock:
            st = self._ensure_arp(hand)
            if isinstance(enabled, bool):
                st["enabled"] = enabled
            else:
                st["enabled"] = bool(int(enabled))
            if not st["enabled"]:
                self._arp_release_sustain(st)
        self._notify()

    def set_arp_rate(self, hand: int, steps_per_beat: float) -> None:
        """Steps per beat 0..8. 0 ≈ sustain single note at window center."""
        with self._lock:
            st = self._ensure_arp(hand)
            st["target_rate"] = max(0.0, min(8.0, float(steps_per_beat)))
        self._notify()

    def set_arp_direction(self, hand: int, direction: float) -> None:
        """Direction -1..1 (down/up). Near 0 keeps last non-zero sign for stepping."""
        with self._lock:
            st = self._ensure_arp(hand)
            st["target_dir"] = max(-1.0, min(1.0, float(direction)))
        self._notify()

    def set_arp_density(self, hand: int, fill: float) -> None:
        """Fill 0..1 → quantized subdivision {1,2,3,4,6,8}."""
        with self._lock:
            st = self._ensure_arp(hand)
            st["target_density"] = max(0.0, min(1.0, float(fill)))
        self._notify()

    def set_arp_register_lo(self, hand: int, n: int) -> None:
        """Lower edge of the partial window (1..32)."""
        with self._lock:
            st = self._ensure_arp(hand)
            st["target_lo"] = float(max(1, min(config.N_BANDS, int(n))))
        self._notify()

    def set_arp_register_hi(self, hand: int, n: int) -> None:
        """Upper edge of the partial window (1..32), later clamped by ceiling."""
        with self._lock:
            st = self._ensure_arp(hand)
            st["target_hi"] = float(max(1, min(config.N_BANDS, int(n))))
        self._notify()

    def set_arp_gate(self, hand: int, frac: float) -> None:
        """Gate fraction 0..1 of step period (staccato↔legato)."""
        with self._lock:
            st = self._ensure_arp(hand)
            st["target_gate"] = max(0.0, min(1.0, float(frac)))
        self._notify()

    def set_arp_gain(self, hand: int, gain: float) -> None:
        """Arpeggiator hand gain 0..1."""
        with self._lock:
            st = self._ensure_arp(hand)
            st["target_gain"] = max(0.0, min(1.0, float(gain)))
        self._notify()

    def _arp_release_sustain(self, st: dict) -> None:
        """Internal: release sustain voice if held (caller holds lock)."""
        vid = st.get("sustain_voice_id")
        if vid is not None:
            # Inline voice_off without re-acquiring / notify spam mid-callback.
            for n, v in self._voices.items():
                if v.voice_id == vid:
                    if v.envelope_profile in ENVELOPE_PROFILES:
                        v.apply_envelope_profile(v.envelope_profile)
                    v.active = False
                    if n in self._active_history:
                        self._active_history.remove(n)
                    break
            st["sustain_voice_id"] = None
            st["sustain_n"] = None

    def _arp_voice_on(
        self,
        hand: int,
        harmonic_n: int,
        gain: float,
        envelope_profile: str = "pluck",
    ) -> int:
        """Internal voice_on for arp band (caller holds lock). Returns voice_id."""
        n = int(harmonic_n)
        vid = arp_voice_id(hand, n)
        freq = self.f1 * n
        self._ensure(n)
        v = self._voices[n]
        v.voice_id = vid
        v.freq = freq
        v.active = True
        v.gain = max(0.0, min(1.0, float(gain)))
        v.apply_envelope_profile(envelope_profile)
        if n in self._active_history:
            self._active_history.remove(n)
        self._active_history.append(n)
        while len(self._active_history) > config.MAX_VOICES:
            oldest_n = self._active_history.pop(0)
            self._voices[oldest_n].active = False
        return vid

    def _arp_voice_off(self, voice_id: int) -> None:
        """Internal voice_off (caller holds lock)."""
        for n, v in self._voices.items():
            if v.voice_id == voice_id:
                if v.envelope_profile in ENVELOPE_PROFILES:
                    v.apply_envelope_profile(v.envelope_profile)
                v.active = False
                if n in self._active_history:
                    self._active_history.remove(n)
                break

    def _arp_effective_window(self, st: dict) -> tuple[int, int]:
        """Effective partial window [lo, hi] ∩ [1, ceiling]."""
        ceiling = self._partial_ceiling
        lo = int(round(st["settled_lo"]))
        hi = int(round(st["settled_hi"]))
        lo = max(1, min(ceiling, lo))
        hi = max(1, min(ceiling, hi))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    @staticmethod
    def _arp_dir_sign(direction: float, last_sign: float) -> float:
        if direction > _ARP_DIR_EPS:
            return 1.0
        if direction < -_ARP_DIR_EPS:
            return -1.0
        return 1.0 if last_sign >= 0 else -1.0

    def advance_arp(self, dt: float) -> None:
        """Advance all enabled arpeggiators by ``dt`` seconds (audio callback).

        Eases target params with ``eased_target`` over ``settle_beats``, then
        either sustains the window-center partial (rate≈0) or steps the cursor
        at ``rate * density_subdiv`` steps per beat.
        """
        dt = float(dt)
        if dt <= 0.0:
            return
        with self._lock:
            if not self._generator_enable:
                # Still tick scheduled offs so notes don't hang forever.
                for st in self._arp_state.values():
                    self._arp_tick_pending_offs(st, dt)
                return

            bpm = self._clock_bpm
            settle = self._settle_beats
            delta_beats = (bpm / 60.0) * dt

            for hand, st in list(self._arp_state.items()):
                self._arp_ease_params(st, delta_beats, settle)
                self._arp_tick_pending_offs(st, dt)
                if not st["enabled"]:
                    continue
                self._arp_run_hand(hand, st, dt, bpm)

    def _arp_ease_params(self, st: dict, delta_beats: float, settle: float) -> None:
        for key in (
            "rate", "dir", "density", "lo", "hi", "gate", "gain",
        ):
            tkey = f"target_{key}"
            skey = f"settled_{key}"
            st[skey] = self.eased_target(st[skey], st[tkey], delta_beats, settle)

    def _arp_tick_pending_offs(self, st: dict, dt: float) -> None:
        remaining: list[tuple[float, int]] = []
        for t_left, vid in st["pending_offs"]:
            t_left -= dt
            if t_left <= 0.0:
                self._arp_voice_off(vid)
            else:
                remaining.append((t_left, vid))
        st["pending_offs"] = remaining

    def _arp_run_hand(self, hand: int, st: dict, dt: float, bpm: float) -> None:
        lo, hi = self._arp_effective_window(st)
        window_center = int(round((lo + hi) / 2.0))
        rate = st["settled_rate"]
        direction = st["settled_dir"]
        density = st["settled_density"]
        gate = max(0.0, min(1.0, st["settled_gate"]))
        gain = st["settled_gain"]

        new_sign = self._arp_dir_sign(direction, st["last_dir_sign"])
        prev_sign = st["last_dir_sign"]
        # Direction zero-cross: non-zero → opposite non-zero triggers immediately.
        dir_crossed = (
            abs(direction) > _ARP_DIR_EPS
            and prev_sign != 0.0
            and new_sign != 0.0
            and (prev_sign * new_sign) < 0.0
        )
        if abs(direction) > _ARP_DIR_EPS:
            st["last_dir_sign"] = new_sign

        # rate≈0 → sustain single note at window center
        if abs(rate) < _ARP_RATE_EPS:
            st["step_phase"] = 0.0
            st["cursor_n"] = window_center
            if st["sustain_n"] != window_center or st["sustain_voice_id"] is None:
                self._arp_release_sustain(st)
                vid = self._arp_voice_on(hand, window_center, gain, "pluck")
                st["sustain_voice_id"] = vid
                st["sustain_n"] = window_center
            else:
                # Keep gain/freq current while held.
                n = window_center
                if n in self._voices and self._voices[n].voice_id == st["sustain_voice_id"]:
                    self._voices[n].gain = max(0.0, min(1.0, float(gain)))
                    self._voices[n].freq = self.f1 * n
            if dir_crossed:
                # Zero-cross while sustained: re-trigger same center (new attack).
                vid = self._arp_voice_on(hand, window_center, gain, "pluck")
                st["sustain_voice_id"] = vid
                st["sustain_n"] = window_center
            return

        # rate > 0: leave sustain mode
        if st["sustain_voice_id"] is not None:
            self._arp_release_sustain(st)

        subdiv = density_to_subdiv(density)
        steps_per_beat = max(0.0, rate) * subdiv
        step_rate_hz = steps_per_beat * (bpm / 60.0)
        if step_rate_hz <= 0.0:
            return
        step_period = 1.0 / step_rate_hz

        # Clamp cursor into current window before stepping.
        cursor = int(st["cursor_n"])
        if cursor < lo or cursor > hi:
            cursor = max(lo, min(hi, cursor))
            st["cursor_n"] = cursor

        if dir_crossed:
            cursor = self._arp_step_cursor(cursor, new_sign, lo, hi)
            st["cursor_n"] = cursor
            self._arp_trigger_step(hand, st, cursor, gain, gate, step_period)

        st["step_phase"] += step_rate_hz * dt
        # Guard against huge dt (catch-up) flooding the polyphony budget.
        max_steps = 32
        steps = 0
        while st["step_phase"] >= 1.0 and steps < max_steps:
            st["step_phase"] -= 1.0
            steps += 1
            sign = self._arp_dir_sign(st["settled_dir"], st["last_dir_sign"])
            cursor = self._arp_step_cursor(int(st["cursor_n"]), sign, lo, hi)
            st["cursor_n"] = cursor
            self._arp_trigger_step(hand, st, cursor, gain, gate, step_period)

    @staticmethod
    def _arp_step_cursor(cursor: int, sign: float, lo: int, hi: int) -> int:
        step = 1 if sign >= 0 else -1
        n = cursor + step
        if n > hi:
            n = lo
        elif n < lo:
            n = hi
        return n

    def _arp_cancel_pending_for(self, st: dict, voice_id: int) -> None:
        """Drop scheduled offs for ``voice_id`` so a re-trigger is not killed early."""
        st["pending_offs"] = [
            (t, vid) for (t, vid) in st["pending_offs"] if vid != voice_id
        ]

    def _arp_trigger_step(
        self,
        hand: int,
        st: dict,
        cursor: int,
        gain: float,
        gate: float,
        step_period: float,
    ) -> None:
        vid = arp_voice_id(hand, cursor)
        # Cancel any prior gate for this band id before re-triggering.
        self._arp_cancel_pending_for(st, vid)
        vid = self._arp_voice_on(hand, cursor, gain, "pluck")
        # Schedule voice_off after gate * step_period (legato if gate≈1).
        off_after = max(0.0, float(gate) * float(step_period))
        if off_after <= 0.0:
            # Immediate off still allows a one-block click via attack; release now.
            self._arp_voice_off(vid)
        else:
            st["pending_offs"].append((off_after, vid))

    # ─── Foot percussion (dedicated pool, separate melodic norm) ─────

    def set_perc_enable(self, enabled) -> None:
        """Enable/disable foot percussion. Accepts bool or int 0|1."""
        with self._lock:
            if isinstance(enabled, bool):
                self._perc_enabled = enabled
            else:
                self._perc_enabled = bool(int(enabled))
            self._perc_state["enabled"] = self._perc_enabled
            if not self._perc_enabled:
                # Stop scheduling new hits; pending offs still tick in advance_perc.
                self._perc_state["step_phase"] = 0.0
        self._notify()

    def set_perc_rate(self, pulses_per_beat: float) -> None:
        """Pulses per musical beat 0..8. 0 = silent (no steps)."""
        with self._lock:
            self._perc_rate = max(0.0, min(8.0, float(pulses_per_beat)))
            self._perc_state["rate"] = self._perc_rate
        self._notify()

    def set_perc_gain(self, gain: float) -> None:
        """Percussion gain 0..1."""
        with self._lock:
            self._perc_gain = max(0.0, min(1.0, float(gain)))
            self._perc_state["gain"] = self._perc_gain
        self._notify()

    def set_perc_accent(self, accent: float) -> None:
        """Downbeat accent amount 0..1 (extra gain on pulse_index % rate ≈ 0)."""
        with self._lock:
            self._perc_accent = max(0.0, min(1.0, float(accent)))
            self._perc_state["accent"] = self._perc_accent
        self._notify()

    def get_perc_state(self) -> dict:
        """Copy of percussion runtime state for tests / diagnostics."""
        with self._lock:
            out = {
                "enabled": self._perc_enabled,
                "rate": self._perc_rate,
                "gain": self._perc_gain,
                "accent": self._perc_accent,
                "step_phase": self._perc_state["step_phase"],
                "pulse_index": self._perc_state["pulse_index"],
                "next_slot": self._perc_state["next_slot"],
                "pending_offs": list(self._perc_state["pending_offs"]),
                "active_slots": list(self._perc_state["active_slots"]),
            }
            return out

    def _perc_slot_key(self, slot: int) -> int:
        """Dict key for perc pool slot — same as voice_id (-30000..-30007)."""
        return perc_voice_id(slot)

    def _perc_voice_on(self, slot: int, gain: float, partial_n: int) -> int:
        """Internal voice_on for perc pool (caller holds lock). Returns voice_id.

        Stored under a dedicated key outside 1..32 so hits never steal melodic
        harmonic slots. ``harmonic_n`` carries the partial used for f1*n.
        """
        vid = perc_voice_id(slot)
        key = self._perc_slot_key(slot)
        n = max(1, min(config.N_BANDS, int(partial_n)))
        freq = self.f1 * n
        if key not in self._voices:
            self._voices[key] = VoiceParams(harmonic_n=n)
        v = self._voices[key]
        v.harmonic_n = n
        v.voice_id = vid
        v.freq = freq
        v.active = True
        v.gain = max(0.0, min(1.0, float(gain)))
        v.apply_envelope_profile("perc")
        # Do NOT touch melodic _active_history — perc has its own rotating pool.
        return vid

    def _perc_voice_off(self, voice_id: int) -> None:
        """Internal voice_off for a perc voice_id (caller holds lock)."""
        for key, v in self._voices.items():
            if v.voice_id == voice_id and is_perc_voice_id(voice_id):
                if v.envelope_profile in ENVELOPE_PROFILES:
                    v.apply_envelope_profile(v.envelope_profile)
                v.active = False
                break
        # Drop from active_slots diagnostics
        st = self._perc_state
        st["active_slots"] = [vid for vid in st["active_slots"] if vid != voice_id]

    def _perc_tick_pending_offs(self, dt: float) -> None:
        st = self._perc_state
        remaining: list[tuple[float, int]] = []
        for t_left, vid in st["pending_offs"]:
            t_left -= dt
            if t_left <= 0.0:
                self._perc_voice_off(vid)
            else:
                remaining.append((t_left, vid))
        st["pending_offs"] = remaining

    def _perc_trigger_hit(self) -> None:
        """Fire one percussion hit on the oldest rotating pool slot (caller holds lock)."""
        st = self._perc_state
        rate = self._perc_rate
        base_gain = self._perc_gain
        accent = self._perc_accent
        pulse_index = int(st["pulse_index"])

        # Downbeat accent: first pulse of each beat group (when rate ≥ 1).
        # pulse_index counts hits; every ``max(1, round(rate))`` pulses is a downbeat.
        steps_per_beat = max(1, int(round(rate))) if rate >= 0.5 else 1
        is_downbeat = (pulse_index % steps_per_beat) == 0
        gain = base_gain * (1.0 + (accent if is_downbeat else 0.0))
        gain = max(0.0, min(1.0, gain))

        slot = int(st["next_slot"]) % _PERC_POOL_SIZE
        partial = _PERC_PARTIALS[slot % len(_PERC_PARTIALS)]
        vid = self._perc_voice_on(slot, gain, partial)

        # Cancel any prior pending off for this slot before re-using it.
        st["pending_offs"] = [
            (t, v) for (t, v) in st["pending_offs"] if v != vid
        ]
        # Short perc envelope: schedule voice_off after release (~80 ms).
        st["pending_offs"].append((_PERC_RELEASE_S, vid))

        if vid in st["active_slots"]:
            st["active_slots"].remove(vid)
        st["active_slots"].append(vid)

        st["next_slot"] = (slot + 1) % _PERC_POOL_SIZE
        st["pulse_index"] = pulse_index + 1

    def advance_perc(self, dt: float) -> None:
        """Advance foot percussion by ``dt`` seconds (audio callback).

        Separate from the arpeggiator. Steps at ``perc_rate * bpm/60`` Hz.
        Gated by ``generator_enable`` and ``perc_enable``. Perc voices live in
        a dedicated pool (voice_id -30000..-30007) and must be excluded from
        the melodic 1/√N normalisation in the audio engine.
        """
        dt = float(dt)
        if dt <= 0.0:
            return
        with self._lock:
            # Always tick pending offs so notes don't hang after disable/panic.
            self._perc_tick_pending_offs(dt)

            if not self._generator_enable or not self._perc_enabled:
                return

            rate = self._perc_rate
            if rate < _ARP_RATE_EPS:
                self._perc_state["step_phase"] = 0.0
                return

            bpm = self._clock_bpm
            pulse_rate_hz = rate * (bpm / 60.0)
            if pulse_rate_hz <= 0.0:
                return

            st = self._perc_state
            st["step_phase"] += pulse_rate_hz * dt
            max_steps = 32
            steps = 0
            while st["step_phase"] >= 1.0 and steps < max_steps:
                st["step_phase"] -= 1.0
                steps += 1
                self._perc_trigger_hit()

    def _clear_arp_runtime(self) -> None:
        """Reset all arp hands to default runtime (caller holds lock)."""
        for hand in list(self._arp_state.keys()):
            self._arp_state[hand] = _default_arp_state(hand)
        # Ensure both H=0 and H=1 always exist after panic.
        for h in _ARP_HANDS:
            if h not in self._arp_state:
                self._arp_state[h] = _default_arp_state(h)

    def _clear_perc_runtime(self) -> None:
        """Reset percussion pool runtime (caller holds lock). Scene params kept."""
        # Deactivate any perc-pool voices still in the store.
        for key in list(self._voices.keys()):
            v = self._voices[key]
            if is_perc_voice_id(v.voice_id) or (
                key <= _PERC_VOICE_ID_BASE
                and key >= _PERC_VOICE_ID_BASE - (_PERC_POOL_SIZE - 1)
            ):
                v.active = False
                v.voice_id = None
        # Preserve enable/rate/gain/accent scene params; clear runtime only.
        enabled = self._perc_enabled
        rate = self._perc_rate
        gain = self._perc_gain
        accent = self._perc_accent
        self._perc_state = _default_perc_state()
        self._perc_state["enabled"] = enabled
        self._perc_state["rate"] = rate
        self._perc_state["gain"] = gain
        self._perc_state["accent"] = accent

    def get_beat_phase(self) -> float:
        """Current beat phase in 0..1 (fraction of one beat)."""
        with self._lock:
            return self._beat_phase

    def advance_beat(self, dt: float) -> float:
        """Advance beat phase by ``dt`` seconds. Returns phase after advance (0..1).

        At ``clock_bpm`` BPM, phase rate is ``bpm/60`` cycles per second, so
        120 BPM yields two full phase cycles per second.
        """
        with self._lock:
            rate = self._clock_bpm / 60.0
            self._beat_phase = (self._beat_phase + rate * float(dt)) % 1.0
            return self._beat_phase

    @staticmethod
    def eased_target(
        param_current: float,
        param_target: float,
        delta_beats: float,
        settle_beats: float,
    ) -> float:
        """Exponential ease of ``param_current`` toward ``param_target`` over beats.

        ``param_current + (target - current) * (1 - exp(-delta_beats / settle_beats))``.

        With settle_beats=1.0 and delta_beats=1.0 the result is ~63.2% of the
        way from current to target (one time constant). Network-robust: time is
        measured in local musical beats, not wall-clock milliseconds.
        """
        if settle_beats <= 0.0:
            return float(param_target)
        if delta_beats <= 0.0:
            return float(param_current)
        alpha = 1.0 - math.exp(-float(delta_beats) / float(settle_beats))
        return float(param_current) + (float(param_target) - float(param_current)) * alpha

    def set_global_attack(self, attack_s: float) -> None:
        with self._lock:
            self._global_attack_s = max(0.0, min(5.0, float(attack_s)))
        self._notify()

    def set_global_release(self, release_s: float) -> None:
        with self._lock:
            self._global_release_s = max(0.0, min(5.0, float(release_s)))
        self._notify()

    # ─── Timbre ──────────────────────────────────────────────────────────

    def set_shape(self, harmonic_n: int, shape: float) -> None:
        with self._lock:
            self._ensure(harmonic_n)
            self._voices[harmonic_n].shape = max(0.0, min(1.0, float(shape)))
        self._notify()

    # ─── Side-chain ──────────────────────────────────────────────────────

    def set_beacon_level(self, level: float) -> None:
        with self._lock:
            self._beacon_level = max(0.0, min(1.0, float(level)))

    def get_beacon_level(self) -> float:
        with self._lock:
            return self._beacon_level

    def set_sidechain_amount(self, amount: float) -> None:
        with self._lock:
            self._sidechain_amount = max(-1.0, min(1.0, float(amount)))
        self._notify()

    def get_sidechain_amount(self) -> float:
        with self._lock:
            return self._sidechain_amount

    # ─── LFO ─────────────────────────────────────────────────────────────

    def set_lfo_rate_divisor(self, divisor: int) -> None:
        with self._lock:
            self._lfo_rate_divisor = max(1, int(divisor))
        self._notify()

    def set_lfo_waveform(self, waveform: str) -> None:
        with self._lock:
            if waveform in ("sine", "triangle", "saw", "square", "samplehold"):
                self._lfo_waveform = waveform
        self._notify()

    def set_lfo_amount(self, amount: float) -> None:
        with self._lock:
            self._lfo_amount = max(0.0, min(1.0, float(amount)))
        self._notify()

    def get_lfo_amount(self) -> float:
        with self._lock:
            return self._lfo_amount

    def set_lfo_gain(self, harmonic_n: int, amount: float) -> None:
        with self._lock:
            self._ensure(harmonic_n)
            self._voices[harmonic_n].lfo_gain = max(0.0, min(1.0, float(amount)))
        self._notify()

    def set_lfo_pan(self, harmonic_n: int, amount: float) -> None:
        with self._lock:
            self._ensure(harmonic_n)
            self._voices[harmonic_n].lfo_pan = max(0.0, min(1.0, float(amount)))
        self._notify()

    def set_lfo_phase(self, harmonic_n: int, amount: float) -> None:
        with self._lock:
            self._ensure(harmonic_n)
            self._voices[harmonic_n].lfo_phase = max(0.0, min(1.0, float(amount)))
        self._notify()

    def advance_lfo(self, dt: float) -> float:
        """Advance LFO phase by dt seconds. Returns current LFO value -1..+1.

        Also advances the musical beat phase (``_beat_phase``) so the clock
        keeps running on the same audio-callback timebase as the LFO.
        """
        with self._lock:
            # Musical clock: bpm/60 cycles per second.
            rate = self._clock_bpm / 60.0
            self._beat_phase = (self._beat_phase + rate * float(dt)) % 1.0
            period = self._strum_period_s * self._lfo_rate_divisor
            if period <= 0:
                period = 0.5
            self._lfo_phase = (self._lfo_phase + dt / period) % 1.0
            return self._lfo_value()

    def _lfo_value(self) -> float:
        """Compute LFO value -1..+1 from current phase and waveform."""
        p = self._lfo_phase
        wf = self._lfo_waveform
        if wf == "triangle":
            return 1.0 - 4.0 * abs(p - 0.5)
        elif wf == "saw":
            return 1.0 - 2.0 * p
        elif wf == "square":
            return 1.0 if p < 0.5 else -1.0
        elif wf == "samplehold":
            # Return last value; new random each cycle
            if not hasattr(self, '_lfo_sh_value'):
                self._lfo_sh_value = 0.0
            if p < 0.01:  # new cycle
                import random
                self._lfo_sh_value = random.uniform(-1.0, 1.0)
            return self._lfo_sh_value
        else:  # sine
            import math
            return math.sin(2.0 * math.pi * p)

    def record_strum(self, timestamp_s: float) -> None:
        """Record a strum event to estimate period."""
        with self._lock:
            self._strum_times.append(timestamp_s)
            if len(self._strum_times) > config.STRUM_WINDOW:
                self._strum_times.pop(0)
            if len(self._strum_times) >= 2:
                intervals = [self._strum_times[i] - self._strum_times[i-1]
                             for i in range(1, len(self._strum_times))]
                self._strum_period_s = sum(intervals) / len(intervals)

    def panic(self) -> None:
        """Clear all voice state. Does not reset scene params (e.g. ceiling).

        Also resets arpeggiator runtime (both hands) and the percussion pool
        so generators do not keep stepping or holding pending offs after panic.
        """
        with self._lock:
            for v in self._voices.values():
                v.active = False
                v.gain = config.DEFAULT_VOICE_GAIN
                v.pan = 0.0
                v.phase = 0.0
            self._active_history.clear()
            self._clear_arp_runtime()
            self._clear_perc_runtime()
        self._notify()
        # Also notify the MIDI control (Launchpad) to clear lights + state
        if self._panic_callback:
            try:
                self._panic_callback()
            except Exception:
                pass

    # ─── Snapshot accessors ───────────────────────────────────────────────

    def get_snapshot(self) -> dict[int, VoiceParams]:
        """Active voices only — for the audio callback (hot path)."""
        with self._lock:
            return {k: v.copy() for k, v in self._voices.items()
                    if v.active and v.freq > 0}

    def get_all_snapshot(self) -> dict[int, VoiceParams]:
        with self._lock:
            return {k: v.copy() for k, v in self._voices.items()}

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "f1": self.f1,
                "master_gain": self._master_gain,
                "global_attack_s": self._global_attack_s,
                "global_release_s": self._global_release_s,
                "sidechain_amount": self._sidechain_amount,
                "beacon_level": self._beacon_level,
                "lfo_rate_divisor": self._lfo_rate_divisor,
                "lfo_waveform": self._lfo_waveform,
                "lfo_amount": self._lfo_amount,
                "strum_period_s": round(self._strum_period_s, 3),
                "partial_ceiling": self._partial_ceiling,
                "clock_bpm": self._clock_bpm,
                "settle_beats": self._settle_beats,
                "generator_enable": self._generator_enable,
                "arp": {
                    str(h): {
                        "enabled": st["enabled"],
                        "cursor_n": st["cursor_n"],
                        "step_phase": round(st["step_phase"], 4),
                        "rate": round(st["settled_rate"], 4),
                        "direction": round(st["settled_dir"], 4),
                        "density": round(st["settled_density"], 4),
                        "register_lo": int(round(st["settled_lo"])),
                        "register_hi": int(round(st["settled_hi"])),
                        "gate": round(st["settled_gate"], 4),
                        "gain": round(st["settled_gain"], 4),
                    }
                    for h, st in sorted(self._arp_state.items())
                },
                "perc": {
                    "enabled": self._perc_enabled,
                    "rate": round(self._perc_rate, 4),
                    "gain": round(self._perc_gain, 4),
                    "accent": round(self._perc_accent, 4),
                    "step_phase": round(self._perc_state["step_phase"], 4),
                    "pulse_index": int(self._perc_state["pulse_index"]),
                    "next_slot": int(self._perc_state["next_slot"]),
                },
                "voices": {
                    str(k): {
                        "gain": v.gain,
                        "pan": v.pan,
                        "phase_deg": round(math.degrees(v.phase) % 360, 1),
                        "attack_s": v.attack_s,
                        "release_s": v.release_s,
                        "shape": v.shape,
                        "lfo_gain": v.lfo_gain,
                        "lfo_pan": v.lfo_pan,
                        "lfo_phase": v.lfo_phase,
                        "active": v.active,
                        "freq": v.freq,
                        "envelope_profile": v.envelope_profile,
                    }
                    for k, v in sorted(self._voices.items())
                },
            }


def advance_arp(dt: float, store: "VoiceParameterStore") -> None:
    """Module-level entry for the audio callback / tests: advance_arp(dt, store)."""
    store.advance_arp(dt)


def advance_perc(dt: float, store: "VoiceParameterStore") -> None:
    """Module-level entry for the audio callback / tests: advance_perc(dt, store)."""
    store.advance_perc(dt)
