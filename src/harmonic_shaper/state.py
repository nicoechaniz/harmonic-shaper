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

    def copy(self) -> "VoiceParams":
        return copy(self)

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

    def voice_on(self, harmonic_n: int, voice_id: int, freq: float, gain: float = None) -> None:
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

            if harmonic_n in self._active_history:
                self._active_history.remove(harmonic_n)
            self._active_history.append(harmonic_n)

            # Note stealing: drop oldest if over limit
            while len(self._active_history) > config.MAX_VOICES:
                oldest_n = self._active_history.pop(0)
                self._voices[oldest_n].active = False
        self._notify()

    def voice_off(self, voice_id: int) -> None:
        with self._lock:
            for n, v in self._voices.items():
                if v.voice_id == voice_id:
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
        """Advance LFO phase by dt seconds. Returns current LFO value -1..+1."""
        with self._lock:
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
        with self._lock:
            for v in self._voices.values():
                v.active = False
                v.gain = config.DEFAULT_VOICE_GAIN
                v.pan = 0.0
                v.phase = 0.0
            self._active_history.clear()
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
                    }
                    for k, v in sorted(self._voices.items())
                },
            }
