"""Real-time additive synthesis engine for the Shaper.

Adapted from NaturalHarmony/harmonic_shaper/audio_engine.py:
- Same architecture: numpy + sounddevice PortAudio callback.
- Bumped MAX_VOICES to config.MAX_VOICES (32).
- Equal-power stereo pan preserved.
- Phase accumulator continuity preserved across callbacks.
- The audio callback runs in a C thread, so the snapshot read must be
  brief and lock-free from its perspective — we rely on dict copy being
  fast for ≤32 entries.
"""

import logging
import threading
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
    SOUNDDEVICE_IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    HAS_SOUNDDEVICE = False
    SOUNDDEVICE_IMPORT_ERROR = exc
    sd = None  # type: ignore

from .state import VoiceParameterStore
from . import config
from .audio_levels import soft_limit

log = logging.getLogger(__name__)


class AudioEngine:
    """Stereo additive synthesis — one pure sine per active voice."""

    def __init__(
        self,
        store: VoiceParameterStore,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        block_size: int = config.AUDIO_BLOCK_SIZE,
        device: Optional[int | str] = config.AUDIO_DEVICE,
    ):
        self._store = store
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._device = device
        self._stream: Optional["sd.OutputStream"] = None
        # Per-voice state: {harmonic_n: {"phase": float, "env": float, "params": VoiceParams}}
        # env ramps 0→1 on attack, 1→0 on release. Voices with env≈0 and inactive are pruned.
        self._voice_state: dict[int, dict] = {}
        self._running = False
        self._lock = threading.RLock()
        self._record_sink: Optional[list] = None

    def _resolve_stream_params(self) -> tuple[int, Optional[int | str]]:
        """Resolve the effective (sample_rate, device) for the stream.

        JACK (the reliable path over PipeWire on this host — the PipeWire
        ALSA plugin renders silence for PortAudio streams) imposes the
        server sample rate; opening with any other rate fails with
        PaErrorCode -9997. Adopt the device's rate when JACK is selected.
        """
        if not HAS_SOUNDDEVICE:
            return self._sample_rate, self._device
        try:
            info = sd.query_devices(self._device) if self._device is not None else None
            if info is None:
                return self._sample_rate, self._device
            hostapi = sd.query_hostapis(info["hostapi"])["name"]
            if "JACK" in hostapi:
                return int(info["default_samplerate"]), self._device
        except Exception as exc:
            log.debug("device probe failed, keeping configured params: %s", exc)
        return self._sample_rate, self._device

    def start(self) -> None:
        if self._running:
            return
        if not HAS_SOUNDDEVICE:
            detail = str(SOUNDDEVICE_IMPORT_ERROR or "not installed")
            raise ImportError(f"sounddevice/PortAudio is required: {detail}")
        sample_rate, device = self._resolve_stream_params()
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            blocksize=self._block_size,
            channels=2,
            dtype="float32",
            device=device,
            callback=self._audio_callback,
            finished_callback=self._on_stream_finished,
        )
        self._stream.start()
        self._running = True
        self._sample_rate = sample_rate
        self._device = device
        log.info("Shaper audio: sr=%d block=%d device=%s",
                 self._sample_rate, self._block_size, self._device)

    def stop(self) -> None:
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                log.warning("Error closing stream: %s", exc)
            self._stream = None
        log.info("Shaper audio stopped.")

    # ─── Recording tap ──────────────────────────────────────────────────
    # The Recorder calls attach_recorder(list) on start and detach_recorder
    # on stop. The audio callback appends the final mix (post-sidechain,
    # pre-limiter) to the list — copy of the buffer so it survives the
    # callback returning. Zero overhead when no recorder attached.

    def attach_recorder(self, sink: list) -> None:
        """Tap the final mix into `sink` (a Python list of ndarrays)."""
        with self._lock:
            self._record_sink = sink

    def detach_recorder(self) -> None:
        """Stop tapping. The sink list stays (Recorder owns it)."""
        with self._lock:
            self._record_sink = None

    @property
    def is_running(self) -> bool:
        return bool(self._running and self._stream and self._stream.active)

    @staticmethod
    def list_devices() -> str:
        if HAS_SOUNDDEVICE:
            return str(sd.query_devices())
        return "(sounddevice not installed)"

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.debug("Audio status: %s", status)
        dt = frames / self._sample_rate

        # ── LFO + beat clock, then Shaper-owned generators ────────────
        lfo_val = self._store.advance_lfo(dt)  # -1..+1; also advances beat phase
        lfo_amount = self._store.get_lfo_amount()  # global 0..1
        # Arpeggiator reads settled params and emits voice_on/off on its band.
        self._store.advance_arp(dt)

        voices = self._store.get_snapshot()  # active voices only (post-arp)
        active_ns = set(voices.keys())
        tracked_ns = set(self._voice_state.keys())

        # ── Add new voices (just became active) ──────────────────────
        for n in active_ns - tracked_ns:
            self._voice_state[n] = {"phase": 0.0, "env": 0.0, "params": voices[n]}

        # ── Mark released voices (was tracked, no longer active) ────
        for n in tracked_ns - active_ns:
            self._voice_state[n]["params"].active = False

        # ── Update active voices' params ─────────────────────────────
        for n in active_ns & tracked_ns:
            self._voice_state[n]["params"] = voices[n]

        # Per-voice normalization. Count both currently-active voices AND
        # voices still in release tail (env > 0), because the latter still
        # contribute to the mix until env ramps down to zero. Without this,
        # the norm is too low during voice_on/off transitions and the mix
        # clips (was: n_active = len(active_ns), which undercounted).
        n_active = len(active_ns)
        for n, state in self._voice_state.items():
            if state["env"] > 0.001 and n not in active_ns:
                n_active += 1
        norm = 1.0 / (n_active ** 0.5) if n_active > 0 else 1.0

        # Scene mask: harmonics above the ceiling enter natural release
        # (target_env=0) without hard-cutting; rising ceiling re-enables them.
        partial_ceiling = self._store.get_partial_ceiling()

        mix = np.zeros((frames, 2), dtype=np.float32)

        to_prune = []
        for n, state in self._voice_state.items():
            params = state["params"]
            if params.freq <= 0:
                to_prune.append(n)
                continue

            target_env = 1.0 if params.active else 0.0
            # partial_ceiling mask over fixed grid 1..32 (not a hard cut).
            if params.harmonic_n > partial_ceiling:
                target_env = 0.0
            current_env = state["env"]
            attack_s = params.attack_s
            release_s = params.release_s

            # Compute envelope ramp
            if target_env > current_env:
                rate = 1.0 / max(attack_s, 0.0001)
                new_env = min(target_env, current_env + rate * dt)
            elif target_env < current_env:
                rate = 1.0 / max(release_s, 0.0001)
                new_env = max(target_env, current_env - rate * dt)
            else:
                new_env = current_env

            state["env"] = new_env

            if not params.active and new_env <= 0.0:
                to_prune.append(n)
                continue

            if new_env <= 0.0:
                continue

            # ── LFO modulation per voice ──────────────────────────────
            lfo_mod = lfo_val * lfo_amount
            mod_gain = params.gain * (1.0 + params.lfo_gain * lfo_mod)
            mod_pan = params.pan + params.lfo_pan * lfo_mod * 2.0  # ±2 range
            mod_pan = max(-1.0, min(1.0, mod_pan))
            mod_phase = params.phase + params.lfo_phase * lfo_mod * np.pi

            # ── Generate sine ─────────────────────────────────────────
            t = np.arange(frames, dtype=np.float64) / self._sample_rate
            start_phase = state["phase"]
            carrier_phases = 2.0 * np.pi * params.freq * t + start_phase
            sine = np.sin(carrier_phases + mod_phase).astype(np.float32)

            # ── Waveshaper (didgeridoo/vocal timbre) ──────────────────
            shape = params.shape
            if shape > 0.0:
                drive = 1.0 + shape * 4.0
                sine = np.tanh(sine * drive) / np.tanh(drive)

            sine *= float(mod_gain) * norm * new_env
            state["phase"] = (
                carrier_phases[-1] + 2.0 * np.pi * params.freq / self._sample_rate
            ) % (2.0 * np.pi)

            angle = (float(mod_pan) + 1.0) * (np.pi / 4.0)
            mix[:, 0] += sine * float(np.cos(angle))
            mix[:, 1] += sine * float(np.sin(angle))

        for n in to_prune:
            del self._voice_state[n]

        # ── Side-chain: beacon envelope → shaper master modulation ────
        beacon_level = self._store.get_beacon_level()
        sidechain_amount = self._store.get_sidechain_amount()
        if sidechain_amount >= 0:
            sc_factor = (1.0 - sidechain_amount) + sidechain_amount * beacon_level
        else:
            sc_factor = 1.0 + abs(sidechain_amount) * (1.0 - beacon_level)

        # Master gain + side-chain + soft limiter
        mix *= self._store.get_master_gain() * sc_factor

        # ── Recording tap: hand a copy of the final mix to the recorder
        #    BEFORE the soft limiter, so the Recorder can do its own
        #    controlled limiting on the full mix (SC + Shaper).
        sink = self._record_sink
        if sink is not None:
            sink.append(mix.copy())

        mix = soft_limit(mix)
        outdata[:] = mix

    def _on_stream_finished(self) -> None:
        log.warning("Audio stream finished unexpectedly.")
        self._running = False
