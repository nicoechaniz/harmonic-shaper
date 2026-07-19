"""Launchpad Mini, Minilab3, and native keyboard control for the Shaper.

Brought from NaturalHarmony/harmonic_beacon/main.py (the original Pad Mode
logic) and adapted to:
  - 32 bands (1..32) instead of 64 — the top half repeats the same range
    as the bottom half (per Nico's spec: F1..F32 on lower half, F1..F32 on
    upper half with toggle behavior).
  - Direct routing into the standalone Shaper voice store.
  - Polyphony limit + visual note stealing on the Launchpad (the new lights
    turn off when the limit is exceeded).
  - Native MIDI-note harmonic source for generic keyboards (no NaturalHarmony
    beacon required): see :class:`NativeNoteHandler` / :class:`NativeMidiNoteSource`.

Pad layout (Launchpad Mini, Programmer mode, stride 16):

    y=0 is the TOP row (physical A), y=7 is the BOTTOM row (physical H).
    row_from_bottom = 7 - y  (matches NaturalHarmony/harmonic_beacon).

    rows 0..3 from bottom (physical H, G, F, E): MOMENTARY
    n = 1 + x + row_from_bottom*8
    H1 = row_from_bottom 0, x=0 → F1 (40 Hz, bottom-left)
    H8 = row_from_bottom 0, x=7 → F8 (320 Hz, bottom-right)
    ...
    E8 = row_from_bottom 3, x=7 → F32 (1000 Hz)

    rows 4..7 from bottom (physical D, C, B, A): TOGGLE (latching)
    n = 1 + x + (row_from_bottom-4)*8
    D1 = row_from_bottom 4, x=0 → F1 toggle
    ...
    A8 = row_from_bottom 7, x=7 → F32 toggle (top-right)

X = column 0..7 left to right.

Stride autodetect:
  - First incoming pad event sets the stride (8 for Note mode, 16 for Programmer).
  - Stride is preserved for the lifetime of the session.

Lights:
  - On press of a momentary pad: pad lights up (color ON).
  - On release of a momentary pad: pad turns off.
  - On press of a toggle pad: pad lights up (color TOGGLE_ON) and stays on.
  - On press of a toggle pad that was already lit: pad turns off (toggle release).
  - On panic: all 64 pads off.

Polyphony:
  - MAX_VOICES (32 by default) total active voices.
  - When a new toggle-on would exceed the limit, the oldest toggled harmonic
    is auto-released (and its light turned off).

Native keyboard voice lifecycle:
  - note-on → map_midi_note(f1, anchor) → voice_on(band, voice_id, freq, gain)
  - note-off / note-on vel=0 → voice_off(originating voice_id only)
  - concurrent notes tracked per MIDI note number so releases never cross
"""

import logging
import threading
from collections import OrderedDict
from typing import Optional, Sequence, Set

try:
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False
    mido = None

from .harmonic_mapping import (
    map_midi_note,
    map_sequential_harmonic,
    velocity_to_gain,
)
from .state import VoiceParameterStore
from . import config

log = logging.getLogger(__name__)


# ─── Native MIDI-note harmonic source ───────────────────────────────────────


class NativeNoteHandler:
    """Port-independent MIDI note → Shaper voice lifecycle.

    Safe for headless probes: feed ``note_on`` / ``note_off`` without opening
    hardware ports.  Voice IDs are allocated in a high range so they do not
    collide with Launchpad local IDs (which start at 1).

    Policy
    ------
    - ``sequential_banks`` maps adjacent keys to integer partials and exact
      ``f1*n`` frequencies. One bank is momentary; another is toggle/sustain.
    - ``legacy_hybrid`` retains the prior NaturalHarmony-derived mapper.
    - Each active MIDI note owns exactly one ``voice_id`` and never releases a
      different note's voice.
    """

    # Keep well clear of Launchpad/Minilab local counters (start at 1).
    _VOICE_ID_BASE = 1_000_000

    def __init__(
        self,
        store: VoiceParameterStore,
        *,
        anchor_midi: int = config.DEFAULT_ANCHOR_MIDI,
        enabled: bool = config.NATIVE_MIDI_SOURCE_ENABLED,
        velocity_gain_min: float = config.NATIVE_MIDI_VELOCITY_GAIN_MIN,
        velocity_gain_max: float = config.NATIVE_MIDI_VELOCITY_GAIN_MAX,
        max_bands: int = config.NATIVE_MIDI_BANK_SIZE,
        mapping_mode: str = config.NATIVE_MIDI_MAPPING_MODE,
        momentary_start_midi: int = config.NATIVE_MIDI_MOMENTARY_START,
        toggle_start_midi: int = config.NATIVE_MIDI_TOGGLE_START,
        panic_midi_note: int | None = config.NATIVE_MIDI_PANIC_NOTE,
    ) -> None:
        self._store = store
        self._anchor_midi = int(anchor_midi)
        self._enabled = bool(enabled)
        self._vel_min = float(velocity_gain_min)
        self._vel_max = float(velocity_gain_max)
        self._max_bands = int(max_bands)
        if mapping_mode not in {"sequential_banks", "legacy_hybrid"}:
            raise ValueError(f"Unsupported native MIDI mapping mode: {mapping_mode}")
        self._mapping_mode = mapping_mode
        self._momentary_start_midi = int(momentary_start_midi)
        self._toggle_start_midi = int(toggle_start_midi)
        self._panic_midi_note = (
            int(panic_midi_note)
            if panic_midi_note is not None and 0 <= int(panic_midi_note) <= 127
            else None
        )
        if self._mapping_mode == "sequential_banks":
            momentary_range = range(self._momentary_start_midi, self._momentary_start_midi + self._max_bands)
            toggle_range = range(self._toggle_start_midi, self._toggle_start_midi + self._max_bands)
            if set(momentary_range) & set(toggle_range):
                raise ValueError("Native MIDI momentary and toggle banks must not overlap")
        self._lock = threading.RLock()
        # midi_note -> voice_id, partitioned so toggle note-offs are ignored.
        self._held: dict[int, int] = {}
        self._toggles: dict[int, int] = {}
        self._next_voice_id = self._VOICE_ID_BASE

    # ── Configuration ────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)
            if not self._enabled:
                self._release_all_locked()

    @property
    def anchor_midi(self) -> int:
        return self._anchor_midi

    def set_anchor_midi(self, anchor_midi: int) -> None:
        with self._lock:
            self._anchor_midi = int(anchor_midi)

    @property
    def held_notes(self) -> dict[int, int]:
        """Snapshot of midi_note → voice_id for active native notes."""
        with self._lock:
            return {**self._held, **self._toggles}

    @property
    def mapping_mode(self) -> str:
        return self._mapping_mode

    def _sequential_mapping(self, note: int):
        if self._momentary_start_midi <= note < self._momentary_start_midi + self._max_bands:
            return "momentary", map_sequential_harmonic(
                note, self._store.f1, self._momentary_start_midi, max_bands=self._max_bands
            )
        if self._toggle_start_midi <= note < self._toggle_start_midi + self._max_bands:
            return "toggle", map_sequential_harmonic(
                note, self._store.f1, self._toggle_start_midi, max_bands=self._max_bands
            )
        return None, None

    # ── Note lifecycle ───────────────────────────────────────────────────

    def note_on(self, midi_note: int, velocity: int) -> Optional[int]:
        """Handle note-on.  Velocity 0 is treated as note-off.

        Returns the allocated ``voice_id``, or ``None`` if ignored/released.
        """
        if not self._enabled:
            return None
        if velocity <= 0:
            self.note_off(midi_note)
            return None

        note = max(0, min(127, int(midi_note)))
        if note == self._panic_midi_note:
            log.info("Native MIDI panic key pressed: midi=%d", note)
            self.panic()
            self._store.panic()
            return None
        with self._lock:
            mapping_label = "note"
            if self._mapping_mode == "sequential_banks":
                bank, mapping = self._sequential_mapping(note)
                if mapping is None:
                    log.debug("Native note ignored outside configured banks: midi=%d", note)
                    return None
                if bank == "toggle" and note in self._toggles:
                    vid = self._toggles.pop(note)
                    self._store.voice_off(vid)
                    log.debug("Native toggle OFF midi=%d n=%d vid=%d", note, mapping.harmonic_n, vid)
                    return vid
                owners = self._toggles if bank == "toggle" else self._held
                mapping_label = bank
            else:
                mapping = map_midi_note(
                    note,
                    f1=self._store.f1,
                    anchor_midi=self._anchor_midi,
                )
                owners = self._held
            # Re-trigger: release previous voice for this key first
            prev = owners.pop(note, None)
            if prev is not None:
                self._store.voice_off(prev)
            band = mapping.store_band(self._max_bands)
            gain = velocity_to_gain(
                velocity,
                min_gain=self._vel_min,
                max_gain=self._vel_max,
            )
            vid = self._next_voice_id
            self._next_voice_id += 1
            owners[note] = vid
            self._store.voice_on(band, vid, mapping.frequency_hz, gain=gain)
            log.debug(
                "Native %s ON midi=%d n=%d band=%d freq=%.3fHz gain=%.3f vid=%d src=%s",
                mapping_label,
                note,
                mapping.harmonic_n,
                band,
                mapping.frequency_hz,
                gain,
                vid,
                mapping.source,
            )
            return vid

    def note_off(self, midi_note: int) -> Optional[int]:
        """Release the voice that originated from this MIDI note, if any."""
        note = max(0, min(127, int(midi_note)))
        with self._lock:
            if self._mapping_mode == "sequential_banks":
                bank, _ = self._sequential_mapping(note)
                if bank == "toggle":
                    return None
                if bank is None:
                    return None
            vid = self._held.pop(note, None)
            if vid is None:
                return None
            self._store.voice_off(vid)
            log.debug("Native note OFF midi=%d vid=%d", note, vid)
            return vid

    def handle_message(self, msg) -> None:
        """Dispatch a mido-like message (``type``, ``note``, ``velocity``)."""
        if not self._enabled:
            return
        if msg.type == "note_on":
            self.note_on(msg.note, msg.velocity)
        elif msg.type == "note_off":
            self.note_off(msg.note)

    def panic(self) -> None:
        """Release all native-held voices (does not call store.panic)."""
        with self._lock:
            self._release_all_locked()

    def _release_all_locked(self) -> None:
        for vid in [*self._held.values(), *self._toggles.values()]:
            self._store.voice_off(vid)
        self._held.clear()
        self._toggles.clear()


class NativeMidiNoteSource:
    """Open generic keyboard MIDI ports and feed :class:`NativeNoteHandler`.

    Ports matching Launchpad / Minilab patterns are skipped so dedicated
    controllers keep exclusive ownership of those devices.
    """

    def __init__(
        self,
        handler: NativeNoteHandler,
        exclude_patterns: Sequence[str] = config.NATIVE_MIDI_EXCLUDE_PATTERNS,
    ) -> None:
        if not HAS_MIDO:
            raise ImportError("mido is required for MIDI control.")
        self._handler = handler
        self._exclude_patterns = tuple(exclude_patterns)
        self._ports: list = []
        self._threads: list[threading.Thread] = []
        self._running = False

    def start(self) -> None:
        if not self._handler.enabled:
            log.info("Native MIDI note source disabled by configuration")
            return
        names = self._eligible_ports()
        if not names:
            log.info(
                "Native MIDI note source: no generic keyboard ports found "
                "(excluded patterns=%s)",
                self._exclude_patterns,
            )
            return
        self._running = True
        for name in names:
            try:
                port = mido.open_input(name)
            except Exception as exc:
                log.warning("Could not open MIDI input %r: %s", name, exc)
                continue
            self._ports.append(port)
            thread = threading.Thread(
                target=self._run_port,
                args=(port, name),
                name=f"shaper-native-midi-{name[:24]}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
            log.info("Native MIDI note source listening: %s", name)

    def stop(self) -> None:
        self._running = False
        self._handler.panic()
        for port in self._ports:
            try:
                port.close()
            except Exception:
                pass
        self._ports.clear()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()

    def panic(self) -> None:
        self._handler.panic()

    def _eligible_ports(self) -> list[str]:
        result: list[str] = []
        for name in mido.get_input_names():
            lower = name.lower()
            if any(pat.lower() in lower for pat in self._exclude_patterns):
                continue
            result.append(name)
        return result

    def _run_port(self, port, name: str) -> None:
        try:
            for msg in port:
                if not self._running:
                    break
                self._handler.handle_message(msg)
        except Exception as exc:
            if self._running:
                log.debug("Native MIDI port %r closed: %s", name, exc)


class LaunchpadMiniControl:
    """Launchpad Mini → Shaper voice_on/voice_off for 32 harmonics.

    Lower half: momentary (H01..H32).
    Upper half: toggle, same range (H01..H32, repeats).

    Lights are managed via MIDI output to the Launchpad.
    """

    # Feedback colors (Launchpad velocity = color in Programmer mode)
    COLOR_OFF = 0
    COLOR_ON = 60          # Green High
    COLOR_TOGGLE_ON = 21   # Orange

    def __init__(
        self,
        store: VoiceParameterStore,
        port_pattern: str = config.LAUNCHPAD_PORT_PATTERN,
        split_mode: bool = config.SPLIT_MODE_ENABLED_BY_DEFAULT,
        max_voices: int = config.MAX_VOICES,
    ):
        if not HAS_MIDO:
            raise ImportError("mido is required for MIDI control.")
        self._store = store
        self._port_pattern = port_pattern
        self._split_mode = split_mode
        self._max_voices = max_voices

        self._in_port = None
        self._out_port: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Autodetected stride: 8 (Note mode) or 16 (Programmer mode)
        self._stride: Optional[int] = None

        # Toggle state: harmonic_n -> pad note
        self._toggled: Set[int] = set()
        # Active momentary notes: pad note -> (harmonic_n, voice_id)
        self._held: dict[int, tuple] = {}
        # Active toggle notes: pad note -> harmonic_n (for light management)
        self._toggle_pads: dict[int, int] = {}  # note -> n
        # Order of toggles (for note stealing visual feedback)
        self._toggle_order: "OrderedDict[int, None]" = OrderedDict()

        # Voice ID tracking: we use a monotonic local ID. The (n -> voice_id)
        # map is for TOGGLES only; momentary tracks voice_id per pad in _held.
        self._next_voice_id = 1
        self._n_to_vid: dict[int, int] = {}  # harmonic_n -> voice_id (toggles)

    # ─── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        port_name = self._find_port()
        if not port_name:
            log.warning("Launchpad not found (pattern=%r). Pad control disabled.",
                        self._port_pattern)
            return
        try:
            self._in_port = mido.open_input(port_name)
        except Exception as exc:
            log.error("Could not open Launchpad input %r: %s", port_name, exc)
            return

        # Try to open output port for lights. Heuristic: same name, or strip
        # the trailing "MIDI 1" and look for an output.
        self._out_port = self._find_output_port(port_name)
        if self._out_port is not None:
            log.info("Launchpad output (lights) opened: %s", self._out_port.name)
        else:
            log.warning("Launchpad output port not found. Pad lights will be skipped.")

        self._running = True
        self._thread = threading.Thread(target=self._run, name="shaper-launchpad",
                                       daemon=True)
        self._thread.start()
        log.info("Launchpad control started: %s  (split_mode=%s)",
                 port_name, self._split_mode)

    def stop(self) -> None:
        self._running = False
        # Clear all lights
        self._all_lights_off()
        if self._in_port:
            try:
                self._in_port.close()
            except Exception:
                pass
            self._in_port = None
        if self._out_port:
            try:
                self._out_port.close()
            except Exception:
                pass
            self._out_port = None

    def _find_port(self) -> Optional[str]:
        for name in mido.get_input_names():
            if self._port_pattern.lower() in name.lower():
                return name
        return None

    def _find_output_port(self, in_name: str):
        """Find the matching output port for light feedback."""
        # Try exact match first
        if in_name in mido.get_output_names():
            try:
                return mido.open_output(in_name)
            except Exception as exc:
                log.warning("Output port %r open failed: %s", in_name, exc)
        # Heuristic: strip "MIDI 1" and look for that
        for out_name in mido.get_output_names():
            if in_name.split(" MIDI ")[0] in out_name:
                try:
                    return mido.open_output(out_name)
                except Exception as exc:
                    log.warning("Output port %r open failed: %s", out_name, exc)
                    continue
        return None

    # ─── MIDI loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        for msg in self._in_port:
            if not self._running:
                break
            self._handle(msg)

    def _handle(self, msg) -> None:
        if msg.type == "control_change":
            if msg.control == config.SPLIT_MODE_TOGGLE_CC:
                self._handle_split_toggle(msg.value)
            return

        if msg.type not in ("note_on", "note_off"):
            return

        velocity = msg.velocity if msg.type == "note_on" else 0

        # Autodetect stride on first pad event (handle full 0-127 note range)
        if self._stride is None and 0 <= msg.note < 128:
            # Look at the y implied by note and which stride makes it fit
            # the 8x8 grid: y = note // stride must be < 8.
            for s in (16, 8):
                if msg.note // s < 8 and msg.note % s < 8:
                    self._stride = s
                    log.info("Launchpad stride autodetected: %d", s)
                    break
            if self._stride is None:
                # Stride 12 (MPC) or other — fall back to 16
                self._stride = 16
                log.info("Launchpad stride fallback: 16")

        if self._stride is None:
            return  # no stride yet, can't decode
        if not (0 <= msg.note < 128):
            return
        rel = msg.note
        x = rel % self._stride
        y = rel // self._stride
        if x >= 8 or y >= 8:
            return  # outside the 8x8 grid (right column or scene buttons)
        # Same mapping as NaturalHarmony harmonic_beacon:
        # y=0 is the TOP row (physical A), y=7 is the BOTTOM (physical H).
        # Invert so row_from_bottom=0 = H (momentary), row_from_bottom=7 = A (toggle).
        row_from_bottom = 7 - y

        if self._split_mode:
            if row_from_bottom < 4:
                # Lower half (rows 0-3 physical from bottom, H-G-F-E): momentary 1-32
                n = 1 + x + (row_from_bottom * 8)
                if n > 32:
                    return
                if velocity > 0:
                    self._note_on_momentary(msg.note, n, velocity, msg.channel)
                else:
                    self._note_off_momentary(msg.note, n, msg.channel)
            else:
                # Upper half (rows 4-7 physical from bottom, D-C-B-A): toggle 1-32
                n = 1 + x + ((row_from_bottom - 4) * 8)
                if n > 32:
                    return
                if velocity > 0:
                    self._note_on_toggle(msg.note, n, msg.channel)
                # NOTE: note_off on upper half is ignored (latching)
        else:
            # Full mode (no split): momentary 1..32, top rows 4-7 ignored
            n = 1 + x + (row_from_bottom * 8)
            if not (1 <= n <= 32):
                return
            if velocity > 0:
                self._note_on_momentary(msg.note, n, velocity, msg.channel)
            else:
                self._note_off_momentary(msg.note, n, msg.channel)

    # ─── Momentary (lower half) ──────────────────────────────────────────

    def _note_on_momentary(self, pad_note: int, n: int, velocity: int, channel: int):
        if pad_note in self._held:
            return  # already held (duplicate event)
        vid = self._next_voice_id
        self._next_voice_id += 1
        self._held[pad_note] = (n, vid)  # track (n, vid) per pad
        freq = self._store.f1 * n
        vel_norm = velocity / 127.0
        self._store.voice_on(n, vid, freq, gain=vel_norm)
        log.debug("Pad %d ON  (momentary n=%d freq=%.1fHz)", pad_note, n, freq)
        self._set_pad_light(pad_note, self.COLOR_ON, channel)

    def _note_off_momentary(self, pad_note: int, n: int, channel: int):
        if pad_note not in self._held:
            return
        entry = self._held.pop(pad_note)
        n_stored, vid = entry
        self._store.voice_off(vid)
        log.debug("Pad %d OFF (momentary n=%d)", pad_note, n_stored)
        self._set_pad_light(pad_note, self.COLOR_OFF, channel)

    # ─── Toggle (upper half) ────────────────────────────────────────────

    def _note_on_toggle(self, pad_note: int, n: int, channel: int):
        if n in self._toggled:
            # Toggle OFF
            self._toggled.discard(n)
            self._toggle_pads.pop(pad_note, None)
            try:
                self._toggle_order.pop(n)
            except KeyError:
                pass
            vid = self._n_to_vid.pop(n, None)
            if vid is not None:
                self._store.voice_off(vid)
            log.debug("Pad %d TOGGLE OFF (n=%d)", pad_note, n)
            self._set_pad_light(pad_note, self.COLOR_OFF, channel)
            return

        # Toggle ON
        # Enforce polyphony by stealing the oldest toggle if needed
        if self._count_active() >= self._max_voices:
            self._steal_oldest_toggle()

        vid = self._next_voice_id
        self._next_voice_id += 1
        self._toggled.add(n)
        self._toggle_pads[pad_note] = n
        self._toggle_order[n] = None
        self._n_to_vid[n] = vid
        freq = self._store.f1 * n
        self._store.voice_on(n, vid, freq)
        log.debug("Pad %d TOGGLE ON  (n=%d freq=%.1fHz)", pad_note, n, freq)
        self._set_pad_light(pad_note, self.COLOR_TOGGLE_ON, channel)

    # ─── Split mode toggle (CC104) ───────────────────────────────────────

    def _handle_split_toggle(self, value: int):
        if value == 0:
            return
        self._split_mode = not self._split_mode
        log.info("Split mode: %s", "ON" if self._split_mode else "OFF")
        # Reset all lights + state
        self._all_lights_off()
        self._clear_all_voices()

    def _clear_all_voices(self):
        # Send voice_off for all held + toggled
        for vid in list(self._n_to_vid.values()):
            self._store.voice_off(vid)
        for _, vid in list(self._held.values()):
            self._store.voice_off(vid)
        self._n_to_vid.clear()
        self._held.clear()
        self._toggled.clear()
        self._toggle_pads.clear()
        self._toggle_order.clear()

    # ─── Lights ───────────────────────────────────────────────────────────

    def _set_pad_light(self, pad_note: int, color: int, channel: int = 0):
        if self._out_port is None:
            return
        try:
            # Launchpad uses velocity as the color in Programmer mode.
            # For "off" we send note_on with vel=0 (universal "release").
            if color == self.COLOR_OFF:
                msg = mido.Message('note_on', note=pad_note, velocity=0, channel=channel)
            else:
                msg = mido.Message('note_on', note=pad_note, velocity=color, channel=channel)
            self._out_port.send(msg)
        except Exception as exc:
            log.debug("Light send failed (pad %d): %s", pad_note, exc)

    def _all_lights_off(self):
        if self._out_port is None:
            return
        for n in range(128):
            try:
                self._out_port.send(
                    mido.Message('note_on', note=n, velocity=0, channel=0)
                )
            except Exception:
                pass

    # ─── Polyphony + note stealing ───────────────────────────────────────

    def _count_active(self) -> int:
        return len(self._held) + len(self._toggled)

    def _steal_oldest_toggle(self):
        """Steal the oldest toggled harmonic so a new one can be toggled on."""
        if not self._toggle_order:
            return
        oldest_n, _ = self._toggle_order.popitem(last=False)
        # Find the pad for this n
        pad_note = None
        for p, n in list(self._toggle_pads.items()):
            if n == oldest_n:
                pad_note = p
                break
        if pad_note is not None:
            self._toggle_pads.pop(pad_note, None)
        self._toggled.discard(oldest_n)
        vid = self._n_to_vid.pop(oldest_n, None)
        if vid is not None:
            self._store.voice_off(vid)
        if pad_note is not None:
            self._set_pad_light(pad_note, self.COLOR_OFF, 0)
        log.info("Poly limit reached — stole toggle n=%d (pad %s)", oldest_n, pad_note)

    # ─── External API: panic ────────────────────────────────────────────

    def panic(self):
        """Clear controller state and feedback after any Shaper panic."""
        self._clear_all_voices()
        self._all_lights_off()


class Minilab3Control:
    """Minilab3 encoders and faders for four upper active voices.

    This controller existed in the NaturalHarmony original but was absent
    from the digital-beacon fork.  Its mappings are preserved here:

    - sliders 1-4: gain
    - top knobs 1-4: pan
    - bottom knobs 5-8: phase
    - modulation wheel: master gain
    - pad 4: panic

    The lowest active harmonic remains an unmodified reference; controls map
    to the next four active harmonics in ascending order.
    """

    def __init__(
        self,
        store: VoiceParameterStore,
        port_pattern: str = config.MINILAB_PORT_PATTERN,
    ) -> None:
        if not HAS_MIDO:
            raise ImportError("mido is required for MIDI control.")
        self._store = store
        self._port_pattern = port_pattern
        self._port: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._slider_to_slot = {
            cc: index for index, cc in enumerate(config.MINILAB_SLIDER_CCS)
        }
        self._pan_to_slot = {
            cc: index for index, cc in enumerate(config.MINILAB_PAN_CCS)
        }
        self._phase_to_slot = {
            cc: index for index, cc in enumerate(config.MINILAB_PHASE_CCS)
        }

    def start(self) -> None:
        port_name = self._find_port()
        if not port_name:
            log.warning(
                "Minilab3 not found (pattern=%r); controller disabled.",
                self._port_pattern,
            )
            return
        try:
            self._port = mido.open_input(port_name)
        except Exception as exc:
            log.error("Could not open Minilab3 input %r: %s", port_name, exc)
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="shaper-minilab",
            daemon=True,
        )
        self._thread.start()
        log.info("Minilab3 control started: %s", port_name)

    def stop(self) -> None:
        self._running = False
        if self._port:
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None

    def _find_port(self) -> Optional[str]:
        for name in mido.get_input_names():
            if self._port_pattern.lower() in name.lower():
                return name
        return None

    def _run(self) -> None:
        for msg in self._port:
            if not self._running:
                break
            self._handle(msg)

    def _handle(self, msg) -> None:
        if msg.type == "control_change":
            self._handle_cc(msg.control, msg.value)
        elif msg.type in ("note_on", "note_off") and msg.velocity > 0:
            self._handle_pad(msg.note)

    def _handle_cc(self, cc: int, value: int) -> None:
        normalized = value / 127.0
        if cc == 1:
            self._store.set_master_gain(normalized)
            return

        slot = self._slider_to_slot.get(cc)
        if slot is not None:
            n = self._slot_to_harmonic_n(slot)
            if n is not None:
                self._store.set_gain(n, normalized)
            return

        slot = self._pan_to_slot.get(cc)
        if slot is not None:
            n = self._slot_to_harmonic_n(slot)
            if n is not None:
                self._store.set_pan(n, normalized * 2.0 - 1.0)
            return

        slot = self._phase_to_slot.get(cc)
        if slot is not None:
            n = self._slot_to_harmonic_n(slot)
            if n is not None:
                self._store.set_phase(n, normalized * 360.0)

    def _handle_pad(self, note: int) -> None:
        if note == config.MINILAB_PANIC_PAD:
            self._store.panic()

    def _slot_to_harmonic_n(self, slot: int) -> Optional[int]:
        active = sorted(self._store.get_snapshot())
        if slot + 1 < len(active):
            return active[slot + 1]
        return None
