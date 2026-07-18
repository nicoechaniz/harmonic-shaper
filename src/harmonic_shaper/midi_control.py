"""Launchpad Mini and Minilab3 control for the standalone Harmonic Shaper.

Brought from NaturalHarmony/harmonic_beacon/main.py (the original Pad Mode
logic) and adapted to:
  - 32 bands (1..32) instead of 64 — the top half repeats the same range
    as the bottom half (per Nico's spec: F1..F32 on lower half, F1..F32 on
    upper half with toggle behavior).
  - Direct routing into the standalone Shaper voice store.
  - Polyphony limit + visual note stealing on the Launchpad (the new lights
    turn off when the limit is exceeded).

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
"""

import logging
import threading
from collections import OrderedDict
from typing import Optional, Set

try:
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False
    mido = None

from .state import VoiceParameterStore
from . import config

log = logging.getLogger(__name__)


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
