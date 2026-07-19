"""OSC input for the standalone Shaper.

The native Instrument Control v1 listener is wire-compatible with the
digital-beacon fork: ``/digital/*`` on UDP 9002.  ``/shaper/*`` is the planned
namespace for a later contract bump and is intentionally not mapped here.

NaturalHarmony ``/beacon/*`` broadcasts on UDP 9001 remain available as an
optional slave listener.  Standalone mode does not bind that port.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Optional

try:
    from pythonosc import dispatcher as osc_dispatcher
    from pythonosc import osc_server

    HAS_OSC = True
except ImportError:
    HAS_OSC = False

from . import config
from .state import VoiceParameterStore

log = logging.getLogger(__name__)


class _ReusePortUDPServer(osc_server.BlockingOSCUDPServer if HAS_OSC else object):
    """OSC server that can co-listen with the NH visualizer on UDP 9001."""

    def server_bind(self) -> None:
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            log.warning("SO_REUSEPORT unavailable; slave port may be unavailable")
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


class ShaperOSCReceiver:
    """Native OSC listener with an opt-in NaturalHarmony slave listener."""

    def __init__(
        self,
        store: VoiceParameterStore,
        beacon_port: int = config.BEACON_BROADCAST_PORT,
        shaper_port: int = config.SHAPER_OSC_PORT,
        host: str = config.OSC_HOST,
        slave: bool = False,
    ) -> None:
        if not HAS_OSC:
            raise ImportError("python-osc is required for OSC control")
        self._store = store
        self._beacon_port = beacon_port
        self._shaper_port = shaper_port
        self._host = host
        self._slave = slave
        self._servers: list = []
        self._threads: list[threading.Thread] = []

    @property
    def slave_enabled(self) -> bool:
        return self._slave

    def start(self) -> None:
        self._start_shaper_listener()
        if self._slave:
            self._start_beacon_listener()

    def stop(self) -> None:
        for server in self._servers:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                log.debug("Error while stopping OSC server", exc_info=True)
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._servers.clear()
        self._threads.clear()

    # NaturalHarmony broadcast slave (/beacon/* on 9001)

    def _start_beacon_listener(self) -> None:
        dispatcher = osc_dispatcher.Dispatcher()
        dispatcher.map("/beacon/voice/on", self._on_voice_on)
        dispatcher.map("/beacon/voice/off", self._on_voice_off)
        dispatcher.map("/beacon/voice/freq", self._on_voice_freq)
        dispatcher.map("/beacon/f1", self._on_f1)
        dispatcher.map("/beacon/panic", lambda *_: self._store.panic())
        dispatcher.map("/beacon/level", self._on_beacon_level)
        dispatcher.set_default_handler(lambda *_: None)
        self._serve(dispatcher, self._beacon_port, "shaper-beacon-osc")
        log.info("Optional slave OSC on %s:%d (/beacon/*)", self._host, self._beacon_port)

    def _on_voice_on(
        self,
        addr,
        voice_id,
        freq,
        gain,
        source_note,
        harmonic_n=None,
        *_,
    ) -> None:
        del addr
        n = int(source_note if harmonic_n is None else harmonic_n)
        if not 1 <= n <= config.N_BANDS:
            log.warning("Ignoring slave voice outside 1..%d: %d", config.N_BANDS, n)
            return
        self._store.voice_on(n, int(voice_id), float(freq), gain=float(gain))
        self._store.record_strum(time.time())

    def _on_voice_off(self, addr, voice_id, *_) -> None:
        del addr
        self._store.voice_off(int(voice_id))

    def _on_voice_freq(self, addr, voice_id, freq, *_) -> None:
        del addr
        self._store.voice_freq(int(voice_id), float(freq))

    def _on_f1(self, addr, f1, *_) -> None:
        del addr
        self._store.update_f1(float(f1))

    def _on_beacon_level(self, addr, level, *_) -> None:
        del addr
        self._store.set_beacon_level(float(level))

    # Current native wire protocol (/digital/* on 9002)

    def _start_shaper_listener(self) -> None:
        dispatcher = osc_dispatcher.Dispatcher()
        dispatcher.map("/digital/harmonic/*/gain", self._on_gain)
        dispatcher.map("/digital/harmonic/*/envelope", self._on_envelope)
        dispatcher.map("/digital/harmonic/*/pan", self._on_pan)
        dispatcher.map("/digital/harmonic/*/phase", self._on_phase)
        dispatcher.map("/digital/master", self._on_master)
        dispatcher.map("/digital/panic", lambda *_: self._store.panic())
        dispatcher.set_default_handler(lambda *_: None)
        self._serve(dispatcher, self._shaper_port, "shaper-direct-osc")
        log.info("Native OSC on %s:%d (/digital/*)", self._host, self._shaper_port)

    def _serve(self, dispatcher, port: int, thread_name: str) -> None:
        server = _ReusePortUDPServer((self._host, port), dispatcher)
        self._servers.append(server)
        thread = threading.Thread(
            target=server.serve_forever,
            name=thread_name,
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)

    @staticmethod
    def _parse_n(addr: str) -> Optional[int]:
        try:
            n = int(addr.split("/")[3])
        except (IndexError, ValueError):
            return None
        return n if 1 <= n <= config.N_BANDS else None

    def _on_gain(self, addr, value, *_) -> None:
        n = self._parse_n(addr)
        if n is not None:
            self._store.set_gain(n, float(value))

    def _on_envelope(self, addr, value, *_) -> None:
        n = self._parse_n(addr)
        if n is not None:
            self._store.set_harmonic_envelope(n, float(value))

    def _on_pan(self, addr, value, *_) -> None:
        n = self._parse_n(addr)
        if n is not None:
            self._store.set_pan(n, float(value))

    def _on_phase(self, addr, value, *_) -> None:
        n = self._parse_n(addr)
        if n is not None:
            self._store.set_phase(n, float(value))

    def _on_master(self, addr, value, *_) -> None:
        del addr
        self._store.set_master_gain(float(value))
