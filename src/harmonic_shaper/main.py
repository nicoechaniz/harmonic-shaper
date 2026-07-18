"""Standalone runtime for the Harmonic Shaper."""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from collections.abc import Sequence

from . import config
from .audio_engine import AudioEngine
from .midi_control import (
    LaunchpadMiniControl,
    Minilab3Control,
    NativeMidiNoteSource,
    NativeNoteHandler,
)
from .osc_receiver import ShaperOSCReceiver
from .state import VoiceParameterStore

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Harmonic Shaper")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--list-midi", action="store_true", help="List MIDI ports and exit")
    parser.add_argument("--device", help="Audio device ID or name substring")
    parser.add_argument("--no-audio", action="store_true", help="Disable the audio stream")
    parser.add_argument(
        "--no-midi",
        action="store_true",
        help="Disable Launchpad, Minilab3, and native keyboard MIDI",
    )
    parser.add_argument(
        "--no-native-midi",
        action="store_true",
        help="Disable native MIDI-note harmonic source (generic keyboards)",
    )
    parser.add_argument(
        "--f1",
        type=float,
        default=config.DEFAULT_F1,
        help=f"Fundamental frequency f1 in Hz (default {config.DEFAULT_F1})",
    )
    parser.add_argument(
        "--anchor",
        type=int,
        default=config.DEFAULT_ANCHOR_MIDI,
        help=f"Anchor MIDI note for f1 (default {config.DEFAULT_ANCHOR_MIDI} = C1)",
    )
    parser.add_argument("--no-osc", action="store_true", help="Disable all OSC input")
    parser.add_argument("--no-api", action="store_true", help="Disable HTTP/WebSocket state API")
    parser.add_argument(
        "--slave",
        action="store_true",
        help="Also consume NaturalHarmony /beacon/* broadcasts on UDP 9001",
    )
    parser.add_argument("--osc-host", default=config.OSC_HOST, help="OSC bind host")
    parser.add_argument("--osc-port", type=int, default=config.SHAPER_OSC_PORT, help="Native /digital OSC port")
    parser.add_argument(
        "--slave-port",
        type=int,
        default=config.BEACON_BROADCAST_PORT,
        help="Optional /beacon slave port",
    )
    parser.add_argument("--api-host", default=config.API_HOST, help="API bind host")
    parser.add_argument("--api-port", type=int, default=config.API_PORT, help="API port")
    parser.add_argument("--log-level", default=config.LOG_LEVEL, help="Python log level")
    return parser


def list_midi_ports() -> None:
    try:
        import mido
    except ImportError:
        print("mido is not installed")
        return
    print("MIDI input ports:")
    for name in mido.get_input_names():
        print(f"  - {name}")
    print("MIDI output ports:")
    for name in mido.get_output_names():
        print(f"  - {name}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    if args.list_devices:
        print(AudioEngine.list_devices())
        return 0
    if args.list_midi:
        list_midi_ports()
        return 0

    store = VoiceParameterStore()
    store.update_f1(float(args.f1))

    native_enabled = (
        config.NATIVE_MIDI_SOURCE_ENABLED
        and not args.no_midi
        and not args.no_native_midi
    )
    native_handler = NativeNoteHandler(
        store,
        anchor_midi=int(args.anchor),
        enabled=native_enabled,
    )

    audio = None if args.no_audio else AudioEngine(store, device=args.device or config.AUDIO_DEVICE)
    osc = None
    if not args.no_osc:
        osc = ShaperOSCReceiver(
            store,
            beacon_port=args.slave_port,
            shaper_port=args.osc_port,
            host=args.osc_host,
            slave=args.slave,
        )

    midi_controls: list = []
    if not args.no_midi:
        launchpad = LaunchpadMiniControl(store)
        minilab = Minilab3Control(store)
        midi_controls.extend((launchpad, minilab))
        if native_enabled:
            midi_controls.append(NativeMidiNoteSource(native_handler))

        def _panic_surfaces() -> None:
            launchpad.panic()
            native_handler.panic()

        store._panic_callback = _panic_surfaces

    api_server = None
    api_thread = None
    stop_event = threading.Event()

    def _request_shutdown(signum, frame) -> None:
        del frame
        log.info("Signal %d received; stopping", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGHUP, _request_shutdown)

    log.info(
        "Starting Harmonic Shaper: f1=%.2f Hz, anchor_midi=%d, bands=%d, polyphony=%d, native_midi=%s",
        store.f1,
        native_handler.anchor_midi,
        config.N_BANDS,
        config.MAX_VOICES,
        native_enabled,
    )

    try:
        if audio is not None:
            audio.start()
        if osc is not None:
            osc.start()
        for controller in midi_controls:
            controller.start()

        if not args.no_api:
            import uvicorn

            from .api import create_app

            api_config = uvicorn.Config(
                create_app(store),
                host=args.api_host,
                port=args.api_port,
                log_level="warning",
                access_log=False,
            )
            api_server = uvicorn.Server(api_config)
            api_thread = threading.Thread(
                target=api_server.run,
                name="shaper-api",
                daemon=True,
            )
            api_thread.start()

            startup_deadline = time.monotonic() + 5.0
            while (
                not api_server.started
                and api_thread.is_alive()
                and time.monotonic() < startup_deadline
            ):
                time.sleep(0.01)
            if not api_server.started:
                raise RuntimeError("FastAPI server did not start within 5 seconds")

        if osc is not None:
            log.info("Native OSC: %s:%d /digital/*", args.osc_host, args.osc_port)
            log.info("Planned namespace (not yet on wire): /shaper/*")
            if args.slave:
                log.info("Slave OSC enabled: %s:%d /beacon/*", args.osc_host, args.slave_port)
            else:
                log.info("Slave OSC off (pass --slave for optional /beacon/*)")
        if native_enabled:
            log.info(
                "Native MIDI note source ON (anchor=%d, f1=%.2f Hz)",
                native_handler.anchor_midi,
                store.f1,
            )
        if not args.no_api:
            log.info("State API: http://%s:%d (WebSocket /ws)", args.api_host, args.api_port)
        log.info("Shaper running; press Ctrl-C to stop")

        while not stop_event.wait(0.5):
            pass
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received; stopping")
    finally:
        store.panic()
        for controller in reversed(midi_controls):
            controller.stop()
        if osc is not None:
            osc.stop()
        if api_server is not None:
            api_server.should_exit = True
        if api_thread is not None:
            api_thread.join(timeout=5.0)
        if audio is not None:
            audio.stop()

    log.info("Harmonic Shaper stopped")
    return 0
