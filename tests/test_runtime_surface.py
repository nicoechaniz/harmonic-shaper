"""Standalone runtime flags, API routes, and local OSC table."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from harmonic_shaper import api
from harmonic_shaper.main import build_parser
from harmonic_shaper.state import VoiceParameterStore


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_slave_listener_is_opt_in() -> None:
    parser = build_parser()
    assert parser.parse_args([]).slave is False
    assert parser.parse_args(["--slave"]).slave is True


def test_native_midi_defaults_and_flags() -> None:
    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.no_native_midi is False
    assert defaults.f1 == 40.40
    assert defaults.anchor == 24
    assert parser.parse_args(["--no-native-midi"]).no_native_midi is True


def test_local_osc_table_stays_wire_compatible() -> None:
    source = (REPO_ROOT / "src/harmonic_shaper/osc_receiver.py").read_text()
    addresses = set(re.findall(r'\.map\("([^"]+)"', source))
    assert addresses == {
        "/beacon/voice/on",
        "/beacon/voice/off",
        "/beacon/voice/freq",
        "/beacon/f1",
        "/beacon/panic",
        "/beacon/level",
        "/digital/harmonic/*/gain",
        "/digital/harmonic/*/envelope",
        "/digital/harmonic/*/pan",
        "/digital/harmonic/*/phase",
        "/digital/master",
        "/digital/ceiling",
        "/digital/clock/bpm",
        "/digital/settle_beats",
        "/digital/generator/enable",
        "/digital/arp/*/enable",
        "/digital/arp/*/rate",
        "/digital/arp/*/direction",
        "/digital/arp/*/density",
        "/digital/arp/*/register_lo",
        "/digital/arp/*/register_hi",
        "/digital/arp/*/gate",
        "/digital/arp/*/gain",
        "/digital/perc/enable",
        "/digital/perc/rate",
        "/digital/perc/gain",
        "/digital/perc/accent",
        "/digital/panic",
    }


@pytest.mark.skipif(not api.HAS_FASTAPI, reason="FastAPI is not installed")
def test_extracted_api_routes_are_present() -> None:
    app = api.create_app(VoiceParameterStore())
    routes = {(route.path, method) for route in app.routes for method in getattr(route, "methods", set())}
    assert ("/api/state", "GET") in routes
    assert ("/api/panic", "POST") in routes
    assert ("/api/shaper/global/{param}", "POST") in routes
    assert ("/api/shaper/{n}/{param}", "POST") in routes
    assert any(route.path == "/ws" for route in app.routes)
