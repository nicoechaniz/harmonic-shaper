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
        "/digital/harmonic/*/pan",
        "/digital/harmonic/*/phase",
        "/digital/master",
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
