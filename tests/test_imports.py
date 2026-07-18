"""All package modules must import without opening hardware or sockets."""

from __future__ import annotations

import importlib


MODULES = [
    "harmonic_shaper",
    "harmonic_shaper.__main__",
    "harmonic_shaper.api",
    "harmonic_shaper.audio_engine",
    "harmonic_shaper.config",
    "harmonic_shaper.contract_codec",
    "harmonic_shaper.harmonic_mapping",
    "harmonic_shaper.main",
    "harmonic_shaper.midi_control",
    "harmonic_shaper.osc_receiver",
    "harmonic_shaper.state",
    "harmonic_shaper.synth_pure",
    "harmonic_shaper.voice_cache",
]


def test_every_module_imports_headlessly() -> None:
    for module_name in MODULES:
        imported = importlib.import_module(module_name)
        assert imported.__name__ == module_name
