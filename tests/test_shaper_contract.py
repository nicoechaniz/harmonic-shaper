"""Tests for the Shaper Instrument Control v1 contract.

Validates contracts/shaper.contract.json against the copied contract_codec,
checks the golden sidecar, and asserts that every OSC address mapped in
digital-beacon's osc_receiver.py is covered by the manifest.
"""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "contracts" / "shaper.contract.json"
GOLDEN_PATH = REPO_ROOT / "contracts" / "shaper.contract_id.golden"

# Source of truth for the live OSC table (read-only sibling repo by default).
_DEFAULT_OSC_RECEIVER = (
    Path.home() / "Projects" / "digital-beacon" / "digital_beacon" / "osc_receiver.py"
)
OSC_RECEIVER_PATH = Path(
    os.environ.get("DIGITAL_BEACON_OSC_RECEIVER", str(_DEFAULT_OSC_RECEIVER))
)

# d.map("address", ...) — capture the address pattern string literal.
_MAP_RE = re.compile(
    r"""\.map\(\s*(['"])(?P<addr>[^'"]+)\1""",
    re.MULTILINE,
)


import sys

_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from harmonic_shaper.contract_codec import (  # noqa: E402
    check_golden_sidecar,
    contract_id_from_manifest,
    decode_manifest,
    encode_manifest,
    load_manifest,
    validate_manifest,
)


def extract_osc_addresses(source_path: Path) -> list[str]:
    """Extract every OSC address pattern registered via dispatcher.map()."""

    text = source_path.read_text(encoding="utf-8")
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _MAP_RE.finditer(text):
        addr = match.group("addr")
        if addr not in seen:
            seen.add(addr)
            ordered.append(addr)
    return ordered


def _wildcard_to_placeholder(addr: str) -> str:
    """Map python-osc '*' path segments to Instrument Control {N} form.

    digital-beacon registers ``/digital/harmonic/*/gain``; the manifest uses
    ``/digital/harmonic/{N}/gain``.
    """

    parts = addr.split("/")
    out: list[str] = []
    for part in parts:
        if part == "*":
            out.append("{N}")
        else:
            out.append(part)
    return "/".join(out)


def manifest_covered_addresses(manifest: dict) -> set[str]:
    """Collect every OSC address/pattern declared in the manifest."""

    covered: set[str] = set()
    for capability in manifest.get("capabilities", []):
        pattern = capability.get("address_pattern")
        if isinstance(pattern, str):
            covered.add(pattern)

    slave = manifest.get("slave_broadcasts") or {}
    for entry in slave.get("addresses", []):
        pattern = entry.get("address_pattern")
        if isinstance(pattern, str):
            covered.add(pattern)

    # Also accept patterns written as a flat list anywhere under address_table.
    for key in ("address_table", "osc_addresses", "addresses"):
        block = manifest.get(key)
        if isinstance(block, dict):
            covered.update(k for k in block if isinstance(k, str) and k.startswith("/"))
        elif isinstance(block, list):
            for item in block:
                if isinstance(item, str) and item.startswith("/"):
                    covered.add(item)
                elif isinstance(item, dict):
                    p = item.get("address_pattern") or item.get("address")
                    if isinstance(p, str):
                        covered.add(p)

    return covered


class ShaperContractTests(unittest.TestCase):
    def test_manifest_validates(self) -> None:
        self.assertTrue(CONTRACT_PATH.is_file(), f"missing {CONTRACT_PATH}")
        manifest = load_manifest(CONTRACT_PATH)
        self.assertIs(validate_manifest(manifest), manifest)
        self.assertEqual(manifest["contract_type"], "instrument_control")
        self.assertEqual(manifest["namespace"], "/digital")
        self.assertEqual(manifest["instrument"]["instrument_id"], "shaper")

    def test_contract_id_matches_golden_sidecar(self) -> None:
        self.assertTrue(GOLDEN_PATH.is_file(), f"missing {GOLDEN_PATH}")
        manifest = load_manifest(CONTRACT_PATH)
        computed = contract_id_from_manifest(manifest)
        golden = GOLDEN_PATH.read_text(encoding="ascii").strip()
        self.assertEqual(computed, golden)
        self.assertEqual(check_golden_sidecar(manifest, GOLDEN_PATH), golden)

    def test_canonical_round_trip_preserves_golden_identity(self) -> None:
        manifest = load_manifest(CONTRACT_PATH)
        encoded = encode_manifest(manifest)
        decoded = decode_manifest(encoded)

        self.assertEqual(decoded, manifest)
        self.assertEqual(encode_manifest(decoded), encoded)
        self.assertEqual(
            contract_id_from_manifest(decoded),
            check_golden_sidecar(decoded, GOLDEN_PATH),
        )

    def test_voice_model_alias_enabled(self) -> None:
        manifest = load_manifest(CONTRACT_PATH)
        alias = manifest["voice_model_alias"]
        self.assertTrue(alias["enabled"])
        self.assertEqual(alias["voice_bounds"], [1, 32])
        self.assertEqual(
            set(alias["mapping"]),
            {"gain", "pan", "phase"},
        )
        patterns = {c["address_pattern"] for c in manifest["capabilities"]}
        for logical, address in alias["mapping"].items():
            self.assertIn(address, patterns, msg=logical)

    def test_handshake_documents_absent_implementation(self) -> None:
        manifest = load_manifest(CONTRACT_PATH)
        handshake = manifest["handshake"]
        status = str(handshake.get("status", "")).lower()
        self.assertIn("none today", status)
        self.assertFalse(handshake.get("implemented", True))

    def test_state_sync_documents_websocket_and_planned_osc(self) -> None:
        manifest = load_manifest(CONTRACT_PATH)
        state_sync = manifest["state_sync"]
        current = state_sync.get("current_mechanism") or {}
        self.assertEqual(current.get("path"), "/ws")
        planned = state_sync.get("planned") or {}
        self.assertEqual(planned.get("status"), "planned")
        self.assertEqual(planned.get("osc_dump_address"), "/shaper/state")

    def test_every_osc_receiver_address_is_covered(self) -> None:
        if not OSC_RECEIVER_PATH.is_file():
            self.fail(
                f"digital-beacon osc_receiver.py not found at {OSC_RECEIVER_PATH}. "
                "Set DIGITAL_BEACON_OSC_RECEIVER to the absolute path."
            )
        addresses = extract_osc_addresses(OSC_RECEIVER_PATH)
        self.assertGreaterEqual(
            len(addresses),
            10,
            f"expected dual-listener address table, got {addresses!r}",
        )

        manifest = load_manifest(CONTRACT_PATH)
        covered = manifest_covered_addresses(manifest)

        missing: list[str] = []
        for addr in addresses:
            candidates = {addr, _wildcard_to_placeholder(addr)}
            if not candidates.intersection(covered):
                missing.append(addr)

        self.assertEqual(
            missing,
            [],
            f"OSC addresses from osc_receiver.py not covered by manifest: {missing}; "
            f"covered={sorted(covered)}; extracted={addresses}",
        )

    def test_extractor_finds_both_listeners(self) -> None:
        if not OSC_RECEIVER_PATH.is_file():
            self.skipTest(f"osc_receiver.py not at {OSC_RECEIVER_PATH}")
        addresses = extract_osc_addresses(OSC_RECEIVER_PATH)
        digital = [a for a in addresses if a.startswith("/digital")]
        beacon = [a for a in addresses if a.startswith("/beacon")]
        self.assertEqual(len(digital), 5, digital)
        self.assertEqual(len(beacon), 6, beacon)


if __name__ == "__main__":
    unittest.main()
