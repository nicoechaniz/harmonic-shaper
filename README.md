# harmonic-shaper

The canonical additive harmonic synthesizer of the Harmonic Beacon ecosystem:
32 voices + waveshaper + per-voice LFO + sidechain.

This package is the reconciled standalone extraction of the evolved Shaper in
`digital-beacon`. The older `NaturalHarmony/harmonic_shaper` implementation is
historical; its Minilab3 controller was recovered here because it had been lost
from the fork.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

PortAudio must be available on the host for real-time output through
`sounddevice`.

## Run

```bash
python -m harmonic_shaper
# or, after installation:
harmonic-shaper
```

The default process starts the audio engine, native OSC listener, both MIDI
controllers when present, and the FastAPI state service. Useful flags:

```bash
python -m harmonic_shaper --slave       # opt in to NH /beacon/* broadcasts
python -m harmonic_shaper --no-audio    # headless control/API process
python -m harmonic_shaper --no-midi
python -m harmonic_shaper --no-api
python -m harmonic_shaper --help
```

Default bindings:

- UDP `:9002`: current v1 wire protocol under `/digital/*`.
- UDP `:9001`: optional `/beacon/*` slave input, only with `--slave`.
- HTTP `127.0.0.1:8080`: `GET /api/state`, `POST /api/shaper/*`, and
  WebSocket `/ws`.

`/shaper/*` is the planned native namespace. It is intentionally not mapped on
the wire yet because renaming `/digital/*` requires a contract version bump.

## Pure reference renderer

The extracted offline reference is also installable:

```bash
harmonic-shaper-synth-pure input.wav --out output.wav
```

It retains the fork's voice-analysis and NumPy rendering path for the clipping
work tracked after this extraction.

## Test

```bash
pytest -q
```

See [the extraction report](docs/T2.2_EXTRACTION_REPORT.md) for the module map,
fork reconciliation, dependency audit, and clipping notes.

## License

MIT — see [LICENSE](LICENSE).
