"""Runtime configuration for the standalone Harmonic Shaper.

``/shaper`` is the planned native namespace for a future contract bump.  The
current Instrument Control v1 wire contract remains ``/digital`` on UDP 9002;
changing those addresses here would break existing clients.
"""

# Audio
AUDIO_SAMPLE_RATE = 44_100
AUDIO_BLOCK_SIZE = 256
AUDIO_DEVICE = None

# Harmonic lattice
DEFAULT_F1 = 40.40
N_BANDS = 32
F1_MIN = 20.0
F1_MAX = 200.0
MAX_VOICES = 32

# Native MIDI-note harmonic source (standalone keyboard playability)
# Anchor MIDI note that represents f₁ (C1 = 24), matching NaturalHarmony.
DEFAULT_ANCHOR_MIDI = 24
# Default standalone mode: play generic MIDI keyboards without NH beacon.
NATIVE_MIDI_SOURCE_ENABLED = True
# Ports whose names contain these substrings are left to dedicated controllers.
NATIVE_MIDI_EXCLUDE_PATTERNS = ("Launchpad", "Minilab")
# Velocity → voice gain bounds (linear map, then clamp)
NATIVE_MIDI_VELOCITY_GAIN_MIN = 0.0
NATIVE_MIDI_VELOCITY_GAIN_MAX = 1.0

# OSC
PLANNED_OSC_NAMESPACE = "/shaper"
WIRE_OSC_NAMESPACE = "/digital"
SLAVE_OSC_NAMESPACE = "/beacon"
SHAPER_OSC_PORT = 9002
BEACON_BROADCAST_PORT = 9001
OSC_HOST = "0.0.0.0"

# HTTP / WebSocket state API
API_HOST = "127.0.0.1"
API_PORT = 8080

# Voice defaults
DEFAULT_VOICE_GAIN = 0.6
DEFAULT_VOICE_PAN = 0.0
DEFAULT_VOICE_PHASE_DEG = 0.0
DEFAULT_VOICE_ATTACK_S = 0.01
DEFAULT_VOICE_RELEASE_S = 0.15
DEFAULT_VOICE_SHAPE = 0.0

# Sidechain: -1=ducking, 0=off, +1=follow beacon level
DEFAULT_SIDECHAIN_AMOUNT = 0.0

# Per-voice LFO, synchronized to observed slave strums when slave mode is on
DEFAULT_LFO_RATE_DIVISOR = 1
DEFAULT_LFO_WAVEFORM = "sine"
DEFAULT_LFO_AMOUNT = 0.0
DEFAULT_LFO_GAIN = 0.0
DEFAULT_LFO_PAN = 0.0
DEFAULT_LFO_PHASE = 0.0
STRUM_WINDOW = 8
DEFAULT_STRUM_PERIOD_S = 0.5

# Mix
DEFAULT_SHAPER_MASTER = 0.8

# Launchpad Mini
LAUNCHPAD_PORT_PATTERN = "Launchpad"
LAUNCHPAD_PADS_X = 8
LAUNCHPAD_PADS_Y = 8
SPLIT_MODE_ENABLED_BY_DEFAULT = True
SPLIT_MODE_TOGGLE_CC = 104
PAD_FEEDBACK_COLOR_ON = 60
PAD_FEEDBACK_COLOR_TOGGLE_ON = 21

# Minilab3 mappings recovered from the NaturalHarmony original
MINILAB_PORT_PATTERN = "Minilab"
MINILAB_SLIDER_CCS = [14, 15, 30, 31]
MINILAB_PAN_CCS = [86, 87, 89, 90]
MINILAB_PHASE_CCS = [110, 111, 116, 117]
MINILAB_PANIC_PAD = 39

LOG_LEVEL = "INFO"
