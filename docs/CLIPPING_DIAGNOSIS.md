# Shaper clipping diagnosis

Date: 2026-07-18

## Reproduce headlessly

No audio device or `sounddevice` installation is used by this characterization:

```bash
pytest tests/test_audio_smoke.py -q -s
```

The prepared reference inputs use a 110 Hz fundamental, 44.1 kHz sample rate,
512-sample blocks, zero spectral tilt, no aperiodic noise, unity voice gains,
and zero starting phases. The transition history is 12 blocks with 32 voices,
4 blocks with one voice, 12 blocks with 32 voices, then 20 blocks with one
voice. Live voices use 10 ms attack and 30 ms release. Master gain is one;
sidechain, LFO, waveshaping, and pan modulation are disabled. Equal-power pan
is centered.

`at/over full-scale` counts scalar sample values whose absolute value is at
least 1.0. Live counts cover both channels, while the pure renderer is mono.
The pre-limiter mix is intentionally measured separately from the bounded
output so overload is not hidden.

## Measurements

| Scenario | Stage | Frames | Peak | RMS | At/over full-scale |
|---|---:|---:|---:|---:|---:|
| Pure, one voice | pre-limiter | 8,192 | 0.999999746 | 0.695299409 | 0 |
| Pure, one voice | output | 8,192 | 0.742715941 | 0.555432348 | 0 |
| Pure, 32 voices | pre-limiter | 8,192 | 4.162159388 | 0.696503570 | 499 |
| Pure, 32 voices | output | 8,192 | 0.949696109 | 0.289968749 | 0 |
| Pure, rapid transition | pre-limiter | 24,576 | 4.318684358 | 0.675194076 | 795 |
| Pure, rapid transition | output | 24,576 | 0.949781232 | 0.416696829 | 0 |
| Pure, first release window | pre-limiter | 2,048 | 3.327852936 | 0.535621632 | 30 |
| Live callback, rapid transition | pre-limiter | 24,576 stereo | 2.976611614 | 0.474958690 | 1,038 |
| Live callback, rapid transition | output | 24,576 stereo | 0.946342468 | 0.330714676 | 0 |

The live pre-limiter comparison must account for centered equal-power pan:
each channel is the mono reference times `1/sqrt(2)`. On that basis the
reference transition predicts peak 3.053770995 and RMS 0.477434308. The live
measurements are 2.5% lower in peak and 0.5% lower in RMS. A steady section away
from envelope edges matches sample-for-sample within `1e-6` after accounting
for the renderers' one-sample oscillator convention.

These are numerical signal measurements only. No audible-audio conclusion is
claimed.

## Diagnosis and correction

The main overload is not a missing active-voice count. Both renderers use
`1/sqrt(N)` normalization, and the live engine includes release tails in `N`.
That normalization keeps the dense case near the one-voice RMS, as the table
shows, but it cannot bound the peak of correlated harmonic sines. With aligned
phases, worst-case peak growth can remain proportional to `sqrt(N)`. The
existing output limiter in `AudioEngine._audio_callback` is therefore the
stage that declares and enforces the ±0.95 output range.

One exact reference discrepancy was found and corrected in
`synth_pure.synthesize_prepared`: when the active mask changed, a releasing
voice's envelope remained nonzero but its gain was immediately read from the
new inactive frame (`-120 dB`). That erased the release contribution even
though the live callback retains the voice's last parameters. The pure renderer
now holds the last active gain until the release envelope reaches zero. Before
the correction, the first 32-to-1 release window peaked at approximately 1.0
with zero values at/over full-scale; it now peaks at 3.327852936 with 30, which
tracks the live behavior being modeled.

The live and pure output stages also used slightly different limiter drive
values. Both now call `audio_levels.soft_limit`, which implements the existing
live formula `tanh(mix * 1.05) * 0.95`. `synthesize_prepared(limit_output=True)`
exposes that declared output stage while its default remains the unbounded
pre-limiter mix for analysis and compatibility.

The residual transition difference is explained by an implementation
difference: the pure envelope changes every sample, while the live envelope is
updated once per callback and held for the block. The tests compare the
aggregate result with a 5% tolerance and prove exact agreement in a steady
section. No further correction is justified by this evidence.

Hypothesis: the transition peak being slightly above the first dense peak is
caused by phase relationships after voices release, freeze, and reactivate.
This is not needed to explain the full-scale overload and has not been isolated
as a defect.

## Decision

Keep release-tail-aware `1/sqrt(N)` normalization and the existing bounded
soft limiter. Do not add peak normalization or another arbitrary gain factor:
that would change level behavior without a specified headroom target. The
pre-limiter recording tap remains capable of exceeding full scale by design.
The next decision, if raw recording or downstream headroom becomes a product
requirement, is to specify that target first and then choose either a documented
fixed headroom budget or an explicitly characterized recording limiter.

The Instrument Control v1 manifest and `/digital/*` surface are unaffected.
Their canonical round trip and golden contract identifier remain regression
tested.
