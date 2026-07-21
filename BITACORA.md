# BITACORA

- 2026-07-18: repo scaffolded.
- 2026-07-18: T2.2 extracted and reconciled the standalone Shaper from digital-beacon; NaturalHarmony Minilab3 support recovered; headless and contract tests added.
- 2026-07-19: Added source-owned `harmonic_envelope` to the native `/digital/harmonic/{N}` contract. Positive envelopes activate the exact `f1*n` partial; zero releases only the envelope source, so body-driven releases cannot turn off keyboard-owned voices. The contract ID is `763efea4f567f6c9396b13b7af33c540`.
- 2026-07-19: Calibrated the final physical high-transpose key as MIDI 108 and made it a configurable panic key. It clears all active Shaper voices through the shared store; the keyboard-specific state is also reset. Shaper verification: 73 tests passed.
- 2026-07-19: Live camera integration reached the audible JACK/R24 Shaper through harmonic-weaver. User confirmed that movement produced a musical audible response. The later HarMoCAP CUDA/ReID failure did not originate in Shaper; runtime stability remains an external open issue.
- 2026-07-21: Arpeggiator H=0 (P2-S1). Voice band `-20000-n`, settled params via `settle_beats`, OSC `/digital/arp/0/*`, `envelope_profile` pluck/perc/pad, `advance_arp` in the audio callback. Contract ID `3c5bb7a6c6c8190c8d9da8098be967af`. 95 tests passed.
- 2026-07-21: H=1 + foot percussion (P3-S1). Second arp hand voice band `-21001..-21032` (same pluck/zero-cross pattern); perc pool `-30000..-30007` with OSC `/digital/perc/*`, `advance_perc`, envelope `perc` (~80 ms), separate melodic 1/√N norm so hits do not dilute arp. Panic clears both hands + perc pool. Contract ID `4309137a803a1df793b0d814454fef11`. 105 tests passed.
