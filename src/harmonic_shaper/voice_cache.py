"""VoiceCache — RAM + .npz persistence for prepared voice-analysis dicts.

The voice-comparison pipeline (build_voice_compare_v3) repeats a fair amount
of work per WAV: librosa resample, F0 estimation via synth_pure.analyze,
RMS balancing, spectrogram + PNG generation. Each run scans every WAV
under ~/Music/voice-analysis/ and re-does all of it.

VoiceCache short-circuits that work for files we have already seen. The
pipeline flow becomes:

    cache = VoiceCache()
    prepared = cache.get(wav_path)
    if prepared is None:
        prepared = compute(wav_path)   # the heavy work
        cache.store(wav_path, prepared)
    # use prepared...

Storage format
--------------
On ``store`` we write a single ``.npz`` per source WAV:

    <cache_dir>/<wav_path.stem>_analysis.npz

Arrays are passed through to ``np.savez`` unchanged. Scalars from the
prepared dict (``sr``, ``duration`` and any other plain Python numbers)
are stored as 0-d numpy arrays. Two extra 0-d arrays carry cache
validation metadata so we can detect when the source WAV has changed
on disk and needs recomputation:

    _meta_src_mtime  — float, source WAV's mtime in seconds since epoch
    _meta_src_size   — int64, source WAV's size in bytes

On ``get`` we compare the current (mtime, size) of the source WAV
against the values stored in the .npz. If either differs, we treat the
cache as stale and return None so the caller recomputes. mtime alone
is not enough (touching a file does not change size), size alone is
not enough (an in-place rewrite can keep size constant), so both are
checked.

The on-disk .npz is the authoritative store; the RAM cache is a hot
front-end keyed by ``str(wav_path.resolve())`` so the same WAV loaded
under two different relative paths hits the same RAM entry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np


# Default base directory for voice-analysis artifacts (matches the rest of
# the voice pipeline under tools/build_voice_compare_v3.py).
VOICE_DIR = Path.home() / "Music" / "voice-analysis"


def cache_key_for(wav_path: Path) -> str:
    """Stable RAM-cache key for a WAV path.

    Resolves symlinks and relative components so the same physical file
    hit via different paths (e.g. ``./foo.wav`` vs ``/abs/foo.wav``) maps
    to the same cache entry.
    """
    return str(Path(wav_path).resolve())


class VoiceCache:
    """Two-tier cache (RAM + .npz on disk) for prepared voice-analysis dicts.

    Lookup order in :meth:`get`:

        1. RAM cache keyed by resolved WAV path
        2. .npz file under ``cache_dir`` (if mtime + size still match)
        3. miss — return None and let the caller compute

    :meth:`store` writes the .npz and refreshes the RAM entry.
    :meth:`invalidate` removes both.
    """

    # Suffix for the on-disk cache file. Public so callers (tests, tooling)
    # can find existing caches without hard-coding the pattern.
    CACHE_SUFFIX = "_analysis.npz"

    # Keys inside the .npz that carry cache validation metadata (not part
    # of the prepared-analysis payload). Prefixed with an underscore to
    # keep them out of any dict-style iteration over the prepared data.
    META_MTIME_KEY = "_meta_src_mtime"
    META_SIZE_KEY = "_meta_src_size"

    def __init__(self, cache_dir: Path | str = VOICE_DIR / "cache") -> None:
        self.cache_dir = Path(cache_dir)
        # parents=True so nested paths work; exist_ok so first run is clean.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ram_cache: dict[str, dict[str, Any]] = {}

    # ----- path helpers -------------------------------------------------

    def cache_path_for(self, wav_path: Path) -> Path:
        """Where the .npz for this WAV lives (may or may not exist yet)."""
        return self.cache_dir / f"{Path(wav_path).stem}{self.CACHE_SUFFIX}"

    def __contains__(self, wav_path: Path) -> bool:
        """Cheap existence check: RAM entry OR .npz on disk (no mtime/size
        validation here — use :meth:`get` for the authoritative test)."""
        key = cache_key_for(wav_path)
        if key in self.ram_cache:
            return True
        return self.cache_path_for(wav_path).exists()

    # ----- core API -----------------------------------------------------

    def get(self, wav_path: Path) -> dict[str, Any] | None:
        """Return the prepared-analysis dict for ``wav_path``, or None if
        no fresh cache entry is available.

        "Fresh" means either:

        * the RAM cache holds a valid entry, OR
        * the on-disk .npz exists AND the source WAV's current
          (mtime, size) match the metadata stored in the .npz.

        Otherwise the entry is treated as stale and we drop it from RAM
        (if present) so the next ``store`` writes a clean copy.
        """
        wav_path = Path(wav_path)
        key = cache_key_for(wav_path)

        # 1. RAM cache: still authoritative as long as the source file
        #    has not been touched since we stored it. We re-validate
        #    against disk metadata every time to catch edits even when
        #    the RAM entry is "fresh" — cheap (two stat() calls).
        ram_entry = self.ram_cache.get(key)
        if ram_entry is not None:
            if self._source_matches(ram_entry, wav_path):
                return ram_entry
            # Source changed since we cached — drop the stale RAM entry.
            self.ram_cache.pop(key, None)

        # 2. Disk cache.
        npz_path = self.cache_path_for(wav_path)
        if not npz_path.exists():
            return None

        try:
            with np.load(npz_path, allow_pickle=False) as data:
                stored_mtime = float(data[self.META_MTIME_KEY])
                stored_size = int(data[self.META_SIZE_KEY])

                try:
                    stat = wav_path.stat()
                except OSError:
                    # Source WAV is gone — treat the cache as stale.
                    return None

                if stat.st_mtime != stored_mtime or stat.st_size != stored_size:
                    return None

                prepared = _unpack_payload(data)
        except (OSError, KeyError, ValueError, EOFError):
            # Corrupt or unreadable cache file — drop it and miss.
            try:
                npz_path.unlink()
            except OSError:
                pass
            return None

        # Promote the disk hit into RAM for the next call.
        self.ram_cache[key] = prepared
        return prepared

    def store(self, wav_path: Path, prepared: dict[str, Any]) -> None:
        """Persist ``prepared`` for ``wav_path`` to RAM and .npz.

        Overwrites any existing cache file for this WAV. Source-file
        metadata (mtime + size) is captured at store time so the next
        :meth:`get` can detect when the WAV has been modified.
        """
        wav_path = Path(wav_path)
        key = cache_key_for(wav_path)

        # Snapshot source metadata. If the WAV is missing we still want
        # to store the prepared data (the caller may have built it from
        # an in-memory signal) — fall back to zeros so the cache stays
        # self-consistent but any later get() will treat it as stale.
        try:
            stat = wav_path.stat()
            src_mtime = float(stat.st_mtime)
            src_size = int(stat.st_size)
        except OSError:
            src_mtime = 0.0
            src_size = 0

        # RAM copy: keep the caller's dict untouched so they can keep
        # mutating it without poisoning the cache.
        self.ram_cache[key] = dict(prepared)

        npz_path = self.cache_path_for(wav_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, Any] = {
            self.META_MTIME_KEY: np.float64(src_mtime),
            self.META_SIZE_KEY: np.int64(src_size),
        }
        for name, value in prepared.items():
            arrays[name] = _to_array(value, name)

        # Atomic-ish write: savez to a sibling temp file then rename, so a
        # crash mid-write cannot leave a half-written .npz that subsequent
        # get() calls would have to clean up. np.savez silently appends
        # ".npz" when the target path does not already end in ".npz", so
        # the temp filename has to keep the .npz extension — we just add
        # a ".tmp" segment before it.
        tmp_path = npz_path.with_name(npz_path.stem + ".tmp.npz")
        np.savez(tmp_path, **arrays)  # type: ignore[arg-type]  # scalars auto-coerced
        os.replace(tmp_path, npz_path)

    def invalidate(self, wav_path: Path) -> None:
        """Drop the WAV from RAM and delete its .npz if present.

        Safe to call when nothing is cached — both operations are no-ops
        in that case. Used when a caller knows the prepared dict is no
        longer trustworthy (e.g. the source WAV was overwritten and the
        pipeline wants to force a recompute on the next pass).
        """
        wav_path = Path(wav_path)
        key = cache_key_for(wav_path)
        self.ram_cache.pop(key, None)
        npz_path = self.cache_path_for(wav_path)
        try:
            npz_path.unlink()
        except FileNotFoundError:
            pass

    # ----- internals ----------------------------------------------------

    def _source_matches(self, entry: dict[str, Any], wav_path: Path) -> bool:
        """True iff the cached entry's source metadata matches the WAV on disk.

        We stash the snapshot in the RAM entry itself on store so this
        check stays a single stat() call instead of round-tripping
        through the .npz file.
        """
        try:
            stat = wav_path.stat()
        except OSError:
            return False
        return (
            stat.st_mtime == entry.get("_meta_src_mtime")
            and stat.st_size == entry.get("_meta_src_size")
        )


# ---------------------------------------------------------------------------
# Array packing / unpacking helpers
# ---------------------------------------------------------------------------
#
# The prepared dict mixes numpy arrays (audio buffers, F0 traces, spectrograms)
# with Python scalars (sample rate, duration). np.savez accepts numpy arrays
# only — Python ints/floats would be silently dropped. We coerce scalars to
# 0-d arrays on the way in and reconstruct them on the way out so the
# round-trip is transparent to callers.


def _to_array(value: Any, name: str) -> np.ndarray:
    """Coerce a prepared-dict value into something np.savez can persist.

    - numpy arrays pass through (we copy to make the on-disk file independent
      of later in-place edits to the caller's buffer).
    - Python scalars (int, float, bool, str) become 0-d arrays of the
      appropriate dtype.
    - Anything else raises TypeError so a bad prepared dict fails loudly
      at store time rather than silently losing data on reload.
    """
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, bool):
        # bool must come BEFORE int (bool is a subclass of int in Python).
        return cast(np.ndarray, np.bool_(value))
    if isinstance(value, (int, np.integer)):
        return cast(np.ndarray, np.int64(value))
    if isinstance(value, (float, np.floating)):
        return cast(np.ndarray, np.float64(value))
    if isinstance(value, str):
        # Stored as a 0-d unicode array; round-trips back to str on load.
        return np.array(value, dtype=object)
    raise TypeError(
        f"VoiceCache.store: unsupported type {type(value).__name__} "
        f"for prepared[{name!r}; coerce to ndarray or scalar before storing."
    )


def _from_array(arr: np.ndarray) -> Any:
    """Inverse of :func:`_to_array` — recover the original Python value.

    0-d arrays become their scalar equivalent (``arr.item()``); higher-dim
    arrays pass through. Object-dtype 0-d arrays are unpacked to ``str``
    so the round-trip preserves the type the caller originally stored.
    """
    if arr.ndim == 0:
        item = arr.item()
        # np.array("foo", dtype=object) round-trips to a 0-d object array
        # whose .item() is the original str — leave it as-is.
        return item
    return arr


def _unpack_payload(npz_data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    """Reconstruct the prepared dict from a loaded .npz, skipping meta keys."""
    payload: dict[str, Any] = {}
    for name in npz_data.files:
        if name == VoiceCache.META_MTIME_KEY or name == VoiceCache.META_SIZE_KEY:
            continue
        payload[name] = _from_array(npz_data[name])
    return payload
