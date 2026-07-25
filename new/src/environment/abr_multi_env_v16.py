"""
Per-chunk (non-monotone) VMAF environment, V16.

V16 == V14 dynamics and corrected reward scale, but the VMAF a rung earns is
looked up PER CHUNK from a non-monotone, multi-resolution ladder
(``vmaf_perchunk_multires.csv`` produced by data/build_multires_vmaf.py) instead
of the single session-average value per bitrate.

Why this is the environment that can validate the original hero claim
--------------------------------------------------------------------
On the single-resolution session ladder, VMAF is monotone in the bitrate index,
so a VMAF-aware shield provably equals an index shield (confirmed: exactly zero
gain at every realistic downgrade depth). The perceptual ranking only becomes a
live design choice when the ladder is non-monotone per chunk, which happens with
a real multi-resolution ladder (resolution/quality crossovers). V16 wires that
non-monotone ladder into BOTH:
  * the reward   (fidelity = the chosen rung's VMAF on THIS chunk), and
  * the shield   (safe_adjust_action reads env.current_vmaf_scores, which we keep
                  pointed at the current chunk's ladder), so VMAF-aware vs.
                  highest-index selection can actually diverge.

The buffer/rebuffering dynamics are byte-driven (bitrate x VBR profile) and are
therefore IDENTICAL to V14; only the VMAF mapping changes. This isolates the
per-chunk perceptual effect.

Invariant maintained: ``current_vmaf_scores`` / ``current_vmaf_norm`` always
reflect the ladder of the current ``chunk_idx`` (the chunk about to be
downloaded / observed), which is exactly what both the reward step and the shield
read.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .abr_multi_env_v14 import ABREnv as _ABREnvV14

_DEFAULT_PERCHUNK = Path(__file__).resolve().parents[2] / "data" / "vmaf_scores" / "vmaf_perchunk_multires.csv"


class ABREnv(_ABREnvV14):
    """V14 environment with a per-chunk, non-monotone VMAF ladder."""

    def __init__(self, *args, vmaf_perchunk_path: str | None = None, **kwargs):
        self._perchunk_path = Path(vmaf_perchunk_path) if vmaf_perchunk_path else _DEFAULT_PERCHUNK
        self._perchunk: dict[str, list[dict]] = {}
        super().__init__(*args, **kwargs)
        self._load_perchunk()

    def _load_perchunk(self):
        """Load {video: [ {bitrate:int -> vmaf:float}, ... per chunk ]}."""
        self._perchunk = {}
        if not self._perchunk_path.exists():
            print(f"[v16][WARN] per-chunk ladder not found at {self._perchunk_path}; "
                  f"falling back to the monotone session ladder (VMAF-aware will be inert).")
            return
        df = pd.read_csv(self._perchunk_path)
        for vid, g in df.groupby("video"):
            n_chunks = int(g["chunk"].max()) + 1
            table: list[dict] = [dict() for _ in range(n_chunks)]
            for _, r in g.iterrows():
                table[int(r["chunk"])][int(r["bitrate_kbps"])] = float(r["vmaf"])
            self._perchunk[str(vid)] = table

    def _apply_chunk_vmaf(self):
        """Point current_vmaf_scores/norm at the current chunk's ladder."""
        table = self._perchunk.get(self.current_video_name)
        if not table:
            return  # keep the session-average ladder set by the base env
        c = min(int(self.chunk_idx), len(table) - 1)
        scores = table[c]
        if not scores:
            return
        self.current_vmaf_scores = dict(scores)
        self.current_vmaf_norm = {br: v / 100.0 for br, v in scores.items()}

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._apply_chunk_vmaf()
        self.last_vmaf = float(self.current_vmaf_scores.get(int(self.BITRATE_LEVELS[0]), 35.0))
        return self._get_observation(), info

    def step(self, action):
        # At entry, current_vmaf_scores reflects chunk_idx = c (the chunk being
        # downloaded), so the base reward uses the correct per-chunk VMAF and the
        # shield (called before this in the wrapper) also saw chunk c.
        obs, reward, terminated, truncated, info = super().step(action)
        # chunk_idx is now c+1; refresh the ladder and rebuild the observation so
        # the next state exposes the upcoming chunk's (non-monotone) ladder.
        self._apply_chunk_vmaf()
        obs = self._get_observation()
        return obs, reward, terminated, truncated, info


__all__ = ["ABREnv"]
