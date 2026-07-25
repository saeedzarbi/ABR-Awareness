"""
Torch-free smoke test for the V15 low-latency regime.

Validates, WITHOUT training a policy, that:
  1. the broadband/harsh traces load and what their throughput distribution is,
  2. shrinking the buffer 30 -> 6 s actually creates rebuffering stress under a
     fixed scripted policy (i.e. the regime change bites), and
  3. the runtime shield actually intervenes and reduces stalls at buffer 6 s.

Run:  py _smoke_v15.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))

from configs.paths import get_paths
from src.environment.abr_multi_env_v15 import ABREnv as EnvV15  # buffer 6, fixed loader
from src.training.safety_shield_v14 import ShieldConfig, safe_adjust_action


class EnvV14(EnvV15):
    """Big-buffer (v14) operating point but with the v15 robust trace loader, so
    the buffer-30-vs-6 comparison isolates ONLY the buffer geometry."""
    BUFFER_MAX = 30.0
    BUFFER_TARGET = 15.0
    B_REF = 8.0

PATHS = get_paths()
TRAIN_VIDEOS = ["bigbuckbunny", "crowd_run", "tearsofsteel_short"]
BROADBAND = str(PATHS["test_traces"])
# The harsh set produced by run_lowlat_v15.sh step [0]; falls back gracefully if
# it has not been generated yet.
HARSH5G = str(Path("data/standardized/test_traces_5g_harsh_v15"))
HAS_HARSH = any(Path(HARSH5G).glob("*.json")) if Path(HARSH5G).exists() else False
N_EP = 40


def trace_stats(trace_dir, label):
    import json
    files = sorted(Path(trace_dir).glob("*.json"))
    files = [f for f in files if f.name != "trace_stats.json"]
    allv = []
    for f in files:
        d = json.loads(f.read_text())
        s = d.get("throughput_kbps")
        if s:
            allv.append(np.asarray(s, dtype=float))
    if not allv:
        print(f"[{label}] NO TRACES at {trace_dir}")
        return
    cat = np.concatenate(allv)
    print(f"[{label}] files={len(files)}  mean={cat.mean():.0f} kbps  "
          f"median={np.median(cat):.0f}  p05={np.percentile(cat,5):.0f}  "
          f"min={cat.min():.0f}  (ladder top=6000)")


def make_env(env_cls, trace_dir, seed):
    return env_cls(
        video_names=TRAIN_VIDEOS,
        trace_dir=trace_dir,
        vmaf_dir=str(PATHS["vmaf_scores"]),
        siti_dir=str(PATHS["content_features"]),
        max_chunks=48, random_seed=seed, use_future=True, use_lyapunov=True,
    )


def policy_fixed(idx):
    return lambda env: idx


def policy_bba(env):
    frac = float(np.clip(env.buffer_level / env.BUFFER_MAX, 0, 1))
    return int(round(frac * (len(env.BITRATE_LEVELS) - 1)))


def run(env_cls, trace_dir, policy, shield_cfg=None, seed=123):
    env = make_env(env_cls, trace_dir, seed)
    tot_rebuf_ratio, tot_stallfree, tot_interv = [], [], []
    for ep in range(N_EP):
        obs, info = env.reset(seed=ep)
        done = False
        chunks = 0
        interv = 0
        while not done and env.chunk_idx < env.max_chunks:
            a = policy(env)
            a = max(0, min(int(a), len(env.BITRATE_LEVELS) - 1))
            if shield_cfg is not None:
                a, was = safe_adjust_action(env, a, shield_cfg)
                interv += int(was)
            obs, _, done, _, info = env.step(int(a))
            chunks += 1
        dur = chunks * env.CHUNK_DURATION
        rebuf = float(info.get("total_rebuffer", 0.0))
        tot_rebuf_ratio.append(rebuf / dur * 100.0 if dur > 0 else 0.0)
        tot_stallfree.append(1.0 if rebuf <= 1e-6 else 0.0)
        tot_interv.append(interv / max(1, chunks) * 100.0)
    return (float(np.mean(tot_rebuf_ratio)), float(np.mean(tot_stallfree)),
            float(np.mean(tot_interv)))


def main():
    print("=" * 70)
    print("V15 LOW-LATENCY SMOKE TEST (no training)")
    print(f"buffer: v14={EnvV14.BUFFER_MAX}s  v15={EnvV15.BUFFER_MAX}s   episodes/cond={N_EP}")
    print("=" * 70)

    print("\n--- 1. Trace throughput distribution ---")
    trace_stats(BROADBAND, "broadband_test")
    trace_stats(str(PATHS["train_traces"]), "broadband_train")
    trace_stats(HARSH5G, "harsh_5g")

    trace_sets = [("broadband", BROADBAND)] + ([("harsh_5g", HARSH5G)] if HAS_HARSH else [])

    print("\n--- 2. Regime bite: same policy, buffer 30 (v14) vs buffer 6 (v15) ---")
    print(f"{'policy':<16}{'traces':<16}{'buf':<5}{'Rebuf%':>9}{'StallFree':>11}")
    for pname, pol in [("always_2850", policy_fixed(4)), ("always_6000", policy_fixed(5)), ("bba", policy_bba)]:
        for tname, tdir in trace_sets:
            for bname, cls in [("30", EnvV14), ("6", EnvV15)]:
                rb, sf, _ = run(cls, tdir, pol)
                print(f"{pname:<16}{tname:<16}{bname:<5}{rb:>9.2f}{sf:>11.2f}")

    print("\n--- 3. Shield activity at buffer 6 (v15): no-shield vs shield(light) ---")
    print(f"{'policy':<16}{'traces':<16}{'cond':<12}{'Rebuf%':>9}{'StallFree':>11}{'Interv%':>9}")
    grid = {
        "off": None,
        "shield_light": ShieldConfig(level="light"),
    }
    for pname, pol in [("always_6000", policy_fixed(5)), ("bba", policy_bba)]:
        for tname, tdir in trace_sets:
            for cond, cfg in grid.items():
                rb, sf, iv = run(EnvV15, tdir, pol, shield_cfg=cfg)
                print(f"{pname:<16}{tname:<16}{cond:<12}{rb:>9.2f}{sf:>11.2f}{iv:>9.2f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
