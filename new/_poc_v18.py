"""Torch-free proof-of-concept for the Certified Perceptual Shield (V18).

Runs a BITRATE-GREEDY policy (always the top rung -- the adversarial/worst case a
shield must protect, and exactly what a bitrate-reward agent like Pensieve tends
to do) on the low-latency (v15) env, comparing three arms:
  RAW          : no shield
  SAFETY-ONLY  : conformal feasibility projection, banking OFF
  CERTIFIED    : conformal feasibility + VMAF-knee bandwidth banking (full)

We want to see, for CERTIFIED vs RAW:
  * rebuffering DOWN, stall-free UP, buffer UP   (banking helps safety), while
  * mean VMAF within ~epsilon                     (perceptually ~lossless), and
  * empirical conformal coverage ~ 1 - alpha      (the certificate holds).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
from configs.paths import get_paths
from src.environment.abr_multi_env_v14 import ABREnv as _V14
from src.training.certified_perceptual_shield import (
    CertifiedPerceptualShieldWrapper, CPShieldConfig, ConformalConfig)

P = get_paths()
EPISODES = 80
EPS = 1.0
ALPHA = 0.10

# A "moderate" operating point: buffer large enough that high (saturated) rungs
# are usually FEASIBLE (so pure safety rarely fires), yet small enough that an
# occasional dip can stall -- exactly where perceptual banking should convert
# perceptually-worthless top-rung bytes into a protective buffer margin.
BUFFERS = [8.0, 12.0, 20.0]


def _synth_traces(kind: str):
    """High-throughput 5G-like traces where the SATURATED rungs are feasible
    (so perceptual banking has headroom), with periodic dips."""
    import numpy as _np
    rng = _np.random.default_rng(7)
    traces = []
    for _ in range(20):
        if kind == "steady":
            tp = _np.full(1000, 9000.0)
        else:  # "dips": ~9 Mbps with periodic 4s dips to ~700 kbps
            tp = _np.full(1000, 9000.0)
            for start in range(24, 1000, 40):
                tp[start:start + 4] = 700.0
        tp *= rng.uniform(0.85, 1.15, size=tp.shape)  # mild jitter
        traces.append({"throughput_kbps": tp.tolist()})
    return traces


def make_env(buffer_max: float, trace_kind: str | None = None):
    class ModEnv(_V14):
        BUFFER_MAX = buffer_max
        BUFFER_TARGET = buffer_max / 2.0
        B_REF = max(2.0, buffer_max / 3.0)
    env = ModEnv(
        video_names=["bigbuckbunny", "crowd_run", "tearsofsteel_short"],
        trace_dir=str(P["train_traces"]),
        vmaf_dir=str(P["vmaf_scores"]),
        siti_dir=str(P["content_features"]),
        max_chunks=48, random_seed=0, use_lyapunov=True, use_future=True,
    )
    if trace_kind:
        env.traces = _synth_traces(trace_kind)
    return env


def run(arm: str, buffer_max: float, policy: str = "greedy", trace_kind: str | None = None):
    wrapped = arm != "raw"
    if not wrapped:
        env = make_env(buffer_max, trace_kind)
    else:
        cfg = CPShieldConfig(
            enabled=True,
            enable_banking=(arm == "certified"),
            epsilon_vmaf=EPS,
            enable_conformal=True,
            conformal=ConformalConfig(alpha=ALPHA, window=200, k_predict=5),
            safety_margin=0.5, min_buffer=0.3,
        )
        env = CertifiedPerceptualShieldWrapper(make_env(buffer_max, trace_kind), cfg)

    n = len(env.BITRATE_LEVELS)
    reb, vmaf, stallfree, buffers, bitrates = [], [], [], [], []
    banked_bits, interv, cover = 0.0, 0, []
    for ep in range(EPISODES):
        obs, info = env.reset(seed=1000 + ep)
        ep_reb, ep_vmaf, ep_stall, ep_br = 0.0, [], 0, []
        done = False
        while not done:
            action = n - 1 if policy == "greedy" else n - 2
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reb += float(info.get("rebuffer", 0.0))
            ep_vmaf.append(float(info.get("vmaf", 0.0)))
            ep_br.append(float(info.get("bitrate", 0.0)))
            ep_stall += int(float(info.get("rebuffer", 0.0)) > 1e-6)
            buffers.append(float(info.get("buffer", 0.0)))
            if wrapped:
                banked_bits += float(info.get("banked_bits", 0.0))
                interv += int(info.get("shield_intervened", 0))
        reb.append(ep_reb)
        vmaf.append(float(np.mean(ep_vmaf)))
        bitrates.append(float(np.mean(ep_br)))
        stallfree.append(1.0 if ep_stall == 0 else 0.0)
        if wrapped:
            c = info.get("conformal_coverage", float("nan"))
            if not np.isnan(c):
                cover.append(float(c))
    return {
        "arm": arm,
        "reb": float(np.mean(reb)),
        "reb_p95": float(np.percentile(reb, 95)),
        "stallfree": float(np.mean(stallfree)),
        "vmaf": float(np.mean(vmaf)),
        "bitrate": float(np.mean(bitrates)),
        "buffer": float(np.mean(buffers)),
        "interv": (interv / max(len(reb) * 48, 1)) if wrapped else 0.0,
        "bankMB": banked_bits / 8e6 if wrapped else 0.0,
        "cover": float(np.mean(cover)) if cover else float("nan"),
    }


def _report(title, rows):
    print(f"===== {title} =====")
    print(f"{'arm':>10}{'reb':>9}{'reb_p95':>9}{'stallfree':>10}{'vmaf':>8}"
          f"{'bitrate':>9}{'buffer':>8}{'interv':>8}{'cover':>7}")
    for r in rows:
        print(f"{r['arm']:>10}{r['reb']:>9.3f}{r['reb_p95']:>9.3f}{r['stallfree']:>10.3f}"
              f"{r['vmaf']:>8.2f}{r['bitrate']:>9.0f}{r['buffer']:>8.2f}{r['interv']:>8.3f}{r['cover']:>7.3f}")
    s, c = rows[1], rows[2]
    bw = 100 * (c["bitrate"] - s["bitrate"]) / max(s["bitrate"], 1e-9)
    print(f"  BANKING (certified - safety): reb {c['reb']-s['reb']:+.3f}  "
          f"stallfree {c['stallfree']-s['stallfree']:+.3f}  bitrate {bw:+.1f}%  "
          f"VMAF {c['vmaf']-s['vmaf']:+.3f} (budget {EPS})\n")


def main():
    print(f"bitrate-greedy policy | {EPISODES} eps | eps={EPS} VMAF | alpha={ALPHA}")
    print("Isolating BANKING: SAFETY (banking off) vs CERTIFIED (banking on), same conformal safety.\n")

    print("### A) Real train traces, moderate buffers (throughput-limited regime)")
    for bmax in [8.0, 12.0, 20.0]:
        _report(f"real traces | buffer {bmax:.0f}s",
                [run("raw", bmax), run("safety", bmax), run("certified", bmax)])

    print("### B) High-throughput 5G-like traces (saturated rungs feasible -> banking has headroom)")
    _report("5G steady ~9Mbps | buffer 12s",
            [run("raw", 12.0, trace_kind="steady"),
             run("safety", 12.0, trace_kind="steady"),
             run("certified", 12.0, trace_kind="steady")])
    _report("5G with dips | buffer 12s",
            [run("raw", 12.0, trace_kind="dips"),
             run("safety", 12.0, trace_kind="dips"),
             run("certified", 12.0, trace_kind="dips")])


if __name__ == "__main__":
    main()
