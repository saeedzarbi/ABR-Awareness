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


def run(arm: str, buffer_max: float, policy: str = "greedy", trace_kind: str | None = None,
        eps: float | None = None):
    wrapped = arm != "raw"
    eps = EPS if eps is None else eps
    if not wrapped:
        env = make_env(buffer_max, trace_kind)
    else:
        predictive = arm in ("certified_plus", "certified_fc")
        cfg = CPShieldConfig(
            enabled=True,
            enable_banking=(arm in ("certified", "certified_plus", "certified_fc")),
            epsilon_vmaf=eps,
            enable_conformal=True,
            conformal=ConformalConfig(alpha=ALPHA, window=200, k_predict=5),
            safety_margin=0.5, min_buffer=0.3,
            predictive=predictive,
            lookahead=(6 if arm == "certified_fc" else 1),
            epsilon_risk=max(4.0, eps), risk_buffer=8.0,
            forecast_dips=(arm == "certified_fc"), horizon_quantile=0.2,
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
    reb_arr = np.array(reb)
    thr = np.percentile(reb_arr, 90)
    tail = reb_arr[reb_arr >= thr]
    reb_cvar = float(tail.mean()) if tail.size else float(reb_arr.max())
    return {
        "arm": arm,
        "reb": float(np.mean(reb)),
        "reb_p95": float(np.percentile(reb, 95)),
        "reb_cvar10": reb_cvar,
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


def _report_plus(title, buffer_max, trace_kind):
    """4-arm report incl. CERTIFIED+ (predictive pre-banking). Isolates the value
    of the conformal look-ahead: certified+ vs certified at the SAME budget."""
    rows = [run("raw", buffer_max, trace_kind=trace_kind),
            run("safety", buffer_max, trace_kind=trace_kind),
            run("certified", buffer_max, trace_kind=trace_kind),
            run("certified_plus", buffer_max, trace_kind=trace_kind)]
    print(f"===== {title} =====")
    print(f"{'arm':>14}{'reb':>9}{'reb_p95':>9}{'stallfree':>10}{'vmaf':>8}"
          f"{'bitrate':>9}{'buffer':>8}{'interv':>8}{'cover':>7}")
    for r in rows:
        print(f"{r['arm']:>14}{r['reb']:>9.3f}{r['reb_p95']:>9.3f}{r['stallfree']:>10.3f}"
              f"{r['vmaf']:>8.2f}{r['bitrate']:>9.0f}{r['buffer']:>8.2f}{r['interv']:>8.3f}{r['cover']:>7.3f}")
    c, p = rows[2], rows[3]
    bw = 100 * (p["bitrate"] - c["bitrate"]) / max(c["bitrate"], 1e-9)
    print(f"  PREDICTIVE (certified+ - certified): reb {p['reb']-c['reb']:+.3f}  "
          f"reb_p95 {p['reb_p95']-c['reb_p95']:+.3f}  stallfree {p['stallfree']-c['stallfree']:+.3f}  "
          f"bitrate {bw:+.1f}%  VMAF {p['vmaf']-c['vmaf']:+.3f}\n")


def _eps_frontier(title, buffer_max, trace_kind, eps_grid=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Sweep the perceptual budget for reactive certified banking. Because the
    VMAF-rate curve saturates, a large fraction of the bandwidth/stall benefit
    is retained even at very tight epsilon -> a favourable operating frontier."""
    base = run("safety", buffer_max, trace_kind=trace_kind)  # banking OFF reference
    print(f"===== {title} =====")
    print(f"{'eps':>6}{'reb':>9}{'reb_p95':>9}{'reb_cvar10':>11}{'vmaf':>8}"
          f"{'bitrate':>9}{'bwSaved%':>9}{'dVMAF':>8}")
    print(f"{'off':>6}{base['reb']:>9.3f}{base['reb_p95']:>9.3f}{base['reb_cvar10']:>11.3f}"
          f"{base['vmaf']:>8.2f}{base['bitrate']:>9.0f}{0.0:>9.1f}{0.0:>8.3f}")
    for e in eps_grid:
        r = run("certified", buffer_max, trace_kind=trace_kind, eps=e)
        bw = 100 * (base["bitrate"] - r["bitrate"]) / max(base["bitrate"], 1e-9)
        print(f"{e:>6.2f}{r['reb']:>9.3f}{r['reb_p95']:>9.3f}{r['reb_cvar10']:>11.3f}"
              f"{r['vmaf']:>8.2f}{r['bitrate']:>9.0f}{bw:>9.1f}{r['vmaf']-base['vmaf']:>8.3f}")
    print()


def _tail_report(title, buffer_max, trace_kind):
    rows = [run("safety", buffer_max, trace_kind=trace_kind),
            run("certified", buffer_max, trace_kind=trace_kind),
            run("certified_plus", buffer_max, trace_kind=trace_kind),
            run("certified_fc", buffer_max, trace_kind=trace_kind)]
    print(f"===== {title} =====")
    print(f"{'arm':>14}{'reb_mean':>10}{'reb_cvar10':>11}{'reb_p95':>9}{'vmaf':>8}{'bitrate':>9}")
    for r in rows:
        print(f"{r['arm']:>14}{r['reb']:>10.3f}{r['reb_cvar10']:>11.3f}"
              f"{r['reb_p95']:>9.3f}{r['vmaf']:>8.2f}{r['bitrate']:>9.0f}")
    s, fc = rows[0], rows[3]
    d = lambda x: 100 * (x - s['reb_cvar10']) / max(s['reb_cvar10'], 1e-9)
    print(f"  tail CVaR@10% vs safety: certified {d(rows[1]['reb_cvar10']):+.1f}%  "
          f"certified+ {d(rows[2]['reb_cvar10']):+.1f}%  certified_fc {d(fc['reb_cvar10']):+.1f}%")
    print(f"  p95 vs safety: certified {rows[1]['reb_p95']-s['reb_p95']:+.1f}  "
          f"certified_fc {fc['reb_p95']-s['reb_p95']:+.1f}  "
          f"(VMAF cost fc {fc['vmaf']-s['vmaf']:+.2f})\n")


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

    print("### C) PREDICTIVE pre-banking (conformal look-ahead) vs reactive banking")
    print("    Certainty check: does the look-ahead cut stalls FURTHER at ~equal VMAF?\n")
    _report_plus("5G with dips | buffer 12s (predictive)", 12.0, "dips")
    _report_plus("5G with dips | buffer 8s  (tighter, predictive)", 8.0, "dips")
    _report_plus("real traces | buffer 8s  (predictive)", 8.0, None)

    print("### D) epsilon-FRONTIER (Pareto): perceptual budget vs bandwidth saved / stalls")
    print("    Certainty check: VMAF saturation => most savings survive at TIGHT budgets.\n")
    _eps_frontier("5G with dips | buffer 12s", 12.0, "dips")
    _eps_frontier("real traces | buffer 8s", 8.0, None)

    print("### E) TAIL (CVaR@10%) of rebuffering: banking concentrates gains in the tail")
    _tail_report("5G with dips | buffer 8s", 8.0, "dips")
    _tail_report("5G with dips | buffer 12s", 12.0, "dips")


if __name__ == "__main__":
    main()
