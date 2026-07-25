"""Torch-free smoke test: confirm v16 exposes a PER-CHUNK VMAF ladder to both the
reward path and the shield. Uses the existing single-res per-chunk CSV as a stand-in
(the real, non-monotone one comes from data/build_multires_vmaf.py on the server)."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from configs.paths import get_paths
from src.environment.abr_multi_env_v16 import ABREnv
from src.training.safety_shield_v14 import ShieldConfig, safe_adjust_action

P = get_paths()
stand_in = str(P["vmaf_scores"] / "vmaf_perchunk.csv")

env = ABREnv(
    video_names=["bigbuckbunny", "crowd_run", "tearsofsteel_short"],
    trace_dir=str(P["train_traces"]),
    vmaf_dir=str(P["vmaf_scores"]),
    siti_dir=str(P["content_features"]),
    max_chunks=48, random_seed=0, use_lyapunov=True, use_future=True,
    vmaf_perchunk_path=stand_in,
)
print("loaded per-chunk tables for videos:", list(env._perchunk.keys()))

obs, info = env.reset(seed=0)
print(f"\nvideo={env.current_video_name}")
cfg = ShieldConfig(level="light", vmaf_aware=True, selection="vmaf")
cfg_idx = ShieldConfig(level="light", vmaf_aware=True, selection="index")

seen = []
for t in range(6):
    scores = {int(k): round(v, 2) for k, v in sorted(env.current_vmaf_scores.items())}
    a_vmaf, _ = safe_adjust_action(env, 5, cfg)       # force top action, let shield project
    a_idx, _ = safe_adjust_action(env, 5, cfg_idx)
    seen.append(scores)
    print(f"chunk {env.chunk_idx:2d} buf={env.buffer_level:5.2f} "
          f"ladder={list(scores.values())} shield(vmaf)->{a_vmaf} shield(index)->{a_idx}")
    obs, r, term, trunc, info = env.step(3)
    if term:
        break

varying = any(seen[i] != seen[i + 1] for i in range(len(seen) - 1))
print(f"\nPER-CHUNK ladder actually varies across chunks: {varying}")
print("(On the real multi-res ladder, shield(vmaf) and shield(index) will diverge")
print(" wherever the per-chunk ladder is non-monotone.)")
