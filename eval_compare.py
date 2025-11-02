import os
import argparse
import csv
import random
from typing import Dict
import numpy as np
import torch

from models.fcc_trace_loader import FCCTraceLoader
from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
from models.content_aware_model import create_content_aware_model
from models.reward_composite import compute_composite_reward  # ✅ نسخه جدید

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_env(mode: str) -> ContentAwareEnvFCC:
    loader = FCCTraceLoader(
        fcc_trace_dir='data/network_traces/fcc',
        train_file='data/network_traces/fcc/splits/fcc_train.txt',
        val_file='data/network_traces/fcc/splits/fcc_val.txt',
        test_file='data/network_traces/fcc/splits/fcc_test.txt',
    )
    env = ContentAwareEnvFCC(
        loader,
        'data/features/si_ti_features.json',
        'data/vmaf/vmaf_table.json',
        'data/videos',
        mode=mode,
    )
    env.use_composite_reward = True
    return env

def load_model(ckpt_path: str, device: torch.device):
    model = create_content_aware_model().to(device)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model

def _to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

def evaluate_model(env, model, device, episodes=100):
    results = []
    for ep in range(episodes):
        s = env.reset(split='test', video_id=np.random.choice([1, 2, 3, 4, 5, 6]))
        done = False
        last_bitrate = None
        ep_reward, ep_rebuf, ep_bitrates, ep_vmafs = 0.0, 0.0, [], []

        while not done:
            with torch.no_grad():
                net = _to_tensor(s['network'], device)
                cont = _to_tensor(s['content'], device)
                vmaf = _to_tensor(s['vmaf'], device)
                action, _, _ = model.select_action(net, cont, vmaf)

            # ✅ Safety Wrapper
            buf = s['network'][2][-1] * 60.0 if 'network' in s else 0.0
            if buf < 5:
                action = min(int(action), 1)
            elif buf < 10:
                action = min(int(action), 2)
            elif buf < 20:
                action = min(int(action), 3)

            s_next, env_r, done, info = env.step(int(action))
            r = compute_composite_reward(info, last_bitrate)
            last_bitrate = info.get("bitrate", last_bitrate)
            ep_reward += r
            ep_rebuf += info.get("rebuffer_time", 0.0)
            ep_bitrates.append(info.get("bitrate", 0.0))
            ep_vmafs.append(info.get("vmaf", 0.0))
            s = s_next

        results.append({
            "Reward": ep_reward,
            "Rebuf": ep_rebuf,
            "Bitrate": np.mean(ep_bitrates),
            "VMAF": np.mean(ep_vmafs)
        })

    mean_vals = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}
    print(f"\n✅ Final Averages — Reward={mean_vals['Reward']:.2f}, "
          f"Rebuffer={mean_vals['Rebuf']:.2f}s, "
          f"Bitrate={mean_vals['Bitrate']:.0f}kbps, "
          f"VMAF={mean_vals['VMAF']:.1f}")
    return mean_vals

def main():
    parser = argparse.ArgumentParser(description="Evaluate model with Safety Wrapper")
    parser.add_argument('--ckpt', type=str, default='results/composite_training/best_model.pth')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seeds(42)
    device = torch.device(args.device)

    print("🧪 Building environment...")
    env = build_env(mode='test')
    print(f"🧠 Loading model: {args.ckpt}")
    model = load_model(args.ckpt, device)

    print("\n🚀 Evaluating with Safety Wrapper (Balanced Reward)...")
    evaluate_model(env, model, device, episodes=args.episodes)

if __name__ == "__main__":
    main()
