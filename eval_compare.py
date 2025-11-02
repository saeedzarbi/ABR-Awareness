import os
import argparse
import csv
import random
from typing import Dict

import numpy as np
import torch

# --- Project imports ---
from models.fcc_trace_loader import FCCTraceLoader
from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
from models.content_aware_model import create_content_aware_model
from models.reward_composite import compute_composite_reward


def set_seeds(seed: int = 42):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env(mode: str) -> ContentAwareEnvFCC:
    """Build FCC evaluation environment with composite reward enabled."""
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
    env.use_composite_reward = True  # ✅ ensure composite reward mode is active
    return env


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained PyTorch model from checkpoint."""
    model = create_content_aware_model().to(device)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def _to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)


def evaluate_model(env: ContentAwareEnvFCC,
                   model: torch.nn.Module,
                   device: torch.device,
                   episodes: int = 100,
                   use_composite: bool = True) -> Dict[str, float]:
    """Run evaluation and return aggregate metrics."""
    env_rewards, comp_rewards = [], []
    rebuf_list, bitrate_list, vmaf_list, switches_list = [], [], [], []

    for ep in range(episodes):
        s = env.reset(split='test', video_id=np.random.choice([1, 2, 3, 4, 5, 6]))
        done = False
        last_bitrate = None
        last_action_bitrate = None

        ep_env_r = ep_comp_r = ep_rebuf = 0.0
        ep_bitrates, ep_vmafs = [], []
        ep_switches = 0

        while not done:
            with torch.no_grad():
                net = _to_tensor(s['network'], device)
                cont = _to_tensor(s['content'], device)
                vmaf = _to_tensor(s['vmaf'], device)
                action, _, _ = model.select_action(net, cont, vmaf)

            s_next, env_reward, done, info = env.step(int(action))

            # Track metrics
            ep_env_r += float(env_reward)
            ep_comp_r += compute_composite_reward(info, last_bitrate)
            ep_rebuf += float(info.get('rebuffer_time', 0.0))
            ep_bitrates.append(float(info.get('bitrate', 0.0)))
            ep_vmafs.append(float(info.get('vmaf', 0.0)))

            # Switches
            current_bitrate = float(info.get('bitrate', 0.0))
            if last_action_bitrate is not None and current_bitrate != last_action_bitrate:
                ep_switches += 1
            last_action_bitrate = current_bitrate
            last_bitrate = current_bitrate if current_bitrate > 0 else last_bitrate
            s = s_next

        env_rewards.append(ep_env_r)
        comp_rewards.append(ep_comp_r)
        rebuf_list.append(ep_rebuf)
        bitrate_list.append(np.mean(ep_bitrates) if ep_bitrates else 0.0)
        vmaf_list.append(np.mean(ep_vmafs) if ep_vmafs else 0.0)
        switches_list.append(ep_switches)

    env_reward_mean = float(np.mean(env_rewards)) if env_rewards else 0.0
    comp_reward_mean = float(np.mean(comp_rewards)) if comp_rewards else 0.0

    metrics = {
        'reward': comp_reward_mean if use_composite else env_reward_mean,
        'reward_env_mean': env_reward_mean,
        'reward_comp_mean': comp_reward_mean,
        'rebuffer_s': float(np.mean(rebuf_list)) if rebuf_list else 0.0,
        'bitrate_kbps': float(np.mean(bitrate_list)) if bitrate_list else 0.0,
        'vmaf': float(np.mean(vmaf_list)) if vmaf_list else 0.0,
        'switches': float(np.mean(switches_list)) if switches_list else 0.0,
        'episodes': episodes,
    }
    return metrics


def save_csv(path: str, rows: Dict[str, Dict[str, float]]):
    """Save evaluation metrics to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        'name', 'episodes', 'reward', 'reward_env_mean', 'reward_comp_mean',
        'rebuffer_s', 'bitrate_kbps', 'vmaf', 'switches'
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, m in rows.items():
            writer.writerow({'name': name, **m})


def main():
    parser = argparse.ArgumentParser(description='Evaluate content-aware ABR model with unified composite reward.')
    parser.add_argument('--ckpt', type=str, default='results/composite_training/best_model.pth',
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of test episodes')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use-composite', action='store_true',
                        help='Report composite reward as the primary reward')
    parser.add_argument('--out', type=str, default='results/compare_eval/fcc_summary_fixed.csv')
    args = parser.parse_args()

    set_seeds(args.seed)
    device = torch.device(args.device)

    print('🧪 Building test environment (FCC)...')
    env = build_env(mode='test')

    print(f'🧠 Loading model from {args.ckpt} on {device}...')
    model = load_model(args.ckpt, device)

    print(f"\n>>> Evaluating OUR MODEL on FCC test set ({args.episodes} eps)\n"
          f"    - Primary metric: {'COMPOSITE' if args.use_composite else 'ENV/DEFAULT'} reward\n")

    ours = evaluate_model(env, model, device, episodes=args.episodes, use_composite=args.use_composite)

    print(f"  Reward={ours['reward']:.2f}  Rebuf={ours['rebuffer_s']:.2f}s  "
          f"Bitrate={ours['bitrate_kbps']:.0f}kbps  VMAF={ours['vmaf']:.1f}  Switches={ours['switches']:.2f}")
    print(f"  (env_reward_mean={ours['reward_env_mean']:.2f}, "
          f"composite_reward_mean={ours['reward_comp_mean']:.2f})\n")

    save_csv(args.out, {'our_model_composite' if args.use_composite else 'our_model_env': ours})
    print(f"✓ Saved summaries to {args.out}")


if __name__ == '__main__':
    main()
