# eval_compare.py
# -----------------------------
# Evaluation of multiple ABR agents (Our PPO, Pensieve, MPC, BOLA, ComycoLike)
# -----------------------------

import os, json, argparse, random, numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.content_aware_model import create_content_aware_model, ContentAwareActor
from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
from models.content_aware_env_v2 import ContentAwareEnvV2
from models.pensieve_reward import PensieveReward

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------
# Policy Classes
# ----------------------------------------------------------

class OurPolicy:
    def __init__(self, ckpt_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_content_aware_model().to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'])
        elif isinstance(state, dict) and 'state_dict' in state:
            self.model.load_state_dict(state['state_dict'])
        else:
            self.model.load_state_dict(state)
        self.model.eval()

    def select_action(self, s):
        net = torch.FloatTensor(s['network']).unsqueeze(0).to(self.device)
        cont = torch.FloatTensor(s['content']).unsqueeze(0).to(self.device)
        vmaf = torch.FloatTensor(s['vmaf']).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self.model(net, cont, vmaf)
        return int(probs.argmax(dim=1).item())

class PensievePolicy:
    """
    Simplified Pensieve-style agent (network-only input)
    compatible with your current ContentAwareActor implementation.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # از همان مدل تو استفاده می‌کنیم، ولی ورودی محتوا و VMAF را صفر می‌دهیم
        self.model = create_content_aware_model().to(self.device)
        self.model.eval()

    def select_action(self, s):
        # فقط ورودی شبکه فعال است؛ بقیه صفر می‌شود
        net = torch.FloatTensor(s['network']).unsqueeze(0).to(self.device)
        # ورودی‌های محتوا و vmaf را با صفر پر می‌کنیم تا شکلشان درست باشد
        cont = torch.zeros_like(torch.FloatTensor(s['content'])).unsqueeze(0).to(self.device)
        vmaf = torch.zeros_like(torch.FloatTensor(s['vmaf'])).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs, _ = self.model(net, cont, vmaf)

        return int(probs.argmax(dim=1).item())


class MPCPolicy:
    def __init__(self, bitrate_levels=[300,750,1850,2850,4300,6000], H=5):
        self.bitrates = bitrate_levels
        self.H = H
        self.reward = PensieveReward(
            rebuffer_penalty=4.3, smoothness_penalty=1.0, bitrate_levels=bitrate_levels
        )

    def _avg_tp(self, past_tp):
        arr = np.array(past_tp[-8:] or [1000.0])
        arr = np.clip(arr, 1e-3, None)
        return len(arr) / np.sum(1.0 / arr)

    def select_action(self, s):
        last_br = getattr(self, "_last_br", 0)
        tp_hat = self._avg_tp(getattr(self, "_past_tp", []))
        buffer = s['network'][2, -1] * 60.0
        best_a, best_q = 0, -1e9
        for a, br in enumerate(self.bitrates):
            dl_time = (br * 4.0) / max(tp_hat, 1e-3)
            rebuf = max(0.0, dl_time - buffer)
            vmaf = float(s['vmaf'][a] * 100.0)
            q = self.reward.compute_reward_vmaf(
                vmaf_score=vmaf,
                rebuffer_time=rebuf,
                last_bitrate=last_br,
                current_bitrate=br
            )
            if q > best_q:
                best_q, best_a = q, a
        self._last_br = self.bitrates[best_a]
        return best_a

class BOLAPolicy:
    def __init__(self, bitrate_levels=[300,750,1850,2850,4300,6000]):
        self.bitrates = bitrate_levels

    def select_action(self, s):
        buf = s['network'][2, -1] * 60.0
        v = (s['vmaf'] * 100.0).clip(1, 100)
        util = np.log(v)
        if buf < 6: idx = 0
        elif buf < 12: idx = 1
        elif buf < 20: idx = 2
        elif buf < 30: idx = 3
        elif buf < 40: idx = 4
        else: idx = 5
        cand = range(max(0, idx-1), min(len(self.bitrates), idx+2))
        return int(np.argmax(util[list(cand)]) + (min(cand)))

class ComycoLikePolicy(MPCPolicy):
    def __init__(self, bitrate_levels=[300,750,1850,2850,4300,6000], H=5):
        super().__init__(bitrate_levels, H)
        self.reward = PensieveReward(
            rebuffer_penalty=4.3, smoothness_penalty=2.0, bitrate_levels=bitrate_levels
        )

# ----------------------------------------------------------
# Environment setup
# ----------------------------------------------------------

def make_env(dataset):
    if dataset == 'fcc':
        loader = FCCTraceLoader(
            fcc_trace_dir='data/network_traces/fcc',
            train_file='data/network_traces/fcc/splits/fcc_train.txt',
            val_file='data/network_traces/fcc/splits/fcc_val.txt',
            test_file='data/network_traces/fcc/splits/fcc_test.txt'
        )
        env_val = ContentAwareEnvFCC(
            fcc_trace_loader=loader,
            features_file='data/features/si_ti_features.json',
            vmaf_file='data/vmaf/vmaf_table.json',
            video_dir='data/videos',
            mode='val'
        )
    elif dataset == 'cooked':
        env_val = ContentAwareEnvV2(
            trace_dir='data/network_traces/cooked_traces',
            features_file='data/features/si_ti_features.json',
            vmaf_file='data/vmaf/vmaf_table.json'
        )
    else:
        raise ValueError("dataset must be fcc or cooked")
    return env_val

# ----------------------------------------------------------
# Evaluation Loop
# ----------------------------------------------------------

def run_policy(env, policy, episodes=100, dataset='fcc'):
    results = []
    for ep in range(episodes):
        s = env.reset(split='val')
        done = False
        total_reward, total_rebuf, total_vmaf, total_bitrate, switches = 0, 0, 0, 0, 0
        last_br = None
        chunks = 0
        while not done:
            a = policy.select_action(s)
            s_next, r, done, info = env.step(a)
            total_reward += r
            total_rebuf += info.get('rebuffer_time', 0.0)
            total_bitrate += info.get('bitrate', 0.0)
            total_vmaf += float(s['vmaf'][a] * 100.0)
            if last_br is not None and info.get('bitrate') != last_br:
                switches += 1
            last_br = info.get('bitrate', last_br)
            s = s_next if s_next is not None else s
            chunks += 1
        results.append({
            'episode': ep,
            'reward': total_reward,
            'rebuffer_s': total_rebuf,
            'avg_bitrate_kbps': total_bitrate / max(1, chunks),
            'avg_vmaf': total_vmaf / max(1, chunks),
            'switches': switches,
            'chunks': chunks
        })
    return pd.DataFrame(results)

# ----------------------------------------------------------
# Main entry
# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['fcc','cooked'], default='fcc')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--out', type=str, default='results/compare_eval')
    parser.add_argument('--our_ckpt', type=str, default='results/fcc_training_auto/best_model.pth')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = make_env(args.dataset)

    policies = {
        'our': OurPolicy(args.our_ckpt),
        'pensieve': PensievePolicy(),
        'mpc': MPCPolicy(),
        'bola': BOLAPolicy(),
        'comyco_like': ComycoLikePolicy()
    }

    summaries = []
    for name, pol in policies.items():
        print(f"\n>>> Evaluating: {name} on {args.dataset} ({args.episodes} eps)")
        df = run_policy(env, pol, episodes=args.episodes, dataset=args.dataset)
        df.to_csv(os.path.join(args.out, f'{args.dataset}_{name}_episodes.csv'), index=False)
        summary = df[['reward','rebuffer_s','avg_bitrate_kbps','avg_vmaf','switches']].mean().to_dict()
        summary['policy'] = name
        summary['episodes'] = len(df)
        summaries.append(summary)
        print(f"  Reward={summary['reward']:.2f}  Rebuf={summary['rebuffer_s']:.2f}s  "
              f"Bitrate={summary['avg_bitrate_kbps']:.0f}kbps  VMAF={summary['avg_vmaf']:.1f}  "
              f"Switches={summary['switches']:.2f}")

    out_df = pd.DataFrame(summaries)
    out_path = os.path.join(args.out, f'{args.dataset}_summary.csv')
    out_df.to_csv(out_path, index=False)
    print(f"\n✓ Saved summaries to {out_path}")

    # Optional: significance test
    try:
        import scipy.stats as st
        a = pd.read_csv(os.path.join(args.out, f'{args.dataset}_our_episodes.csv'))['reward']
        b = pd.read_csv(os.path.join(args.out, f'{args.dataset}_pensieve_episodes.csv'))['reward']
        t, p = st.ttest_rel(a, b)
        print(f"Paired t-test (our vs pensieve): t={t:.2f}, p={p:.3g}")
    except Exception as e:
        print("Significance test skipped:", e)

if __name__ == '__main__':
    main()
