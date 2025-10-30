# eval_compare.py
import os, json, argparse, random, numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn

# ---- Imports from your project ----
from models.content_aware_model import create_content_aware_model
from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
from models.content_aware_env_v2 import ContentAwareEnvV2  # Cooked
# اگر pensieve_reward لازم باشد:
from models.pensieve_reward import PensieveReward

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- Policy Adapters ----------
class OurPolicy:
    def __init__(self, ckpt_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_content_aware_model().to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        # سازگار با دو نوع ذخیره‌سازی:
        if isinstance(state, dict) and 'state_dict' in state:
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

class MPCPolicy:
    """
    MPC ساده: افق کوتاه H=5، برآورد زمان دانلود با میانگین هارمونیک سرعت‌های اخیر،
    تابع هزینه: QoE Pensieve با پنالتی rebuffer و smoothness. کامپکت و سریع.
    """
    def __init__(self, bitrate_levels=[300,750,1850,2850,4300,6000], H=5):
        self.bitrates = bitrate_levels
        self.H = H
        self.reward = PensieveReward(rebuffer_penalty=4.3, smoothness_penalty=1.0, bitrate_levels=bitrate_levels)

    def _avg_tp(self, past_tp):
        arr = np.array(past_tp[-8:] or [1000.0])
        arr = np.clip(arr, 1e-3, None)
        return len(arr) / np.sum(1.0 / arr)  # harmonic mean

    def _vmaf_for_action(self, s, a):
        # s['vmaf'] normalized [0..1], scale to 0..100:
        return float(s['vmaf'][a] * 100.0)

    def select_action(self, s):
        last_br = 0 if len(getattr(self, "_past_brs", [])) == 0 else self._past_brs[-1]
        best_a, best_q = 0, -1e9
        tp_hat = self._avg_tp(getattr(self, "_past_tp", []))
        # یک قدمه (greedy) برای سرعت — می‌تونی ترتیب‌های طول H را هم brute-force کنی اگر خواستی
        for a in range(len(self.bitrates)):
            br = self.bitrates[a]
            # تخمین زمان دانلود یک چانک
            chunk_dur = 4.0
            dl_time = (br * chunk_dur) / max(tp_hat, 1e-3)
            buffer = s['network'][2, -1] * 60.0  # denorm
            rebuf = max(0.0, dl_time - buffer)
            vmaf = self._vmaf_for_action(s, a)
            q = self.reward.compute_reward_vmaf(
                vmaf_score=vmaf,
                rebuffer_time=rebuf,
                last_bitrate=last_br,
                current_bitrate=br
            )
            if q > best_q:
                best_q, best_a = q, a
        # track history
        self._past_brs = getattr(self, "_past_brs", []) + [self.bitrates[best_a]]
        return best_a

class BOLAPolicy:
    """
    BOLA تقریبی: utility = log(VMAF) و آستانه‌های بافر؛
    نسخه‌ی ساده برای مقایسه.
    """
    def __init__(self, bitrate_levels=[300,750,1850,2850,4300,6000]):
        self.bitrates = bitrate_levels
    def select_action(self, s):
        buf = s['network'][2, -1] * 60.0
        v = (s['vmaf'] * 100.0).clip(1, 100)
        util = np.log(v)  # تقریبی
        # آستانه‌های ساده بافر:
        if buf < 6: idx = 0
        elif buf < 12: idx = 1
        elif buf < 20: idx = 2
        elif buf < 30: idx = 3
        elif buf < 40: idx = 4
        else: idx = 5
        # بین idx و idx+1 بالاترین utility/bitrate را انتخاب کن
        cand = range(max(0, idx-1), min(len(self.bitrates), idx+2))
        return int(np.argmax(util[list(cand)]) + (min(cand)))

# TODO: اگر مدل Comyco داری لود کن؛ اگر نداری، می‌توان نسخه‌ی MPC با وزن‌دهی محافظه‌کارانه‌تر (penalty بیشتر بر smoothness) را به‌عنوان تقریب به کار برد.
class ComycoLikePolicy(MPCPolicy):
    def __init__(self, bitrate_levels=[300,750,1850,2850,4300,6000], H=5):
        super().__init__(bitrate_levels, H)
        # محافظه‌کارانه‌تر: smoothness_penalty بزرگ‌تر
        self.reward = PensieveReward(rebuffer_penalty=4.3, smoothness_penalty=2.0, bitrate_levels=bitrate_levels)

# ---------- Evaluation ----------
def make_env(dataset):
    if dataset == 'fcc':
        loader = FCCTraceLoader(
            fcc_trace_dir='data/network_traces/fcc',
            train_file='data/network_traces/fcc/splits/fcc_train.txt',
            val_file='data/network_traces/fcc/splits/fcc_val.txt',
            test_file='data/network_traces/fcc/splits/fcc_test.txt'
        )
        env_val = ContentAwareEnvFCC(loader,
            features_file='data/features/si_ti_features.json',
            vmaf_file='data/vmaf/vmaf_table.json',
            video_dir='data/videos', mode='val')
    elif dataset == 'cooked':
        env_val = ContentAwareEnvV2(  # note: no video_dir/mode params in ctor
            trace_dir='data/network_traces/cooked_traces',
            features_file='data/features/si_ti_features.json',
            vmaf_file='data/vmaf/vmaf_table.json'
        )
    else:
        raise ValueError("dataset must be fcc or cooked")
    return env_val

def run_policy(env, policy, episodes=100, dataset='fcc'):
    rows = []
    for ep in range(episodes):
        # split تعیین شود:
        s = env.reset(split='val') if dataset in ('fcc', 'cooked') else env.reset()
        done = False
        total_reward, total_rebuf, total_vmaf, total_bitrate, switches = 0.0, 0.0, 0.0, 0.0, 0
        last_bitrate = None
        chunks = 0
        while not done:
            a = policy.select_action(s)
            s, r, done, info = env.step(a)
            total_reward += r
            total_rebuf += info.get('rebuffer_time', 0.0)
            total_bitrate += info.get('bitrate', 0.0)
            # VMAF از state قبلی:
            # اگر s=None (done)، از آخرین state استفاده می‌کردیم—برای سادگی میانگین تقریبی بگیر:
            # اینجا شبه‌کد ساده: می‌توانی در policy خروجی vmaf انتخابی را برگردانی.
            if s is not None:
                total_vmaf += float(s['vmaf'][a] * 100.0)
            if last_bitrate is not None and info.get('bitrate') is not None:
                if info['bitrate'] != last_bitrate: switches += 1
            last_bitrate = info.get('bitrate', last_bitrate)
            chunks += 1
        rows.append({
            'episode': ep,
            'reward': total_reward,
            'rebuffer_s': total_rebuf,
            'avg_bitrate_kbps': total_bitrate / max(1, chunks),
            'avg_vmaf': total_vmaf / max(1, chunks),
            'switches': switches,
            'chunks': chunks
        })
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, choices=['fcc','cooked'], default='fcc')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--out', type=str, default='results/compare_eval')
    p.add_argument('--our_ckpt', type=str, default='results/fcc_training_auto/best_model.pth')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = make_env(args.dataset)

    policies = {
        'our': OurPolicy(args.our_ckpt),
        'mpc': MPCPolicy(),
        'bola': BOLAPolicy(),
        'comyco_like': ComycoLikePolicy()
        # 'pensieve': PensievePolicy(ckpt_path=...)  # اگر داشتی اضافه کن
    }

    all_summaries = []
    for name, pol in policies.items():
        print(f"\n>>> Evaluating: {name} on {args.dataset} ({args.episodes} eps)")
        df = run_policy(env, pol, episodes=args.episodes, dataset=args.dataset)
        df.to_csv(os.path.join(args.out, f'{args.dataset}_{name}_episodes.csv'), index=False)
        summary = df[['reward','rebuffer_s','avg_bitrate_kbps','avg_vmaf','switches']].mean().to_dict()
        summary['policy'] = name
        summary['episodes'] = len(df)
        all_summaries.append(summary)
        print(f"  Reward={summary['reward']:.2f}  Rebuf={summary['rebuffer_s']:.2f}s  "
              f"Bitrate={summary['avg_bitrate_kbps']:.0f}kbps  VMAF={summary['avg_vmaf']:.1f}  "
              f"Switches={summary['switches']:.2f}")

    out_df = pd.DataFrame(all_summaries)
    out_df.to_csv(os.path.join(args.out, f'{args.dataset}_summary.csv'), index=False)
    print(f"\n✓ Saved summaries to {args.out}/{args.dataset}_summary.csv")

    # آزمون معناداری جفتی (Optional):
    try:
        import scipy.stats as st
        # نمونه: our vs mpc روی reward
        a = pd.read_csv(os.path.join(args.out, f'{args.dataset}_our_episodes.csv'))['reward']
        b = pd.read_csv(os.path.join(args.out, f'{args.dataset}_mpc_episodes.csv'))['reward']
        t, pval = st.ttest_rel(a, b)
        print(f"Paired t-test (our vs mpc) on reward: t={t:.2f}, p={pval:.3g}")
    except Exception as e:
        print("Significance test skipped (install scipy):", e)

if __name__ == '__main__':
    main()
