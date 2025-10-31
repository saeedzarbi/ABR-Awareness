import os
import argparse
import torch
import numpy as np
from models.fcc_trace_loader import FCCTraceLoader
from models.content_aware_model import create_content_aware_model
from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
from models.ppo_trainer import PPOTrainer
from models.logger import TrainingLogger
from reward_vmaf import compute_vmaf_reward
from reward_standard import compute_standard_reward

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def compute_rewarded_rollout(trainer, n_steps, reward_fn):
    buffer = trainer.collect_rollout(n_steps=n_steps)
    infos = trainer.env.info_history if hasattr(trainer.env, 'info_history') else []
    last_bitrate = None
    for i in range(len(buffer.rewards)):
        if i < len(infos):
            info = infos[i]
            buffer.rewards[i] = reward_fn(info, last_bitrate)
            last_bitrate = info.get('bitrate', last_bitrate)
    return buffer

def evaluate_rewarded(trainer, env, reward_fn, n_episodes):
    results = []
    for _ in range(n_episodes):
        s = env.reset(split='val')
        done = False
        total_reward = 0
        last_bitrate = None
        while not done:
            with torch.no_grad():
                action_probs, _ = trainer.model(
                    torch.FloatTensor(s['network']).unsqueeze(0).to(trainer.device),
                    torch.zeros_like(torch.FloatTensor(s['content'])).unsqueeze(0).to(trainer.device)
                    if 'content' in s else torch.zeros(1, 2).to(trainer.device),
                    torch.zeros_like(torch.FloatTensor(s['vmaf'])).unsqueeze(0).to(trainer.device)
                    if 'vmaf' in s else torch.zeros(1, 6).to(trainer.device)
                )
                a = int(torch.argmax(action_probs, dim=1).item())
            s_next, _, done, info = env.step(a)
            r = reward_fn(info, last_bitrate)
            total_reward += r
            last_bitrate = info.get('bitrate', last_bitrate)
            s = s_next
        results.append(total_reward)
    return {"reward": np.mean(results)}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧠 Using device: {device}\n")

    print("📊 Loading traces...")
    loader = FCCTraceLoader(
        fcc_trace_dir='data/network_traces/fcc',
        train_file='data/network_traces/fcc/splits/fcc_train.txt',
        val_file='data/network_traces/fcc/splits/fcc_val.txt',
        test_file='data/network_traces/fcc/splits/fcc_test.txt'
    )

    if args.no_vmaf:
        reward_fn = compute_standard_reward
        reward_type = "Bitrate-Based"
    else:
        reward_fn = compute_vmaf_reward
        reward_type = "VMAF-Based"

    print(f"✅ Reward type: {reward_type}\n")

    env_train = ContentAwareEnvFCC(loader, 'data/features/si_ti_features.json', 'data/vmaf/vmaf_table.json', 'data/videos', mode='train')
    env_val = ContentAwareEnvFCC(loader, 'data/features/si_ti_features.json', 'data/vmaf/vmaf_table.json', 'data/videos', mode='val')

    model = create_content_aware_model().to(device)

    trainer = PPOTrainer(
        model=model,
        env=env_train,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64
    )
    trainer.device = device

    logger = TrainingLogger(log_dir='results/logs', run_name='ablation_study')
    trainer.external_logger = logger

    print("\n🚀 Starting training (Ablation Mode)\n")
    best_reward = float('-inf')
    for update in range(1, 101):
        rollout = compute_rewarded_rollout(trainer, n_steps=2048, reward_fn=reward_fn)
        eval_metrics = evaluate_rewarded(trainer, env_val, reward_fn, n_episodes=10)
        avg_reward = eval_metrics.get('reward', 0.0)
        trainer.update_policy(rollout)

        if avg_reward > best_reward:
            best_reward = avg_reward

        print(f"Update {update:03d} | Val Reward: {avg_reward:.2f} | Best: {best_reward:.2f}")

    print("\n✅ Training complete.")
    print(f"📈 Best Val Reward: {best_reward:.2f}")
    print(f"🔬 Ablation setting: no_content={args.no_content}, no_vmaf={args.no_vmaf}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-content', action='store_true', help="Remove content features from model input")
    parser.add_argument('--no-vmaf', action='store_true', help="Use standard reward (bitrate-based) instead of VMAF")
    args = parser.parse_args()
    main(args)