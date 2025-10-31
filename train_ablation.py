import os
import argparse
import torch
import numpy as np
from models.fcc_trace_loader import FCCTraceLoader
from models.content_aware_model import create_content_aware_model
from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
from models.ppo_trainer import PPOTrainer
from models.logger import TrainingLogger
from models.reward_vmaf import compute_vmaf_reward
from models.reward_standard import compute_standard_reward

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧠 Using device: {device}\n")

    # Load traces
    print("📊 Loading traces...")
    loader = FCCTraceLoader(
        fcc_trace_dir='data/network_traces/fcc',
        train_file='data/network_traces/fcc/splits/fcc_train.txt',
        val_file='data/network_traces/fcc/splits/fcc_val.txt'
    )

    # Adjust reward function
    if args.no_vmaf:
        compute_reward = compute_standard_reward
        reward_type = "Bitrate-Based"
    else:
        compute_reward = compute_vmaf_reward
        reward_type = "VMAF-Based"

    print(f"✅ Reward type: {reward_type}\n")

    # Create environments
    env_train = ContentAwareEnvFCC(loader, 'data/features/si_ti_features.json', 'data/vmaf/vmaf_table.json', 'data/videos', mode='train')
    env_val = ContentAwareEnvFCC(loader, 'data/features/si_ti_features.json', 'data/vmaf/vmaf_table.json', 'data/videos', mode='val')
    env_train.set_custom_reward_fn(compute_reward)
    env_val.set_custom_reward_fn(compute_reward)

    # Create model
    model = create_content_aware_model().to(device)

    if args.no_content:
        model.disable_content_input()
        print("⚠️ Content input disabled")
    if args.no_vmaf:
        model.disable_vmaf_input()
        print("⚠️ VMAF input disabled")

    # Setup training
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

    logger = TrainingLogger(log_dir='results/logs', run_name='ablation_study')
    trainer.external_logger = logger

    print("\n🚀 Starting training (Ablation Mode)\n")
    best_reward = float('-inf')
    for update in range(1, 101):
        rollout = trainer.collect_rollout(n_steps=2048)
        eval_metrics = trainer.evaluate_policy(env_val, n_episodes=10)
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