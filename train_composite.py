import os
import torch
import numpy as np
from models.fcc_trace_loader import FCCTraceLoader
from models.content_aware_model import create_content_aware_model
from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
from models.ppo_trainer import PPOTrainer
from models.logger import TrainingLogger
from models.reward_composite import compute_composite_reward

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    print("🚀 Training Content-Aware ABR Agent (Composite Reward)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}\n")

    # === Load FCC Traces ===
    print("📦 Loading FCC traces...")
    loader = FCCTraceLoader(
        fcc_trace_dir='data/network_traces/fcc',
        train_file='data/network_traces/fcc/splits/fcc_train.txt',
        val_file='data/network_traces/fcc/splits/fcc_val.txt',
        test_file='data/network_traces/fcc/splits/fcc_test.txt'
    )
    print("✅ FCC traces loaded.\n")

    # === Create environments ===
    env_train = ContentAwareEnvFCC(loader, 'data/features/si_ti_features.json',
                                   'data/vmaf/vmaf_table.json', 'data/videos', mode='train')
    env_val = ContentAwareEnvFCC(loader, 'data/features/si_ti_features.json',
                                 'data/vmaf/vmaf_table.json', 'data/videos', mode='val')

    # === Model ===
    model = create_content_aware_model().to(device)
    trainer = PPOTrainer(
        model=model,
        env=env_train,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.02,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64
    )
    trainer.device = device
    logger = TrainingLogger(log_dir='results/logs', run_name='composite_reward')
    trainer.external_logger = logger

    print("🎓 Training initialized.\n")

    # === Adaptive parameters ===
    base_lr = 3e-4
    base_entropy = 0.02
    best_val_reward = float('-inf')
    patience = 15
    stop_counter = 0

    for update in range(1, 201):
        # Rollout
        rollout = trainer.collect_rollout(n_steps=2048)

        # Compute new reward manually
        infos = getattr(env_train, 'info_history', [])
        last_bitrate = None
        for i in range(len(rollout.rewards)):
            if i < len(infos):
                rollout.rewards[i] = compute_composite_reward(infos[i], last_bitrate)
                last_bitrate = infos[i].get('bitrate', last_bitrate)

        # === Normalize advantages ===
        adv = rollout.advantages
        rollout.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        # === Update policy ===
        trainer.lr = max(1e-4, base_lr * (0.995 ** update))
        trainer.entropy_coef = max(0.005, base_entropy * (0.99 ** update))
        train_info = trainer.update_policy(rollout)

        # === Evaluate ===
        val_rewards = []
        for _ in range(5):
            s = env_val.reset(split='val')
            done = False
            last_bitrate = None
            total_r = 0
            while not done:
                with torch.no_grad():
                    a = trainer.model.select_action(s['network'], s['content'], s['vmaf'])
                s_next, _, done, info = env_val.step(a)
                total_r += compute_composite_reward(info, last_bitrate)
                last_bitrate = info.get('bitrate', last_bitrate)
                s = s_next
            val_rewards.append(total_r)
        avg_val_r = np.mean(val_rewards)

        if avg_val_r > best_val_reward:
            best_val_reward = avg_val_r
            stop_counter = 0
            os.makedirs('results/composite_training', exist_ok=True)
            torch.save(model.state_dict(), 'results/composite_training/best_model.pth')
        else:
            stop_counter += 1

        print(f"[{update:03d}] Train Loss: {train_info['policy_loss']:.4f} | "
              f"ValR: {avg_val_r:.2f} | Best: {best_val_reward:.2f} | "
              f"Entropy: {trainer.entropy_coef:.4f} | LR: {trainer.lr:.6f}")

        if stop_counter >= patience:
            print("⏸️ Early stopping triggered.")
            break

    print("\n✅ Training complete.")
    print(f"📈 Best validation reward: {best_val_reward:.2f}")
    print("💾 Model saved to results/composite_training/best_model.pth\n")

if __name__ == "__main__":
    main()
