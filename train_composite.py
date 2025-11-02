import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.fcc_trace_loader import FCCTraceLoader
from models.content_aware_model import create_content_aware_model
from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
from models.ppo_trainer import PPOTrainer
from models.logger import TrainingLogger
from models.reward_composite import compute_balanced_reward

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def _to_int_action(a):
    if isinstance(a, tuple):
        a = a[0]
    if isinstance(a, torch.Tensor):
        a = int(a.item())
    return int(a)

def main():
    print("🚀 Training Content-Aware ABR Agent (Composite Reward)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}\n")

    # === Load FCC Traces ===
    loader = FCCTraceLoader(
        fcc_trace_dir='data/network_traces/fcc',
        train_file='data/network_traces/fcc/splits/fcc_train.txt',
        val_file='data/network_traces/fcc/splits/fcc_val.txt',
        test_file='data/network_traces/fcc/splits/fcc_test.txt'
    )
    print("✅ FCC traces loaded.\n")

    env_train = ContentAwareEnvFCC(
        loader, 'data/features/si_ti_features.json',
        'data/vmaf/vmaf_table.json', 'data/videos', mode='train'
    )
    env_val = ContentAwareEnvFCC(
        loader, 'data/features/si_ti_features.json',
        'data/vmaf/vmaf_table.json', 'data/videos', mode='val'
    )

    # === Model & Trainer ===
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

    # === Adaptive hyperparameters ===
    base_lr = 3e-4
    base_entropy = 0.02
    best_val_reward = float('-inf')
    patience = 25   # ⬅️ افزایش داده شد از 15 به 25
    stop_counter = 0

    val_history = []

    # === Training Loop ===
    for update in range(1, 201):
        rollout = trainer.collect_rollout(n_steps=2048)

        # Apply composite reward
        infos = getattr(env_train, 'info_history', [])
        last_bitrate = None
        n = min(len(rollout.rewards), len(infos))
        for i in range(n):
            rollout.rewards[i] = compute_balanced_reward(infos[i], last_bitrate)
            last_bitrate = infos[i].get('bitrate', last_bitrate)

        # Update learning rate dynamically
        current_lr = max(1e-4, base_lr * (0.995 ** update))
        for g in trainer.optimizer.param_groups:
            g['lr'] = current_lr

        trainer.entropy_coef = max(0.005, base_entropy * (0.99 ** update))

        train_info = trainer.update_policy(rollout) or {}
        policy_loss = float(train_info.get("policy_loss", 0.0))

        # === Validation ===
        val_rewards = []
        for _ in range(5):
            s = env_val.reset(split='val')
            done = False
            last_bitrate = None
            total_r = 0.0
            while not done:
                with torch.no_grad():
                    net = torch.FloatTensor(s['network']).unsqueeze(0).to(device)
                    cont = torch.FloatTensor(s['content']).unsqueeze(0).to(device)
                    vmaf = torch.FloatTensor(s['vmaf']).unsqueeze(0).to(device)
                    a = trainer.model.select_action(net, cont, vmaf)
                    a = _to_int_action(a)
                s_next, _, done, info = env_val.step(a)
                total_r += compute_balanced_reward(info, last_bitrate)
                last_bitrate = info.get('bitrate', last_bitrate)
                s = s_next
            val_rewards.append(total_r)

        avg_val_r = float(np.mean(val_rewards))
        val_history.append(avg_val_r)

        # Early stopping + checkpoint
        if avg_val_r > best_val_reward:
            best_val_reward = avg_val_r
            stop_counter = 0
            os.makedirs('results/composite_training', exist_ok=True)
            torch.save(model.state_dict(), 'results/composite_training/best_model.pth')
        else:
            stop_counter += 1

        print(f"[{update:03d}] TrainLoss: {policy_loss:.4f} | "
              f"ValR: {avg_val_r:7.2f} | Best: {best_val_reward:7.2f} | "
              f"Entropy: {trainer.entropy_coef:.4f} | LR: {current_lr:.6f}")

        # === Save progress chart ===
        if update % 5 == 0:
            plt.figure(figsize=(8, 4))
            plt.plot(val_history, label='Validation Reward', color='royalblue')
            plt.title("Composite Reward Training Progress")
            plt.xlabel("Update")
            plt.ylabel("Validation Reward")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            os.makedirs('results/plots', exist_ok=True)
            plt.tight_layout()
            plt.savefig('results/plots/val_reward_curve.png')
            plt.close()

        if stop_counter >= patience:
            print("⏸️ Early stopping triggered.")
            break

    print("\n✅ Training complete.")
    print(f"📈 Best validation reward: {best_val_reward:.2f}")
    print("💾 Model saved to results/composite_training/best_model.pth\n")
    print("📊 Validation reward curve saved to results/plots/val_reward_curve.png")

if __name__ == "__main__":
    main()
