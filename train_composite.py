# train_composite.py  (Optimized)
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.fcc_trace_loader import FCCTraceLoader
from models.content_aware_model import create_content_aware_model
from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
from models.ppo_trainer import PPOTrainer
from models.logger import TrainingLogger

# ⬅️ نسخه‌ی بالانس‌شده‌ی پاداش
from models.reward_composite import compute_composite_reward

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- Safety Wrapper (بدون تغییر ppo_trainer.py) ----------
class SafetyEnvWrapper:
    """A thin wrapper to clamp risky actions when buffer is low."""
    def __init__(self, env):
        self.env = env
        # proxy attributes needed by PPOTrainer
        for k in ["reset", "get_state", "get_network_state", "get_vmaf_predictions",
                  "get_content_state", "bitrate_levels", "total_chunks", "mode",
                  "fcc_trace_loader"]:
            if hasattr(env, k):
                setattr(self, k, getattr(env, k))

    def step(self, action: int):
        # read last buffer from internal state representation (env.buffer in seconds)
        buf = getattr(self.env, "buffer", 0.0)
        a = int(action)
        if buf < 4:
            a = min(a, 1)
        elif buf < 8:
            a = min(a, 2)
        elif buf < 12:
            a = min(a, 3)


        return self.env.step(a)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

# ---------------------------------------------------------------

def _to_int_action(a):
    if isinstance(a, tuple):
        a = a[0]
    if isinstance(a, torch.Tensor):
        a = int(a.item())
    return int(a)

def main():
    print("🚀 Training Content-Aware ABR Agent (Balanced Composite Reward + Safety)")
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

    # === Environments ===
    env_train_raw = ContentAwareEnvFCC(
        loader, 'data/features/si_ti_features.json',
        'data/vmaf/vmaf_table.json', 'data/videos', mode='train'
    )
    env_val = ContentAwareEnvFCC(
        loader, 'data/features/si_ti_features.json',
        'data/vmaf/vmaf_table.json', 'data/videos', mode='val'
    )

    # Safety wrapper فقط روی train (val بدون safety برای دیدن رفتار خالص)
    env_train = SafetyEnvWrapper(env_train_raw)

    # === Model & Trainer ===
    model = create_content_aware_model().to(device)
    trainer = PPOTrainer(
        model=model,
        env=env_train,
        lr=1.8e-4,          # ← کمی کمتر برای پایداری نهایی
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.025, # ← کمی بیشتر برای جلوگیری از premature convergence
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64
    )

    trainer.device = device

    logger = TrainingLogger(log_dir='results/logs', run_name='composite_reward_balanced')
    trainer.external_logger = logger

    print("🎓 Training initialized.\n")

    # === Adaptive hyperparameters ===
    base_lr = 2e-4
    base_entropy = 0.02
    best_val_reward = float('-inf')
    patience = 35
    stop_counter = 0

    val_history = []

    # === Training Loop ===
    for update in range(1, 301):  # اجازه‌ی updates بیشتر برای رسیدن به سقف عملکرد
        rollout = trainer.collect_rollout(n_steps=4096)  # ⬅️ طولانی‌تر از قبل

        # Apply balanced composite reward over rollout infos
        infos = getattr(env_train.env, 'info_history', []) if hasattr(env_train.env, 'info_history') else getattr(env_train, 'info_history', [])
        last_bitrate = None
        n = min(len(rollout.rewards), len(infos))
        for i in range(n):
            rollout.rewards[i] = compute_composite_reward(infos[i], last_bitrate)
            last_bitrate = infos[i].get('bitrate', last_bitrate)

        # LR & Entropy decay
        current_lr = max(8e-5, base_lr * (0.996 ** update))
        for g in trainer.optimizer.param_groups:
            g['lr'] = current_lr
        trainer.entropy_coef = max(0.010, base_entropy * (0.995 ** update))  # ⬅️ حداقل 0.01

        train_info = trainer.update_policy(rollout) or {}
        policy_loss = float(train_info.get("policy_loss", 0.0))

        # === Validation (بدون safety تا تاثیر یادگیری واقعی دیده شود) ===
        val_rewards = []
        for _ in range(6):
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
                total_r += compute_composite_reward(info, last_bitrate)
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
            plt.title("Composite Reward Training Progress (Balanced + Safety)")
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
