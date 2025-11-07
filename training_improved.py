import time

from models.content_aware_model import ContentAwareActor
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
import os



print("=" * 80)
print("🚀 IMPROVED TRAINING with Early Stopping")
print("=" * 80)
print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ═══════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════

checkpoint_dir = 'results/fcc_training_improved_c'
log_file = os.path.join(checkpoint_dir, 'training_log.json')
os.makedirs(checkpoint_dir, exist_ok=True)

config = {
    # ✅ اصلاح شده
    'learning_rate': 3e-4,              # برگشت به 3e-4
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
@@ -47,27 +47,14 @@
    'batch_size': 64,
    'ppo_epochs': 4,
    'rollout_steps': 2048,
    'n_updates': 400,                   # کمتر شد
    'eval_interval': 10,                # زودتر ارزیابی
    'checkpoint_interval': 25,          # بیشتر ذخیره
    'log_interval': 5,
    'early_stopping_patience': 3,       # جدید!
    'early_stopping_min_delta': 0.5     # حداقل بهبود
}

print("⚙️  Configuration:")
for key, val in config.items():
    print(f"   {key}: {val}")
print()
print(f"📁 Output: {checkpoint_dir}")
print(f"📝 Log file: {log_file}")
print()

# ═══════════════════════════════════════════════════════════
# Load Data
# ═══════════════════════════════════════════════════════════

print("📦 Loading Data...")
loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
@@ -91,22 +78,10 @@
    mode='val'
)

print(f"✅ Data loaded: Train={len(loader.train_traces)}, Val={len(loader.val_traces)}")
print()

# ═══════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════

print("🧠 Creating Model...")
model = create_content_aware_model()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
print(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
print()

# ═══════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
@@ -248,14 +223,6 @@ def evaluate(env, model, n_episodes=10):

    return np.mean(rewards), np.std(rewards)

# ═══════════════════════════════════════════════════════════
# Training Loop with Early Stopping
# ═══════════════════════════════════════════════════════════

print("🎯 Starting Training with Early Stopping...")
print("=" * 80)
print()

training_log = []
best_val_reward = -float('inf')
no_improvement_count = 0
@@ -315,34 +282,20 @@ def evaluate(env, model, n_episodes=10):

        # Check early stopping
        if no_improvement_count >= config['early_stopping_patience']:
            print()
            print("=" * 80)
            print("⏸️  EARLY STOPPING TRIGGERED")
            print("=" * 80)
            print(f"No improvement for {no_improvement_count} evaluations")
            print(f"Best val reward: {best_val_reward:+.2f}")
            print(f"Stopping at update {update}")
            print()
            break

    training_log.append(log_entry)

    # Console output
    if update % config['log_interval'] == 0:
        eta_seconds = (elapsed_time / update) * (config['n_updates'] - update)
        eta_minutes = eta_seconds / 60
        
        print(f"Update {update:3d}/{config['n_updates']} | "
              f"Reward: {mean_reward:+7.2f} | "
              f"Loss: {train_info['policy_loss']:.4f} | "
              f"Entropy: {train_info['entropy']:.3f} | "
              f"Time: {update_time:.1f}s | "
              f"ETA: {eta_minutes:.0f}min")

        if 'val_reward_mean' in log_entry:
            best_marker = "🏆" if log_entry.get('new_best') else "  "
            no_improve_marker = f"[{no_improvement_count}/{config['early_stopping_patience']}]"
            print(f"         {best_marker}  Val: {log_entry['val_reward_mean']:+.2f} ± {log_entry['val_reward_std']:.2f} {no_improve_marker}")

    # Save checkpoint
    if update % config['checkpoint_interval'] == 0:
@@ -354,96 +307,15 @@ def evaluate(env, model, n_episodes=10):
            'config': config,
            'train_info': train_info
        }, checkpoint_path)
        print(f"         💾 Checkpoint saved: checkpoint_{update}.pth")

    # Save log
    if update % 10 == 0:
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)

print()
print("=" * 80)
print("✅ Training Complete!")
print("=" * 80)
print(f"⏰ Total time: {(time.time() - start_time) / 60:.1f} minutes")
print(f"📁 Checkpoints: {checkpoint_dir}")
print(f"🏆 Best val reward: {best_val_reward:+.2f}")
print()

# Final save
with open(log_file, 'w') as f:
    json.dump(training_log, f, indent=2)

print("📝 Training log saved!")
print()

# ═══════════════════════════════════════════════════════════
# تست سریع بهترین checkpoint
# ═══════════════════════════════════════════════════════════

print("=" * 80)
print("🧪 Quick Test on Best Checkpoint")
print("=" * 80)
print()

# بارگذاری best model
try:
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded best checkpoint (update {checkpoint['update']})")
except:
    print("⚠️  No best checkpoint found")

# تست سریع
print("Running 5 test episodes...")
test_rewards = []

env_test = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='test' if len(loader.test_traces) >= 5 else 'val'
)

for i in range(5):
    state = env_test.reset()
    ep_reward = 0
    done = False
    
    while not done:
        net = torch.FloatTensor(state['network']).unsqueeze(0)
        cont = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        with torch.no_grad():
            probs, _ = model(net, cont, vmaf)
        
        # با safety wrapper
        action = probs.argmax(dim=1).item()
        buffer = env_test.buffer
        
        if buffer < 5.0:
            action = min(action, 1)
        elif buffer < 10.0:
            action = min(action, 2)
        elif buffer < 20.0:
            action = min(action, 3)
        
        state, reward, done, info = env_test.step(action)
        ep_reward += reward
    
    test_rewards.append(ep_reward)
    print(f"  Episode {i+1}: {ep_reward:+.2f}")

print()
print(f"Quick Test Result: {np.mean(test_rewards):+.2f} ± {np.std(test_rewards):.2f}")
print()

baseline = 102.16
if np.mean(test_rewards) > baseline * 0.9:
    print(f"✅ Promising! Close to or better than baseline ({baseline:+.2f})")
else:
    print(f"⚠️  Below baseline, but may improve with more testing")

print("=" * 80)