# train_auto_optimal.py
# نسخه عمومی آموزش + تحلیل خودکار n_updates بهینه

import os
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# ماژول‌های پروژه
from models.content_aware_model import create_content_aware_model
from models.content_aware_env_fcc_seeded import ContentAwareEnvFCC
from models.trace_loader import TraceLoader
from models.fcc_trace_loader import FCCTraceLoader

# =========================
# تنظیمات اولیه
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🧠 Using device: {device}\n")

# =========================
# پیکربندی عمومی
# =========================
config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.1,
    'value_coef': 0.5,
    'max_grad_norm': 0.5,
    'batch_size': 128,
    'ppo_epochs': 5,
    'rollout_steps': 2048,
    'n_updates': 400,
    'eval_interval': 10,
    'checkpoint_interval': 25,
    'log_interval': 5,
    'early_stopping_patience': 6,
    'early_stopping_min_delta': 0.2
}

# =========================
# توابع کمکی
# =========================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_value = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

def collect_rollout(env, model, n_steps):
    rollout = {'states': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': [], 'dones': []}
    state = env.reset()
    for step in range(n_steps):
        net = torch.FloatTensor(state['network']).unsqueeze(0).to(device)
        cont = torch.FloatTensor(state['content']).unsqueeze(0).to(device)
        vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0).to(device)

        with torch.no_grad():
            action_probs, value = model(net, cont, vmaf)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done, _ = env.step(action.item())
        rollout['states'].append(state)
        rollout['actions'].append(action.item())
        rollout['rewards'].append(reward)
        rollout['values'].append(value.item())
        rollout['log_probs'].append(log_prob.item())
        rollout['dones'].append(done)

        state = next_state if not done else env.reset()

    return rollout

def ppo_update(model, optimizer, rollout, config):
    advantages, returns = compute_gae(rollout['rewards'], rollout['values'], rollout['dones'], config['gamma'], config['gae_lambda'])
    advantages = (np.array(advantages) - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    old_log_probs = torch.FloatTensor(rollout['log_probs']).to(device)
    returns = torch.FloatTensor(returns).to(device)
    advantages = torch.FloatTensor(advantages).to(device)

    n_samples = len(rollout['states'])
    batch_size = config['batch_size']
    policy_losses, value_losses, entropies = [], [], []

    for epoch in range(config['ppo_epochs']):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            batch_states = [rollout['states'][i] for i in batch_idx]
            batch_actions = torch.LongTensor([rollout['actions'][i] for i in batch_idx]).to(device)
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]

            batch_net = torch.stack([torch.FloatTensor(s['network']) for s in batch_states]).to(device)
            batch_cont = torch.stack([torch.FloatTensor(s['content']) for s in batch_states]).to(device)
            batch_vmaf = torch.stack([torch.FloatTensor(s['vmaf']) for s in batch_states]).to(device)

            action_probs, values = model(batch_net, batch_cont, batch_vmaf)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(batch_actions)

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - config['clip_epsilon'], 1 + config['clip_epsilon']) * batch_advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
            entropy = dist.entropy().mean()

            loss = policy_loss + config['value_coef'] * value_loss - config['entropy_coef'] * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())

    return {'policy_loss': np.mean(policy_losses), 'value_loss': np.mean(value_losses), 'entropy': np.mean(entropies)}

def evaluate(env, model, n_episodes=20):
    rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        done, episode_reward = False, 0
        while not done:
            net = torch.FloatTensor(state['network']).unsqueeze(0).to(device)
            cont = torch.FloatTensor(state['content']).unsqueeze(0).to(device)
            vmaf = torch.FloatTensor(state['vmaf']).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs, _ = model(net, cont, vmaf)
            action = action_probs.argmax(dim=1).item()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards), np.std(rewards)

# =========================
# تابع اصلی
# =========================
def main(dataset):
    base_dir = f"results/{dataset}_training_auto"
    os.makedirs(base_dir, exist_ok=True)
    log_file = os.path.join(base_dir, 'training_log.json')

    # Dataset setup
    if dataset == 'fcc':
        loader = FCCTraceLoader(
            fcc_trace_dir='data/network_traces/fcc',
            train_file='data/network_traces/fcc/splits/fcc_train.txt',
            val_file='data/network_traces/fcc/splits/fcc_val.txt',
            test_file='data/network_traces/fcc/splits/fcc_test.txt'
        )
        env_train = ContentAwareEnvFCC(loader, 'data/features/si_ti_features.json', 'data/vmaf/vmaf_table.json', 'data/videos', mode='train')
        env_val = ContentAwareEnvFCC(loader, 'data/features/si_ti_features.json', 'data/vmaf/vmaf_table.json', 'data/videos', mode='val')
    elif dataset == 'cooked':
        loader = TraceLoader(trace_dir='data/network_traces/cooked_traces')
        env_train = ContentAwareEnvFCC(loader, 'data/features/si_ti_features.json', 'data/vmaf/vmaf_table.json', 'data/videos', mode='train')
        env_val = ContentAwareEnvFCC(loader, 'data/features/si_ti_features.json', 'data/vmaf/vmaf_table.json', 'data/videos', mode='val')
    else:
        raise ValueError("Dataset must be 'fcc' or 'cooked'")

    model = create_content_aware_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    writer = SummaryWriter(f"{base_dir}/tensorboard")

    print(f"\n🚀 Training {dataset.upper()} started at {datetime.now().strftime('%H:%M:%S')}\n")

    training_log, best_val_reward, no_improve = [], -float('inf'), 0

    for update in range(1, config['n_updates'] + 1):
        rollout = collect_rollout(env_train, model, config['rollout_steps'])
        info = ppo_update(model, optimizer, rollout, config)
        mean_reward = np.mean(rollout['rewards'])

        log_entry = {'update': update, **info, 'mean_reward': mean_reward}

        if update % config['eval_interval'] == 0:
            val_mean, val_std = evaluate(env_val, model)
            log_entry['val_reward_mean'], log_entry['val_reward_std'] = val_mean, val_std

            if val_mean - best_val_reward > config['early_stopping_min_delta']:
                best_val_reward = val_mean
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(base_dir, 'best_model.pth'))
            else:
                no_improve += 1

            if no_improve >= config['early_stopping_patience']:
                print(f"\n⏸️ Early stopping at update {update} (Best: {best_val_reward:.2f})")
                break

        training_log.append(log_entry)
        writer.add_scalar('train/reward', mean_reward, update)
        if 'val_reward_mean' in log_entry:
            writer.add_scalar('val/reward', log_entry['val_reward_mean'], update)

        if update % config['log_interval'] == 0:
            msg = f"[{update:03d}] TrainR: {mean_reward:+7.2f}"
            if 'val_reward_mean' in log_entry:
                msg += f" | ValR: {log_entry['val_reward_mean']:+7.2f}"
            print(msg)

        if update % 10 == 0:
            with open(log_file, 'w') as f: json.dump(training_log, f, indent=2)

    # =========================
    # تحلیل خودکار n_updates بهینه
    # =========================
    print("\n📊 Analyzing optimal update count...")
    import pandas as pd
    log = pd.DataFrame(training_log)
    if 'val_reward_mean' not in log.columns:
        print("⚠️ No validation rewards recorded. Skipping analysis.")
        return

    log['ma'] = log['val_reward_mean'].rolling(window=5).mean()
    slopes = log['ma'].diff()
    stable_points = log.loc[slopes.abs() < 0.05, 'update']
    optimal = int(stable_points.iloc[0]) if not stable_points.empty else log['update'].iloc[-1]

    print(f"✅ Optimal n_updates ≈ {optimal} (Best Val Reward: {best_val_reward:+.2f})")

    plt.figure(figsize=(8,4))
    plt.plot(log['update'], log['val_reward_mean'], label='Validation Reward')
    plt.axvline(optimal, color='r', linestyle='--', label=f'Optimal={optimal}')
    plt.xlabel('Update'); plt.ylabel('Val Reward'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(base_dir, 'val_reward_curve.png'))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fcc', help="fcc or cooked")
    args = parser.parse_args()
    main(args.dataset)
