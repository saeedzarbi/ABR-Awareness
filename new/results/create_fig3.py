import sys
from pathlib import Path
# Add new/ to path (src and configs live there)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_multi_env import ABREnv
from src.baselines.mpc_vmaf import RobustMPC
from configs.paths import get_paths
import matplotlib.pyplot as plt
import numpy as np

PATHS = get_paths()

def run_episode_and_log(env, model, model_name):
    obs, info = env.reset(seed=42) # Seed ثابت برای مقایسه عادلانه روی یک شبکه یکسان
    done = False
    
    log = {'chunk': [], 'bitrate': [], 'buffer': [], 'throughput': []}
    last_tp = 2000.0
    
    while not done:
        trace_tp = env.current_trace['throughput_kbps']
        
        if model_name == 'RobustMPC':
            cur_vmaf = getattr(env, 'last_vmaf', 35.0)
            action = model.select_bitrate(info['buffer_level'], last_tp, cur_vmaf)
        else:
            action, _ = model.predict(obs, deterministic=True)
            
        obs, reward, done, _, info = env.step(action)
        last_tp = info.get('throughput', last_tp)
        
        log['chunk'].append(env.chunk_idx)
        log['bitrate'].append(env.BITRATE_LEVELS[action])
        log['buffer'].append(info['buffer_level'])
        log['throughput'].append(last_tp)
        
    return log

# 1. Setup Environment
env = ABREnv(
    video_names=['sintel'],
    trace_dir=str(PATHS['test_traces']),
    vmaf_dir=str(PATHS['vmaf_scores']),
    siti_dir=str(PATHS['content_features']),
    max_chunks=48, random_seed=42
)

# 2. Load Models
path_proposed = PATHS['models'] / 'ppo_proposed_v2_fresh' / 'best_model' / 'best_model.zip'
if not path_proposed.exists():
    path_proposed = PATHS['models'] / 'ppo_proposed_v2_fresh' / 'final_model.zip'

model_proposed = PPO.load(str(path_proposed))
model_mpc = RobustMPC(env)

print("🏃 Running Proposed Model...")
log_proposed = run_episode_and_log(env, model_proposed, 'Proposed')

print("🏃 Running RobustMPC...")
# Reset environment with same seed to ensure exact same network trace
env.reset(seed=42)
log_mpc = run_episode_and_log(env, model_mpc, 'RobustMPC')

# 3. Plotting
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

chunks = log_proposed['chunk']

# Subplot 1: Network Throughput
ax1.plot(chunks, log_proposed['throughput'], color='gray', linestyle='--', linewidth=2, label='Available Throughput')
ax1.set_ylabel('Throughput (kbps)', fontweight='bold')
ax1.legend(loc='upper right')
ax1.set_title('Agent Behavior Comparison: Proposed vs. RobustMPC', fontsize=14, fontweight='bold')

# Subplot 2: Bitrate Selection
ax2.step(chunks, log_mpc['bitrate'], where='post', color='#e74c3c', linewidth=2, alpha=0.8, label='RobustMPC (Chaotic)')
ax2.step(chunks, log_proposed['bitrate'], where='post', color='#2ecc71', linewidth=3, label='Proposed (Stable)')
ax2.set_ylabel('Selected Bitrate (kbps)', fontweight='bold')
ax2.set_yticks(env.BITRATE_LEVELS)
ax2.legend(loc='lower right')

# Subplot 3: Buffer Level
ax3.plot(chunks, log_mpc['buffer'], color='#e74c3c', linewidth=2, alpha=0.8, label='RobustMPC')
ax3.plot(chunks, log_proposed['buffer'], color='#2ecc71', linewidth=3, label='Proposed')
ax3.axhline(y=15.0, color='blue', linestyle=':', label='Target Buffer ($B_{ref}$)')
ax3.set_ylabel('Buffer Level (s)', fontweight='bold')
ax3.set_xlabel('Video Chunk Index', fontweight='bold')
ax3.legend(loc='upper right')
ax3.set_ylim(0, 32)

plt.tight_layout()
script_dir = Path(__file__).resolve().parent
plt.savefig(script_dir / 'fig_7_time_series.png', dpi=300)
print("✅ Saved: fig_7_time_series.png")