"""
scripts/evaluation/evaluate_ablation.py
========================================
Evaluate and compare all ablation models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

from models.ablation_models import load_ablated_model
from models.content_aware_env_v2 import ContentAwareEnvV2
from models.trace_loader import TraceLoader
from models.policy_wrapper import BufferAwarePolicy, SmoothPolicy

print("="*100)
print("🧪 ABLATION STUDY - Complete Evaluation")
print("="*100)

# ============================================
# Configuration
# ============================================
CONFIG = {
    'trace_dir': 'data/network_traces/cooked_traces',
    'features_file': 'data/features/si_ti_features.json',
    'vmaf_file': 'data/vmaf/vmaf_table.json',
    'episodes': 60,  # Same as your per-video test
    
    'models': {
        'Full Model': {
            'type': 'full',
            'checkpoint': 'results/fcc_training_low_lr/checkpoint_best.pth',
            'description': 'Network + SI/TI + VMAF'
        },
        'No SI/TI': {
            'type': 'no_siti',
            'checkpoint': 'results/ablation_no_siti/checkpoint_best.pth',
            'description': 'Network + VMAF only'
        },
        'No VMAF': {
            'type': 'no_vmaf',
            'checkpoint': 'results/ablation_no_vmaf/checkpoint_best.pth',
            'description': 'Network + SI/TI only'
        },
        'Network Only': {
            'type': 'network_only',
            'checkpoint': 'results/ablation_network_only/checkpoint_best.pth',
            'description': 'Network only (like Pensieve)'
        }
    }
}

# ============================================
# Helper Function
# ============================================
def evaluate_model(model, env, num_episodes, model_name, DEVICE):
    """Evaluate one model"""
    
    # Wrap with policy wrapper
    buffer_policy = BufferAwarePolicy(model)
    policy = SmoothPolicy(buffer_policy, max_jump=2)
    
    rewards = []
    rebuffers = []
    bitrates = []
    
    for ep in tqdm(range(num_episodes), desc=f"Eval ({model_name})"):
        policy.reset()
        state = env.reset(split='test')
        if state is None:
            continue
        
        ep_reward = 0
        ep_rebuffer = 0
        ep_bitrates = []
        done = False
        recent_rebuffer = 0.0
        
        while not done:
            action = policy.select_action(state, env.buffer, recent_rebuffer)
            state, reward, done, info = env.step(action)
            
            recent_rebuffer = info['rebuffer_time']
            ep_reward += reward
            ep_rebuffer += info['rebuffer_time']
            ep_bitrates.append(info['bitrate'])
        
        rewards.append(ep_reward)
        rebuffers.append(ep_rebuffer)
        bitrates.append(np.mean(ep_bitrates) if ep_bitrates else 0)
    
    return {
        'rewards': rewards,
        'rebuffers': rebuffers,
        'bitrates': bitrates,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_rebuffer': np.mean(rebuffers),
        'std_rebuffer': np.std(rebuffers),
        'mean_bitrate': np.mean(bitrates),
        'std_bitrate': np.std(bitrates)
    }

# ============================================
# Setup
# ============================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Device: {DEVICE}")

# Setup environment
print("\n🌍 Setting up environment...")
trace_dir = CONFIG['trace_dir']
loader = TraceLoader(trace_dir=trace_dir)

env = ContentAwareEnvV2(
    trace_dir=trace_dir,
    features_file=CONFIG['features_file'],
    vmaf_file=CONFIG['vmaf_file']
)

num_test_traces = len(loader.test_traces)
print(f"   ✅ Loaded {num_test_traces} test traces")
print(f"   Episodes per model: {CONFIG['episodes']}")

# ============================================
# Evaluate All Models
# ============================================
results = {}

for model_name, model_config in CONFIG['models'].items():
    print(f"\n{'='*100}")
    print(f"📊 Evaluating: {model_name}")
    print(f"   {model_config['description']}")
    print(f"{'='*100}")
    
    try:
        # Load model
        model = load_ablated_model(
            ablation_type=model_config['type'],
            checkpoint_path=model_config['checkpoint'],
            device=DEVICE
        )
        model.eval()
        
        # Evaluate
        result = evaluate_model(
            model, env, CONFIG['episodes'], model_name, DEVICE
        )
        results[model_name] = result
        
        # Print results
        print(f"\n   Results:")
        print(f"      Mean Reward:      {result['mean_reward']:+.2f} ± {result['std_reward']:.2f}")
        print(f"      Mean Rebuffering: {result['mean_rebuffer']:.2f}s ± {result['std_rebuffer']:.2f}s")
        print(f"      Mean Bitrate:     {result['mean_bitrate']:.0f} ± {result['std_bitrate']:.0f} kbps")
        print(f"   ✅ Completed")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   ⚠️  Skipping {model_name}")

# ============================================
# Comparison Table
# ============================================
print("\n" + "="*100)
print("📊 ABLATION STUDY RESULTS")
print("="*100)

# Create DataFrame
summary_data = []
for model_name, result in results.items():
    summary_data.append({
        'Model': model_name,
        'Reward': result['mean_reward'],
        'Std': result['std_reward'],
        'Rebuffer(s)': result['mean_rebuffer'],
        'Bitrate(kbps)': result['mean_bitrate']
    })

df = pd.DataFrame(summary_data)
df = df.sort_values('Reward', ascending=False)

print(df.to_string(index=False))
print("="*100)

# ============================================
# Component Contribution Analysis
# ============================================
if 'Full Model' in results:
    print("\n" + "="*100)
    print("💡 COMPONENT CONTRIBUTION ANALYSIS")
    print("="*100)
    
    full_reward = results['Full Model']['mean_reward']
    
    contributions = []
    
    # SI/TI contribution
    if 'No SI/TI' in results:
        no_siti_reward = results['No SI/TI']['mean_reward']
        siti_contribution = full_reward - no_siti_reward
        siti_percent = (siti_contribution / full_reward) * 100
        contributions.append({
            'Component': 'SI/TI Features',
            'Contribution': siti_contribution,
            'Percentage': siti_percent
        })
        print(f"\n✅ SI/TI Features Contribution:")
        print(f"   Full Model:     {full_reward:+.2f}")
        print(f"   Without SI/TI:  {no_siti_reward:+.2f}")
        print(f"   Δ Contribution: {siti_contribution:+.2f} ({siti_percent:+.1f}%)")
    
    # VMAF contribution
    if 'No VMAF' in results:
        no_vmaf_reward = results['No VMAF']['mean_reward']
        vmaf_contribution = full_reward - no_vmaf_reward
        vmaf_percent = (vmaf_contribution / full_reward) * 100
        contributions.append({
            'Component': 'VMAF Predictions',
            'Contribution': vmaf_contribution,
            'Percentage': vmaf_percent
        })
        print(f"\n✅ VMAF Predictions Contribution:")
        print(f"   Full Model:     {full_reward:+.2f}")
        print(f"   Without VMAF:   {no_vmaf_reward:+.2f}")
        print(f"   Δ Contribution: {vmaf_contribution:+.2f} ({vmaf_percent:+.1f}%)")
    
    # Combined content features contribution
    if 'Network Only' in results:
        network_only_reward = results['Network Only']['mean_reward']
        content_contribution = full_reward - network_only_reward
        content_percent = (content_contribution / full_reward) * 100
        contributions.append({
            'Component': 'All Content Features',
            'Contribution': content_contribution,
            'Percentage': content_percent
        })
        print(f"\n✅ All Content Features (SI/TI + VMAF) Contribution:")
        print(f"   Full Model:     {full_reward:+.2f}")
        print(f"   Network Only:   {network_only_reward:+.2f}")
        print(f"   Δ Contribution: {content_contribution:+.2f} ({content_percent:+.1f}%)")
    
    print("\n" + "="*100)
    
    # Contribution summary table
    if contributions:
        contrib_df = pd.DataFrame(contributions)
        print("\n📊 Component Contribution Summary:")
        print(contrib_df.to_string(index=False))

# ============================================
# Statistical Significance Tests
# ============================================
if len(results) >= 2:
    print("\n" + "="*100)
    print("📈 STATISTICAL SIGNIFICANCE TESTS")
    print("="*100)
    
    # t-test between Full Model and each ablation
    if 'Full Model' in results:
        full_rewards = results['Full Model']['rewards']
        
        for model_name, result in results.items():
            if model_name == 'Full Model':
                continue
            
            other_rewards = result['rewards']
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(full_rewards, other_rewards)
            
            # Interpret
            if p_value < 0.001:
                sig = "*** (p<0.001) - Highly Significant"
            elif p_value < 0.01:
                sig = "** (p<0.01) - Very Significant"
            elif p_value < 0.05:
                sig = "* (p<0.05) - Significant"
            else:
                sig = "n.s. (p≥0.05) - Not Significant"
            
            print(f"\nFull Model vs {model_name}:")
            print(f"   t-statistic: {t_stat:.3f}")
            print(f"   p-value: {p_value:.6f}")
            print(f"   Result: {sig}")

# ============================================
# Save Results
# ============================================
print("\n" + "="*100)
print("💾 Saving Results")
print("="*100)

# Save summary
df.to_csv('results/ablation_study_summary.csv', index=False)
print("   ✅ Saved: results/ablation_study_summary.csv")

# Save detailed results
detailed_results = {}
for model_name, result in results.items():
    detailed_results[model_name] = {
        'mean_reward': float(result['mean_reward']),
        'std_reward': float(result['std_reward']),
        'mean_rebuffer': float(result['mean_rebuffer']),
        'mean_bitrate': float(result['mean_bitrate']),
        'all_rewards': [float(x) for x in result['rewards']],
        'all_rebuffers': [float(x) for x in result['rebuffers']],
        'all_bitrates': [float(x) for x in result['bitrates']]
    }

import json
with open('results/ablation_study_detailed.json', 'w') as f:
    json.dump(detailed_results, f, indent=2)
print("   ✅ Saved: results/ablation_study_detailed.json")

# ============================================
# Final Summary
# ============================================
print("\n" + "="*100)
print("✅ ABLATION STUDY COMPLETE!")
print("="*100)

print("\n📊 Key Findings:")
if 'Full Model' in results and 'Network Only' in results:
    improvement = results['Full Model']['mean_reward'] - results['Network Only']['mean_reward']
    percent = (improvement / results['Network Only']['mean_reward']) * 100
    print(f"   • Content-awareness improves performance by {improvement:+.2f} points ({percent:+.1f}%)")

if 'Full Model' in results and 'No SI/TI' in results:
    siti_impact = results['Full Model']['mean_reward'] - results['No SI/TI']['mean_reward']
    print(f"   • SI/TI features contribute {siti_impact:+.2f} points")

if 'Full Model' in results and 'No VMAF' in results:
    vmaf_impact = results['Full Model']['mean_reward'] - results['No VMAF']['mean_reward']
    print(f"   • VMAF predictions contribute {vmaf_impact:+.2f} points")

print("\n📄 Files saved:")
print("   - results/ablation_study_summary.csv")
print("   - results/ablation_study_detailed.json")

print("\n" + "="*100)