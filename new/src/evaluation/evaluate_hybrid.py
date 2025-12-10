"""
Complete Evaluation: Hybrid vs All Baselines
Compare trained hybrid model with RobustMPC, BBA, Genie, Pensieve
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from src.environment.abr_env_simple import ABREnvSimple
from configs.paths import get_paths

PATHS = get_paths()

# ============================================================================
# Baseline Algorithms
# ============================================================================

class RobustMPC:
    """RobustMPC baseline"""
    BITRATE_LEVELS = [300, 750, 1200, 1850, 2850, 6000]
    
    def __init__(self, env):
        self.env = env
    
    def predict_throughput(self, history):
        if len(history) == 0:
            return 3000
        tp_kbps = [tp * 6000 for tp in history[-5:]]
        return len(tp_kbps) / sum(1.0/(tp+1e-6) for tp in tp_kbps)
    
    def select_action(self, obs):
        tp_history = obs[:8].tolist()
        buffer = obs[8] * 30.0
        
        pred_tp = self.predict_throughput(tp_history)
        
        best_action = 0
        best_score = -float('inf')
        
        for action in range(len(self.BITRATE_LEVELS)):
            bitrate = self.BITRATE_LEVELS[action]
            vmaf = self.env.current_vmaf_scores.get(bitrate, 35)
            
            download_time = (bitrate * 1000 * 4.0) / (pred_tp * 1000 + 1e-6)
            rebuffer = max(0, download_time - buffer)
            
            score = vmaf - (4.3 * rebuffer)
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

class BBA:
    """Buffer-Based Adaptation"""
    BITRATE_LEVELS = [300, 750, 1200, 1850, 2850, 6000]
    RESERVOIR = 5.0
    CUSHION = 10.0
    
    def select_action(self, obs):
        buffer = obs[8] * 30.0
        
        if buffer < self.RESERVOIR:
            return 0
        elif buffer >= self.RESERVOIR + self.CUSHION:
            return len(self.BITRATE_LEVELS) - 1
        else:
            ratio = (buffer - self.RESERVOIR) / self.CUSHION
            index = int(ratio * (len(self.BITRATE_LEVELS) - 1))
            return min(index, len(self.BITRATE_LEVELS) - 1)

class Genie:
    """Oracle with perfect future knowledge"""
    BITRATE_LEVELS = [300, 750, 1200, 1850, 2850, 6000]
    
    def __init__(self):
        self.env = None
    
    def select_action(self, obs, env=None):
        # Use provided env
        if env is None:
            # Fallback: use throughput from obs
            tp_history = obs[:8]
            avg_tp = np.mean(tp_history) * 6000
            
            for action in reversed(range(len(self.BITRATE_LEVELS))):
                bitrate = self.BITRATE_LEVELS[action]
                if bitrate < avg_tp * 0.85:
                    return action
            return 0
        
        # Get current trace
        if env.current_trace is None:
            return 2  # Default middle bitrate
        
        trace_tp = env.current_trace['throughput_kbps']
        chunk_idx = env.chunk_idx
        
        # Look at next 5 chunks
        future_tp = []
        for i in range(5):
            idx = int((chunk_idx + i) * 4.0) % len(trace_tp)
            future_tp.append(trace_tp[idx])
        
        avg_future_tp = np.mean(future_tp)
        
        # Select bitrate safely below average throughput
        for action in reversed(range(len(self.BITRATE_LEVELS))):
            bitrate = self.BITRATE_LEVELS[action]
            if bitrate < avg_future_tp * 0.85:
                return action
        
        return 0

# ============================================================================
# Evaluator
# ============================================================================

class HybridEvaluator:
    """Evaluate hybrid and baselines"""
    
    def __init__(self):
        self.test_videos = ['bigbuckbunny', 'parkjoy', 'tearsofsteel_short']
        self.results = []
    
    def evaluate_method(self, method_name, method, episodes=50):
        """Evaluate a single method"""
        
        print(f"\n{'='*70}")
        print(f"🔬 Evaluating: {method_name}")
        print(f"{'='*70}")
        
        env = ABREnvSimple(
            video_names=self.test_videos,
            trace_dir=str(PATHS['test_traces']),
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features']),
            max_chunks=48
        )
        
        episode_results = []
        
        for ep in tqdm(range(episodes), desc=method_name):
            obs, info = env.reset()
            
            ep_reward = 0
            ep_vmaf = 0
            ep_rebuffer = 0
            ep_switches = 0
            last_action = 0
            steps = 0
            
            done = False
            
            while not done:
                # Get action
                if method_name == 'Hybrid':
                    action, _ = method.predict(obs, deterministic=True)
                elif method_name == 'Genie':
                    action = method.select_action(obs, env)
                elif method_name == 'RobustMPC':
                    action = method.select_action(obs)
                else:  # BBA
                    action = method.select_action(obs)
                
                # Step
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                
                # Metrics
                vmaf = env.current_vmaf_scores[env.BITRATE_LEVELS[action]]
                ep_reward += reward
                ep_vmaf += vmaf
                ep_rebuffer += step_info.get('rebuffer', 0)
                
                if action != last_action:
                    ep_switches += 1
                
                last_action = action
                steps += 1
            
            # Save episode
            episode_results.append({
                'method': method_name,
                'episode': ep + 1,
                'video': env.current_video_name,
                'reward': ep_reward,
                'avg_vmaf': ep_vmaf / steps,
                'total_rebuffer': ep_rebuffer,
                'rebuffer_pct': (ep_rebuffer / (steps * 4.0)) * 100,
                'switches': ep_switches,
                'steps': steps
            })
        
        self.results.extend(episode_results)
        
        # Summary
        df = pd.DataFrame(episode_results)
        print(f"\n📊 {method_name} Summary:")
        print(f"  Avg VMAF:     {df['avg_vmaf'].mean():.2f} (±{df['avg_vmaf'].std():.2f})")
        print(f"  Avg Rebuffer: {df['rebuffer_pct'].mean():.2f}%")
        print(f"  Avg Switches: {df['switches'].mean():.1f}")
        print(f"  Avg Reward:   {df['reward'].mean():.2f}")
        
        return df
    
    def run_all(self, hybrid_model_path, episodes=50):
        """Run evaluation on all methods"""
        
        print("\n" + "="*70)
        print("🚀 COMPLETE EVALUATION: Hybrid vs Baselines")
        print("="*70)
        print(f"Videos: {self.test_videos}")
        print(f"Episodes per method: {episodes}")
        print()
        
        # Create env for baselines
        env = ABREnvSimple(
            video_names=self.test_videos,
            trace_dir=str(PATHS['test_traces']),
            vmaf_dir=str(PATHS['vmaf_scores']),
            siti_dir=str(PATHS['content_features'])
        )
        
        methods = {}
        
        # 1. Hybrid (our model)
        try:
            methods['Hybrid'] = PPO.load(hybrid_model_path)
            print("✅ Loaded Hybrid model")
        except Exception as e:
            print(f"❌ Failed to load Hybrid: {e}")
        
        # 2. RobustMPC
        methods['RobustMPC'] = RobustMPC(env)
        
        # 3. BBA
        methods['BBA'] = BBA()
        
        # 4. Genie (Oracle)
        methods['Genie'] = Genie()
        
        # Evaluate all
        all_results = {}
        for name, method in methods.items():
            df = self.evaluate_method(name, method, episodes)
            all_results[name] = df
        
        # Final comparison
        self.print_final_comparison(all_results)
        
        # Save results
        results_df = pd.DataFrame(self.results)
        output_file = 'hybrid_vs_baselines_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        
        return results_df
    
    def print_final_comparison(self, all_results):
        """Print final comparison table"""
        
        print("\n" + "="*70)
        print("📊 FINAL COMPARISON")
        print("="*70)
        
        summary = []
        for method_name, df in all_results.items():
            summary.append({
                'Method': method_name,
                'VMAF': f"{df['avg_vmaf'].mean():.2f}",
                'Rebuffer': f"{df['rebuffer_pct'].mean():.2f}%",
                'Switches': f"{df['switches'].mean():.1f}",
                'Reward': f"{df['reward'].mean():.1f}"
            })
        
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        
        # Highlight best
        print("\n🏆 Best Performance:")
        
        vmaf_scores = {row['Method']: float(row['VMAF']) for _, row in summary_df.iterrows()}
        rebuf_scores = {row['Method']: float(row['Rebuffer'].rstrip('%')) for _, row in summary_df.iterrows()}
        reward_scores = {row['Method']: float(row['Reward']) for _, row in summary_df.iterrows()}
        
        best_vmaf = max(vmaf_scores, key=vmaf_scores.get)
        best_rebuf = min(rebuf_scores, key=rebuf_scores.get)
        best_reward = max(reward_scores, key=reward_scores.get)
        
        print(f"  Best VMAF:     {best_vmaf} ({vmaf_scores[best_vmaf]:.2f})")
        print(f"  Best Rebuffer: {best_rebuf} ({rebuf_scores[best_rebuf]:.2f}%)")
        print(f"  Best Reward:   {best_reward} ({reward_scores[best_reward]:.1f})")
        
        print("="*70)

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate hybrid vs baselines')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to hybrid model')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Episodes per method')
    
    args = parser.parse_args()
    
    evaluator = HybridEvaluator()
    results = evaluator.run_all(args.model, args.episodes)