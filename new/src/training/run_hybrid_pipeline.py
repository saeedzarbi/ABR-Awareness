#!/usr/bin/env python3
"""
Master Script: Complete Hybrid Training Pipeline
Runs all 3 stages automatically
"""

import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd, description):
    """Run command and handle errors"""
    print("\n" + "="*70)
    print(f"🚀 {description}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        return False

def main():
    """Run complete pipeline"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  HYBRID TRAINING PIPELINE                             ║
║                                                                        ║
║  Stage 1: Collect Expert Data (RobustMPC)   ~2 hours                ║
║  Stage 2: Imitation Learning                 ~1 day                  ║
║  Stage 3: PPO Fine-tuning                    ~2-3 days               ║
║                                                                        ║
║  Total: ~3-4 days                                                     ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get paths from user
    print("Please provide paths:")
    trace_dir = input("Train traces directory: ").strip()
    vmaf_dir = input("VMAF directory: ").strip()
    siti_dir = input("SITI directory: ").strip()
    
    if not all([trace_dir, vmaf_dir, siti_dir]):
        print("❌ All paths are required!")
        sys.exit(1)
    
    print("\nStarting pipeline...")
    time.sleep(2)
    
    # ========================================================================
    # Stage 1: Collect Expert Data
    # ========================================================================
    
    stage1_cmd = [
        'python', 'collect_expert_data.py',
        '--episodes', '1000',
        '--videos', 'bigbuckbunny', 'tearsofsteel_short', 'parkjoy',
        '--trace-dir', trace_dir,
        '--vmaf-dir', vmaf_dir,
        '--siti-dir', siti_dir,
        '--output', 'expert_demonstrations.pkl'
    ]
    
    if not run_command(stage1_cmd, "Stage 1: Collecting Expert Data"):
        print("\n❌ Pipeline failed at Stage 1")
        sys.exit(1)
    
    # ========================================================================
    # Stage 2: Imitation Learning
    # ========================================================================
    
    stage2_cmd = [
        'python', 'train_imitation.py',
        '--data', 'expert_demonstrations.pkl',
        '--output', 'imitation_policy.pth',
        '--epochs', '50',
        '--batch-size', '256',
        '--device', 'cuda'
    ]
    
    if not run_command(stage2_cmd, "Stage 2: Imitation Learning"):
        print("\n❌ Pipeline failed at Stage 2")
        sys.exit(1)
    
    # ========================================================================
    # Stage 3: PPO Fine-tuning
    # ========================================================================
    
    stage3_cmd = [
        'python', 'train_hybrid.py',
        '--imitation-model', 'imitation_policy_sb3.pth',
        '--output-dir', 'ppo_hybrid'
    ]
    
    if not run_command(stage3_cmd, "Stage 3: PPO Fine-tuning"):
        print("\n❌ Pipeline failed at Stage 3")
        sys.exit(1)
    
    # ========================================================================
    # Success!
    # ========================================================================
    
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. expert_demonstrations.pkl    (Stage 1 output)")
    print("  2. imitation_policy.pth         (Stage 2 output)")
    print("  3. imitation_policy_sb3.pth     (Stage 2 SB3 format)")
    print("  4. ppo_hybrid/final_model.zip   (Stage 3 output)")
    print("\nNext steps:")
    print("  1. Evaluate: python final_multi.py --model ppo_hybrid/final_model")
    print("  2. Compare with baselines")
    print("  3. Analyze results")
    print("="*70)

if __name__ == '__main__':
    main()
