"""
Complete 2-Phase Training Pipeline
Run both phases automatically
"""

import os
import sys
from datetime import datetime

print("="*80)
print("🚀 COMPLETE 2-PHASE TRAINING PIPELINE")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("\nPhase 1: Behavioral Cloning (Imitation Learning)")
print("Phase 2: RL Fine-tuning (PPO)")
print("="*80)

# PHASE 1
print("\n" + "="*80)
print("Starting Phase 1...")
print("="*80)

try:
    exec(open('phase1_behavioral_cloning.py').read())
    print("\n✅ Phase 1 Complete!")
except Exception as e:
    print(f"\n❌ Phase 1 Failed: {str(e)}")
    sys.exit(1)

# PHASE 2
print("\n" + "="*80)
print("Starting Phase 2...")
print("="*80)

try:
    exec(open('phase2_rl_finetuning.py').read())
    print("\n✅ Phase 2 Complete!")
except Exception as e:
    print(f"\n❌ Phase 2 Failed: {str(e)}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ COMPLETE PIPELINE FINISHED")
print("="*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)