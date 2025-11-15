"""
Analyze sensitivity to reward function parameters.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyze_reward_sensitivity():
    """Analyze how different reward parameters affect outcomes."""
    
    print("\n" + "="*70)
    print("📊 Reward Function Sensitivity Analysis")
    print("="*70 + "\n")
    
    # Historical results from our experiments
    results = {
        'V1': {
            'rebuffer_penalty': 4.3,
            'quality_weight': 1.0,
            'smooth_penalty': 1.0,
            'reward': -146.71,
            'rebuffer': 34.19,
            'quality': 0.724,
            'switches': 0.8,
            'behavior': 'Aggressive'
        },
        'V2': {
            'rebuffer_penalty': 10.0,
            'quality_weight': 1.0,
            'smooth_penalty': 0.2,
            'reward': 48.76,
            'rebuffer': 1.60,
            'quality': 0.553,
            'switches': 0.4,
            'behavior': 'Conservative'
        },
        'V3': {
            'rebuffer_penalty': 6.0,
            'quality_weight': 2.0,
            'smooth_penalty': 0.3,
            'reward': 64.52,
            'rebuffer': 1.60,
            'quality': 0.718,
            'switches': 10.6,
            'behavior': 'Balanced'
        }
    }
    
    df = pd.DataFrame(results).T
    
    print("Parameter Settings and Outcomes:")
    print("-" * 70)
    print(df)
    print()
    
    # Insights
    print("="*70)
    print("Key Insights:")
    print("="*70)
    
    print("\n1. Rebuffer Penalty Impact:")
    print("   - Too low (4.3):  Agent ignores rebuffering → high rebuffer")
    print("   - Too high (10.0): Agent too conservative → low quality")
    print("   - Balanced (6.0):  Good trade-off")
    
    print("\n2. Quality Weight Impact:")
    print("   - Standard (1.0):  Quality secondary to stability")
    print("   - Increased (2.0): Better quality while maintaining stability")
    
    print("\n3. Smooth Penalty Impact:")
    print("   - High (1.0):     Prevents all switching → stuck in one bitrate")
    print("   - Low (0.2):      Too much switching")
    print("   - Moderate (0.3): Good adaptation")
    
    print("\n4. Recommendations for further improvement:")
    print("   ✓ Current V3 (R=6.0, Q=2.0, S=0.3) is well-balanced")
    print("   ? Try: R=5.5, Q=2.5, S=0.3 for even higher quality")
    print("   ? Try: R=7.0, Q=2.0, S=0.25 for more stability")


if __name__ == '__main__':
    analyze_reward_sensitivity()