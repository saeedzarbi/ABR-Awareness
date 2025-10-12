#!/usr/bin/env python3
"""
Approximate VMAF scores based on content features and bitrate
"""

import json
import numpy as np
from pathlib import Path
import argparse

def approximate_vmaf(bitrate, si, ti, bitrate_levels=[300, 750, 1850, 2850, 4300, 6000]):
    """
    Approximate VMAF using empirical formula
    
    Based on research findings:
    - VMAF increases logarithmically with bitrate
    - Content complexity (SI/TI) reduces perceived quality
    - Higher complexity needs higher bitrate for same quality
    
    Args:
        bitrate: Bitrate in kbps
        si: Spatial Information
        ti: Temporal Information
        bitrate_levels: Available bitrate levels
        
    Returns:
        Approximate VMAF score (0-100)
    """
    # Normalize inputs
    max_bitrate = max(bitrate_levels)
    bitrate_norm = bitrate / max_bitrate
    
    # Normalize complexity (empirical values)
    si_norm = np.clip(si / 100.0, 0, 1)
    ti_norm = np.clip(ti / 50.0, 0, 1)
    complexity = (si_norm + ti_norm) / 2
    
    # Base quality from bitrate (logarithmic relationship)
    base_quality = 20 + 60 * np.log(1 + bitrate_norm * 10) / np.log(11)
    
    # Complexity penalty (higher complexity = need higher bitrate)
    complexity_penalty = 15 * complexity * (1 - bitrate_norm)
    
    # Bitrate efficiency bonus (diminishing returns at high bitrate)
    efficiency = bitrate_norm * (1 - 0.3 * bitrate_norm)
    efficiency_bonus = 10 * efficiency
    
    # Final VMAF
    vmaf = base_quality - complexity_penalty + efficiency_bonus
    vmaf = np.clip(vmaf, 0, 100)
    
    return float(vmaf)

def generate_vmaf_table(features_file, output_file, bitrate_levels=[300, 750, 1850, 2850, 4300, 6000]):
    """
    Generate VMAF lookup table for all chunks and bitrates
    """
    # Load features
    with open(features_file, 'r') as f:
        features = json.load(f)
    
    print(f"Loaded {len(features)} feature entries")
    
    vmaf_table = {}
    
    for key, feat in features.items():
        si = feat['si_mean']
        ti = feat['ti_mean']
        
        # Calculate VMAF for each bitrate level
        vmaf_scores = {}
        for br in bitrate_levels:
            vmaf = approximate_vmaf(br, si, ti, bitrate_levels)
            vmaf_scores[str(br)] = vmaf
        
        vmaf_table[key] = vmaf_scores
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(vmaf_table, f, indent=2)
    
    print(f"✓ VMAF table saved to {output_file}")
    print(f"  Total entries: {len(vmaf_table)}")
    
    # Statistics
    all_vmaf = []
    for entry in vmaf_table.values():
        all_vmaf.extend(entry.values())
    
    print(f"\nVMAF Statistics:")
    print(f"  Mean: {np.mean(all_vmaf):.2f}")
    print(f"  Min: {np.min(all_vmaf):.2f}")
    print(f"  Max: {np.max(all_vmaf):.2f}")
    print(f"  Std: {np.std(all_vmaf):.2f}")
    
    return vmaf_table

def main():
    parser = argparse.ArgumentParser(description='Generate VMAF approximations')
    parser.add_argument('--features', type=str,
                       default='data/features/si_ti_features.json',
                       help='Input features JSON file')
    parser.add_argument('--output', type=str,
                       default='data/vmaf/vmaf_table.json',
                       help='Output VMAF table JSON file')
    parser.add_argument('--bitrates', nargs='+', type=int,
                       default=[300, 750, 1850, 2850, 4300, 6000],
                       help='Bitrate levels in kbps')
    
    args = parser.parse_args()
    
    print("Generating VMAF approximations...")
    print(f"Input: {args.features}")
    print(f"Output: {args.output}")
    print(f"Bitrate levels: {args.bitrates}")
    
    generate_vmaf_table(args.features, args.output, args.bitrates)

if __name__ == '__main__':
    main()
