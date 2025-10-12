#!/usr/bin/env python3
"""Generate realistic dummy features for quick prototyping"""

import json
import numpy as np
import os

def generate_features():
    """Generate SI/TI features for 6 videos × 6 bitrates × 48 chunks"""
    
    features = {}
    vmaf_table = {}
    
    bitrates = [300, 750, 1850, 2850, 4300, 6000]
    np.random.seed(42)
    
    print("Generating features...")
    
    for video_num in range(1, 7):  # video1-6
        video_complexity_factor = np.random.uniform(0.7, 1.3)
        
        for chunk_num in range(48):  # 48 chunks per video
            chunk_complexity = np.random.uniform(0.8, 1.2)
            
            # Generate realistic SI/TI
            si_mean = np.random.uniform(35, 75) * video_complexity_factor * chunk_complexity
            ti_mean = np.random.uniform(8, 20) * video_complexity_factor * chunk_complexity
            
            for bitrate in bitrates:
                key = f"video{video_num}/{bitrate}/chunk_{chunk_num:03d}"
                
                features[key] = {
                    'si_mean': float(si_mean),
                    'ti_mean': float(ti_mean),
                    'si_std': float(si_mean * 0.15),
                    'ti_std': float(ti_mean * 0.2),
                    'num_frames': 120
                }
                
                # Generate VMAF
                if key not in vmaf_table:
                    vmaf_table[key] = {}
                
                bitrate_norm = bitrate / 6000.0
                complexity = (si_mean + ti_mean) / 100.0
                
                base_vmaf = 25 + 55 * np.log(1 + bitrate_norm * 10) / np.log(11)
                complexity_penalty = 12 * complexity * (1 - bitrate_norm)
                vmaf = base_vmaf - complexity_penalty + np.random.normal(0, 2)
                vmaf = np.clip(vmaf, 20, 95)
                
                vmaf_table[key][str(bitrate)] = float(vmaf)
    
    return features, vmaf_table

def main():
    os.makedirs('data/features', exist_ok=True)
    os.makedirs('data/vmaf', exist_ok=True)
    
    features, vmaf_table = generate_features()
    
    with open('data/features/si_ti_features.json', 'w') as f:
        json.dump(features, f, indent=2)
    
    with open('data/vmaf/vmaf_table.json', 'w') as f:
        json.dump(vmaf_table, f, indent=2)
    
    print()
    print("=" * 60)
    print("✓ Dummy features generated successfully!")
    print("=" * 60)
    print(f"  SI/TI Features: {len(features)} entries")
    print(f"  VMAF Table: {len(vmaf_table)} entries")
    print()
    print("Files created:")
    print("  → data/features/si_ti_features.json")
    print("  → data/vmaf/vmaf_table.json")
    print()
    
    si_values = [v['si_mean'] for v in features.values()]
    ti_values = [v['ti_mean'] for v in features.values()]
    all_vmaf = []
    for entry in vmaf_table.values():
        all_vmaf.extend(entry.values())
    
    print("Statistics:")
    print(f"  SI  - Range: [{min(si_values):.1f}, {max(si_values):.1f}], Mean: {np.mean(si_values):.1f}")
    print(f"  TI  - Range: [{min(ti_values):.1f}, {max(ti_values):.1f}], Mean: {np.mean(ti_values):.1f}")
    print(f"  VMAF - Range: [{min(all_vmaf):.1f}, {max(all_vmaf):.1f}], Mean: {np.mean(all_vmaf):.1f}")
    print()
    print("✓ Ready to build Content-Aware model!")

if __name__ == '__main__':
    main()
