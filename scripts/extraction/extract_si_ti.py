#!/usr/bin/env python3
"""
Extract Spatial Information (SI) and Temporal Information (TI) 
from video chunks
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_si_ti(video_path):
    """
    Extract Spatial Info and Temporal Info from a video
    
    SI: Spatial Information - measures spatial complexity
    TI: Temporal Information - measures temporal complexity
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict with si_mean, ti_mean, si_std, ti_std
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    si_values = []
    ti_values = []
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Spatial Info: std of Sobel filtered frame
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        si = np.std(sobel)
        si_values.append(si)
        
        # Temporal Info: std of frame difference
        if prev_frame is not None:
            diff = gray.astype(float) - prev_frame.astype(float)
            ti = np.std(diff)
            ti_values.append(ti)
        
        prev_frame = gray.copy()
    
    cap.release()
    
    if not si_values:
        return {'si_mean': 0, 'ti_mean': 0, 'si_std': 0, 'ti_std': 0}
    
    return {
        'si_mean': float(np.mean(si_values)),
        'ti_mean': float(np.mean(ti_values)) if ti_values else 0,
        'si_std': float(np.std(si_values)),
        'ti_std': float(np.std(ti_values)) if ti_values else 0,
        'num_frames': len(si_values)
    }

def process_pensieve_videos(baseline_video_dir, output_file):
    """
    Process videos from Pytorch-Pensieve baseline
    
    Structure: video_server/video1/300/*.m4s
    """
    results = {}
    baseline_video_dir = Path(baseline_video_dir)
    
    if not baseline_video_dir.exists():
        raise ValueError(f"Video directory not found: {baseline_video_dir}")
    
    # Iterate through video1-6
    video_dirs = sorted([d for d in baseline_video_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('video')])
    
    print(f"Found {len(video_dirs)} videos")
    
    for video_dir in tqdm(video_dirs, desc="Videos"):
        video_name = video_dir.name
        
        # Iterate through bitrate directories
        bitrate_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
        
        for bitrate_dir in tqdm(bitrate_dirs, desc=f"  {video_name}", leave=False):
            bitrate = bitrate_dir.name
            
            # Process each chunk
            chunk_files = sorted(bitrate_dir.glob('*.m4s'))
            
            for chunk_file in chunk_files:
                chunk_id = chunk_file.stem
                
                try:
                    features = extract_si_ti(chunk_file)
                    
                    # Key format: video1/300/chunk_0
                    key = f"{video_name}/{bitrate}/{chunk_id}"
                    results[key] = features
                    
                except Exception as e:
                    print(f"\n✗ Error processing {chunk_file}: {e}")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Features saved to {output_file}")
    print(f"  Total chunks processed: {len(results)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Extract SI/TI features from videos')
    parser.add_argument('--video-dir', type=str, 
                       default='baseline/video_server',
                       help='Directory containing videos')
    parser.add_argument('--output', type=str,
                       default='data/features/si_ti_features.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    print("Starting feature extraction...")
    print(f"Video directory: {args.video_dir}")
    print(f"Output file: {args.output}")
    
    results = process_pensieve_videos(args.video_dir, args.output)
    
    # Print statistics
    if results:
        si_values = [v['si_mean'] for v in results.values()]
        ti_values = [v['ti_mean'] for v in results.values()]
        
        print("\nFeature Statistics:")
        print(f"  SI - Mean: {np.mean(si_values):.2f}, Std: {np.std(si_values):.2f}")
        print(f"  TI - Mean: {np.mean(ti_values):.2f}, Std: {np.std(ti_values):.2f}")

if __name__ == '__main__':
    main()
