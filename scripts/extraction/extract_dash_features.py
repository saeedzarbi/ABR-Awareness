#!/usr/bin/env python3
"""
Extract features from DASH-format videos
Structure: video_server/video1/*.m4s (all bitrates in one folder)
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_si_ti(video_path):
    """Extract Spatial and Temporal Information"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        si_values = []
        ti_values = []
        prev_frame = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Spatial Info
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobel_x**2 + sobel_y**2)
            si_values.append(np.std(sobel))
            
            # Temporal Info
            if prev_frame is not None:
                diff = gray.astype(float) - prev_frame.astype(float)
                ti_values.append(np.std(diff))
            
            prev_frame = gray.copy()
        
        cap.release()
        
        if not si_values:
            return None
        
        return {
            'si_mean': float(np.mean(si_values)),
            'ti_mean': float(np.mean(ti_values)) if ti_values else 0.0,
            'si_std': float(np.std(si_values)),
            'ti_std': float(np.std(ti_values)) if ti_values else 0.0,
            'num_frames': frame_count
        }
    
    except Exception as e:
        return None

def process_dash_videos(video_server_dir, output_file):
    """
    Process DASH-format videos
    
    Structure: 
      video_server/
        video1/
          Header.m4s
          1.m4s, 2.m4s, ... (chunks, all bitrates mixed)
        video2/
          ...
    """
    results = {}
    video_server_path = Path(video_server_dir)
    
    if not video_server_path.exists():
        raise ValueError(f"Directory not found: {video_server_dir}")
    
    # Find all video directories
    video_dirs = sorted([d for d in video_server_path.iterdir() 
                        if d.is_dir() and d.name.startswith('video')])
    
    print(f"Found {len(video_dirs)} videos")
    print()
    
    total_processed = 0
    total_failed = 0
    
    for video_dir in video_dirs:
        video_name = video_dir.name
        print(f"Processing {video_name}...")
        
        # Get all .m4s files (skip Header.m4s)
        chunk_files = sorted([f for f in video_dir.glob('*.m4s') 
                             if f.name != 'Header.m4s' and f.stem.isdigit()])
        
        if not chunk_files:
            print(f"  ⚠ No chunk files found")
            continue
        
        print(f"  Found {len(chunk_files)} chunks")
        
        # Process each chunk
        for chunk_file in tqdm(chunk_files, desc=f"  {video_name}", leave=False):
            chunk_id = chunk_file.stem  # e.g., "1", "2", "23"
            
            features = extract_si_ti(chunk_file)
            
            if features is not None:
                # Key format: video1/chunk_1
                # Note: در DASH، هر chunk ممکنه چند bitrate داشته باشه
                # ولی اینجا فرض می‌کنیم adaptive است
                key = f"{video_name}/chunk_{chunk_id}"
                results[key] = features
                total_processed += 1
            else:
                total_failed += 1
        
        print(f"  ✓ Processed {len(chunk_files)} chunks")
        print()
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Successfully processed: {total_processed}")
    print(f"✗ Failed: {total_failed}")
    print(f"✓ Features saved to: {output_path}")
    print()
    
    if total_processed > 0:
        # Statistics
        si_values = [v['si_mean'] for v in results.values()]
        ti_values = [v['ti_mean'] for v in results.values()]
        
        print("Feature Statistics:")
        print(f"  SI - Mean: {np.mean(si_values):.2f}, Std: {np.std(si_values):.2f}")
        print(f"       Range: [{np.min(si_values):.2f}, {np.max(si_values):.2f}]")
        print(f"  TI - Mean: {np.mean(ti_values):.2f}, Std: {np.std(ti_values):.2f}")
        print(f"       Range: [{np.min(ti_values):.2f}, {np.max(ti_values):.2f}]")
        print()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Extract features from DASH videos')
    parser.add_argument('--video-dir', type=str,
                       default='baseline/video_server',
                       help='Video server directory')
    parser.add_argument('--output', type=str,
                       default='data/features/si_ti_features.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DASH Video Feature Extraction")
    print("=" * 70)
    print(f"Video directory: {args.video_dir}")
    print(f"Output file: {args.output}")
    print()
    
    results = process_dash_videos(args.video_dir, args.output)
    
    if len(results) > 0:
        print("✓ Extraction complete! Ready for next step.")
    else:
        print("⚠ No features extracted. Please check your video files.")

if __name__ == '__main__':
    main()
