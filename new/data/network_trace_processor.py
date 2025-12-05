"""
Process network traces for ABR simulation.
Works with FCC trace files in various formats.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
import shutil
import re


class NetworkTraceProcessor:
    """Process and prepare network bandwidth traces."""
    
    def __init__(
        self,
        source_traces_dir: str = '/home/saeedzarbi95/test/ABR-Awareness/data/fcc_traces',
        traces_dir: str = 'data/network_traces',
        output_dir: str = 'data/network_traces/processed'
    ):
        self.source_traces_dir = Path(source_traces_dir)
        self.traces_dir = Path(traces_dir)
        self.output_dir = Path(output_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def copy_traces_from_source(self, max_traces: int = None) -> List[Path]:
        """
        Copy trace files from source directory.
        Supports files without .txt extension.
        
        Args:
            max_traces: Maximum number of traces to copy (None = all)
            
        Returns:
            List of copied trace file paths
        """
        if not self.source_traces_dir.exists():
            print(f"✗ Source directory not found: {self.source_traces_dir}")
            return []
        
        print(f"\n{'='*60}")
        print(f"Copying FCC Network Traces")
        print(f"Source: {self.source_traces_dir}")
        print(f"Target: {self.traces_dir}")
        print(f"{'='*60}\n")
        
        # Find all files (not just .txt)
        source_files = []
        for item in self.source_traces_dir.iterdir():
            if item.is_file():
                # Check if it starts with common FCC patterns
                if any(pattern in item.name for pattern in ['test_', 'fcc_', 'trace_', 'http', 'norway_', 'report_']):
                    source_files.append(item)
        
        # If no pattern match, get all files
        if not source_files:
            source_files = [f for f in self.source_traces_dir.iterdir() if f.is_file()]
        
        if not source_files:
            print(f"✗ No trace files found in source directory")
            return []
        
        print(f"Found {len(source_files)} trace files in source")
        
        if max_traces:
            source_files = source_files[:max_traces]
            print(f"Limiting to {max_traces} traces")
        
        copied = []
        
        for idx, source_file in enumerate(source_files, 1):
            target_file = self.traces_dir / source_file.name
            
            if target_file.exists():
                print(f"[{idx}/{len(source_files)}] {source_file.name[:50]}... - ✓ Already exists")
                copied.append(target_file)
                continue
            
            try:
                shutil.copy2(source_file, target_file)
                print(f"[{idx}/{len(source_files)}] {source_file.name[:50]}... - ✓ Copied")
                copied.append(target_file)
            except Exception as e:
                print(f"[{idx}/{len(source_files)}] {source_file.name[:50]}... - ✗ Error: {e}")
        
        print(f"\n✓ Copied {len(copied)}/{len(source_files)} traces")
        
        return copied
    
    def parse_fcc_trace(self, trace_path: Path) -> pd.DataFrame:
        """
        Parse a single FCC trace file.
        Handles multiple formats:
        - "timestamp throughput"
        - Just "throughput" values
        - Tab or space separated
        
        Args:
            trace_path: Path to trace file
            
        Returns:
            DataFrame with columns: timestamp, throughput_kbps
        """
        try:
            data = []
            line_number = 0
            
            with open(trace_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Split by whitespace (space or tab)
                    parts = line.split()
                    
                    if len(parts) == 0:
                        continue
                    
                    try:
                        if len(parts) >= 2:
                            # Format: "timestamp throughput"
                            timestamp = float(parts[0])
                            throughput_value = float(parts[1])
                        elif len(parts) == 1:
                            # Format: just "throughput" (generate timestamps)
                            timestamp = float(line_number)
                            throughput_value = float(parts[0])
                        else:
                            continue
                        
                        # Detect if throughput is in Mbps or Kbps
                        # Usually FCC traces are in Mbps, but check magnitude
                        if throughput_value < 100:  # Likely Mbps
                            throughput_kbps = throughput_value * 1000
                        else:  # Already in Kbps
                            throughput_kbps = throughput_value
                        
                        # Filter out unrealistic values
                        if throughput_kbps < 1 or throughput_kbps > 100000:
                            continue
                        
                        data.append({
                            'timestamp': timestamp,
                            'throughput_kbps': throughput_kbps
                        })
                        
                        line_number += 1
                        
                    except ValueError:
                        continue
            
            df = pd.DataFrame(data)
            
            # Reset timestamps to start from 0
            if not df.empty:
                df['timestamp'] = df['timestamp'] - df['timestamp'].min()
                # Sort by timestamp
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
        
        except Exception as e:
            print(f"      ✗ Error parsing: {str(e)[:50]}")
            return pd.DataFrame()
    
    def process_trace(
        self,
        trace_path: Path,
        target_duration: float = 60.0,
        sample_interval: float = 1.0
    ) -> Dict:
        """
        Process a trace file for ABR simulation.
        
        Args:
            trace_path: Path to trace file
            target_duration: Target duration in seconds
            sample_interval: Sampling interval in seconds
            
        Returns:
            Dictionary with processed trace data
        """
        df = self.parse_fcc_trace(trace_path)
        
        if df.empty or len(df) < 2:
            return {}
        
        # Resample to fixed interval
        max_time = df['timestamp'].max()
        
        # If trace is too short, skip it
        if max_time < 5.0:
            return {}
        
        # If trace is shorter than target, loop it
        if max_time < target_duration:
            num_loops = int(np.ceil(target_duration / max_time))
            dfs = []
            for i in range(num_loops):
                df_copy = df.copy()
                df_copy['timestamp'] += i * (max_time + sample_interval)
                dfs.append(df_copy)
            df = pd.concat(dfs, ignore_index=True)
        
        # Trim to target duration
        df = df[df['timestamp'] <= target_duration].copy()
        
        if len(df) < 2:
            return {}
        
        # Resample to fixed interval using interpolation
        target_timestamps = np.arange(0, target_duration, sample_interval)
        
        resampled_throughput = np.interp(
            target_timestamps,
            df['timestamp'].values,
            df['throughput_kbps'].values
        )
        
        result = {
            'trace_name': trace_path.stem,
            'duration': target_duration,
            'timestamps': target_timestamps.tolist(),
            'throughput_kbps': resampled_throughput.tolist(),
            'mean_throughput': float(np.mean(resampled_throughput)),
            'std_throughput': float(np.std(resampled_throughput)),
            'min_throughput': float(np.min(resampled_throughput)),
            'max_throughput': float(np.max(resampled_throughput)),
        }
        
        return result
    
    def process_all_traces(
        self,
        target_duration: float = 60.0,
        sample_interval: float = 1.0
    ) -> List[Dict]:
        """
        Process all traces in directory.
        
        Args:
            target_duration: Target duration for each trace
            sample_interval: Sampling interval
            
        Returns:
            List of processed traces
        """
        trace_files = [f for f in self.traces_dir.iterdir() if f.is_file()]
        
        if not trace_files:
            print("✗ No trace files found in traces directory")
            return []
        
        print(f"\n{'='*60}")
        print(f"Processing Network Traces")
        print(f"Traces: {len(trace_files)}")
        print(f"Duration: {target_duration}s per trace")
        print(f"Sample interval: {sample_interval}s")
        print(f"{'='*60}\n")
        
        processed_traces = []
        failed = 0
        skipped = 0
        
        for idx, trace_path in enumerate(trace_files, 1):
            trace_name = trace_path.name[:40] + "..." if len(trace_path.name) > 40 else trace_path.name
            print(f"[{idx}/{len(trace_files)}] {trace_name}", end=' ')
            
            processed = self.process_trace(
                trace_path,
                target_duration=target_duration,
                sample_interval=sample_interval
            )
            
            if processed:
                processed_traces.append(processed)
                
                # Save individual processed trace
                output_path = self.output_dir / f"{trace_path.stem}.json"
                with open(output_path, 'w') as f:
                    json.dump(processed, f)
                
                print(f"✓ {processed['mean_throughput']:.0f} Kbps")
            else:
                print("✗")
                failed += 1
                if failed > 10 and idx < 20:
                    # Too many failures at start, might be wrong format
                    print("\n⚠ Too many failures. Check trace file format.")
        
        print(f"\n{'='*60}")
        print(f"✓ Processed: {len(processed_traces)}/{len(trace_files)} traces")
        if failed > 0:
            print(f"✗ Failed: {failed} traces")
        print(f"{'='*60}")
        
        # Save summary
        if processed_traces:
            summary_df = pd.DataFrame([
                {
                    'trace': t['trace_name'],
                    'mean_kbps': t['mean_throughput'],
                    'std_kbps': t['std_throughput'],
                    'min_kbps': t['min_throughput'],
                    'max_kbps': t['max_throughput']
                }
                for t in processed_traces
            ])
            
            csv_path = self.output_dir / 'traces_summary.csv'
            summary_df.to_csv(csv_path, index=False)
            print(f"\n✓ Summary saved to: {csv_path}")
        
        return processed_traces
    
    def get_trace_statistics(self) -> pd.DataFrame:
        """Get statistics of all processed traces."""
        csv_path = self.output_dir / 'traces_summary.csv'
        
        if csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            return pd.DataFrame()
    
    def print_summary(self):
        """Print summary statistics of traces."""
        df = self.get_trace_statistics()
        
        if df.empty:
            print("No trace statistics available")
            return
        
        print(f"\n{'='*60}")
        print("Network Traces Summary")
        print(f"{'='*60}\n")
        
        print(f"Total traces: {len(df)}")
        print(f"\nThroughput Statistics (Kbps):")
        print(f"  Mean:   {df['mean_kbps'].mean():8.1f} ± {df['mean_kbps'].std():.1f}")
        print(f"  Min:    {df['min_kbps'].min():8.1f}")
        print(f"  Max:    {df['max_kbps'].max():8.1f}")
        print(f"  Median: {df['mean_kbps'].median():8.1f}")
        
        # Classify traces
        print(f"\nTrace Classification:")
        low = len(df[df['mean_kbps'] < 1000])
        medium = len(df[(df['mean_kbps'] >= 1000) & (df['mean_kbps'] < 3000)])
        high = len(df[df['mean_kbps'] >= 3000])
        
        print(f"  Low (<1 Mbps):      {low:3d} traces")
        print(f"  Medium (1-3 Mbps):  {medium:3d} traces")
        print(f"  High (>3 Mbps):     {high:3d} traces")


def main():
    """Main trace processing script."""
    print("\n📡 Network Trace Processor for ABR Research\n")
    
    processor = NetworkTraceProcessor(
        source_traces_dir='/home/saeedzarbi95/test/ABR-Awareness/data/fcc_traces',
        traces_dir='data/network_traces',
        output_dir='data/network_traces/processed'
    )
    
    # Step 1: Copy traces from source
    print("Step 1: Copy traces from source directory")
    choice = input("Copy all traces or limit? (all/limit): ").strip().lower()
    
    if choice == 'limit':
        num = input("How many traces? (default: 50): ").strip()
        max_traces = int(num) if num else 50
    else:
        max_traces = None
    
    copied = processor.copy_traces_from_source(max_traces=max_traces)
    
    if not copied:
        print("✗ No traces copied. Exiting.")
        return
    
    # Step 2: Process traces
    print("\nStep 2: Process traces")
    input("Press Enter to start processing...")
    
    processed = processor.process_all_traces(
        target_duration=60.0,  # 60 seconds per trace
        sample_interval=1.0    # 1 second intervals
    )
    
    if processed:
        processor.print_summary()
        
        print("\n✓ Network trace processing complete!")
        print(f"\nProcessed traces saved in: {processor.output_dir}")
        print("\nNext step: Build ABR environment")
        print("  python src/environment/abr_env.py")
    else:
        print("\n✗ No traces processed successfully")
        print("\nTip: Check if trace files have correct format:")
        print("  Each line should be: timestamp throughput")
        print("  Or just: throughput")


if __name__ == '__main__':
    main()