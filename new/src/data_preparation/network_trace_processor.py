"""
Process network traces for ABR simulation.
Supports FCC broadband dataset and custom trace formats.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import requests
import gzip
import shutil


class NetworkTraceProcessor:
    """Process and prepare network bandwidth traces."""
    
    # FCC dataset URLs (sample subset)
    FCC_TRACES_URLS = [
        "https://raw.githubusercontent.com/hongzimao/pensieve/master/sim/cooked_traces/",
    ]
    
    def __init__(
        self,
        traces_dir: str = 'data/network_traces',
        output_dir: str = 'data/network_traces/processed'
    ):
        self.traces_dir = Path(traces_dir)
        self.output_dir = Path(output_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_fcc_traces(self, num_traces: int = 50) -> List[Path]:
        """
        Download FCC network traces from Pensieve repository.
        
        Args:
            num_traces: Number of traces to download
            
        Returns:
            List of downloaded trace file paths
        """
        print(f"\n{'='*60}")
        print(f"Downloading FCC Network Traces")
        print(f"Target: {num_traces} traces")
        print(f"{'='*60}\n")
        
        # Common trace names from Pensieve dataset
        trace_names = [
            f"report_bicycle_0{i}.txt" for i in range(0, min(10, num_traces))
        ] + [
            f"report_bus_0{i}.txt" for i in range(0, min(10, num_traces-10))
        ] + [
            f"report_car_0{i}.txt" for i in range(0, min(10, num_traces-20))
        ] + [
            f"report_ferry_0{i}.txt" for i in range(0, min(10, num_traces-30))
        ] + [
            f"report_metro_0{i}.txt" for i in range(0, min(10, num_traces-40))
        ]
        
        base_url = "https://raw.githubusercontent.com/hongzimao/pensieve/master/sim/cooked_traces/"
        
        downloaded = []
        
        for idx, trace_name in enumerate(trace_names[:num_traces], 1):
            trace_path = self.traces_dir / trace_name
            
            if trace_path.exists():
                print(f"[{idx}/{num_traces}] {trace_name} - ✓ Already exists")
                downloaded.append(trace_path)
                continue
            
            url = base_url + trace_name
            
            try:
                print(f"[{idx}/{num_traces}] Downloading {trace_name}...", end=' ')
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    with open(trace_path, 'wb') as f:
                        f.write(response.content)
                    print("✓")
                    downloaded.append(trace_path)
                else:
                    print(f"✗ (HTTP {response.status_code})")
            
            except Exception as e:
                print(f"✗ ({str(e)[:30]})")
        
        print(f"\n✓ Downloaded {len(downloaded)}/{num_traces} traces")
        
        return downloaded
    
    def parse_fcc_trace(self, trace_path: Path) -> pd.DataFrame:
        """
        Parse a single FCC trace file.
        
        FCC format: Each line is "timestamp throughput"
        - timestamp: in seconds
        - throughput: in Mbps
        
        Args:
            trace_path: Path to trace file
            
        Returns:
            DataFrame with columns: timestamp, throughput_kbps
        """
        try:
            data = []
            
            with open(trace_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        timestamp = float(parts[0])
                        throughput_mbps = float(parts[1])
                        throughput_kbps = throughput_mbps * 1000  # Convert to Kbps
                        
                        data.append({
                            'timestamp': timestamp,
                            'throughput_kbps': throughput_kbps
                        })
            
            df = pd.DataFrame(data)
            
            # Reset timestamps to start from 0
            if not df.empty:
                df['timestamp'] = df['timestamp'] - df['timestamp'].min()
            
            return df
        
        except Exception as e:
            print(f"✗ Error parsing {trace_path.name}: {e}")
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
        
        if df.empty:
            return {}
        
        # Resample to fixed interval
        max_time = df['timestamp'].max()
        
        # If trace is shorter than target, loop it
        if max_time < target_duration:
            num_loops = int(np.ceil(target_duration / max_time))
            dfs = []
            for i in range(num_loops):
                df_copy = df.copy()
                df_copy['timestamp'] += i * max_time
                dfs.append(df_copy)
            df = pd.concat(dfs, ignore_index=True)
        
        # Trim to target duration
        df = df[df['timestamp'] <= target_duration].copy()
        
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
        trace_files = list(self.traces_dir.glob("*.txt"))
        
        if not trace_files:
            print("✗ No trace files found")
            return []
        
        print(f"\n{'='*60}")
        print(f"Processing Network Traces")
        print(f"Traces: {len(trace_files)}")
        print(f"Duration: {target_duration}s per trace")
        print(f"Sample interval: {sample_interval}s")
        print(f"{'='*60}\n")
        
        processed_traces = []
        
        for idx, trace_path in enumerate(trace_files, 1):
            print(f"[{idx}/{len(trace_files)}] {trace_path.name}...", end=' ')
            
            processed = self.process_trace(
                trace_path,
                target_duration=target_duration,
                sample_interval=sample_interval
            )
            
            if processed:
                processed_traces.append(processed)
                
                # Save individual processed trace
                output_path = self.output_dir / f"{trace_path.stem}.json"
                import json
                with open(output_path, 'w') as f:
                    json.dump(processed, f, indent=2)
                
                print(f"✓ (avg: {processed['mean_throughput']:.0f} Kbps)")
            else:
                print("✗ Failed")
        
        print(f"\n✓ Processed {len(processed_traces)}/{len(trace_files)} traces")
        
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
            print(f"✓ Summary saved to: {csv_path}")
        
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
        traces_dir='data/network_traces',
        output_dir='data/network_traces/processed'
    )
    
    # Download traces
    print("Step 1: Download traces")
    choice = input("Download FCC traces? (y/n): ").strip().lower()
    
    if choice == 'y':
        num_traces = input("How many traces? (default: 20): ").strip()
        num_traces = int(num_traces) if num_traces else 20
        processor.download_fcc_traces(num_traces=num_traces)
    
    # Process traces
    print("\nStep 2: Process traces")
    processed = processor.process_all_traces(
        target_duration=60.0,  # 60 seconds per trace
        sample_interval=1.0    # 1 second intervals
    )
    
    if processed:
        processor.print_summary()
        
        print("\n✓ Network trace processing complete!")
        print("\nNext step: Build ABR environment")
        print("  python src/environment/abr_env.py")
    else:
        print("\n✗ No traces processed")


if __name__ == '__main__':
    main()