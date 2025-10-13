"""
Convert FCC traces to Pensieve cooked format
"""

import os
import csv
import numpy as np
from pathlib import Path


def parse_fcc_trace(fcc_file):
    """
    Parse FCC CSV and extract throughput measurements
    
    FCC format:
    - curr_httpgetmt table has bytes_sec, transfer_time
    - Need to convert to time-throughput pairs
    """
    
    measurements = []
    
    try:
        with open(fcc_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Skip if not download test
                if 'bytes_sec' not in row:
                    continue
                
                # Extract throughput (bytes/sec → Mbps)
                bytes_per_sec = float(row['bytes_sec'])
                throughput_mbps = (bytes_per_sec * 8) / 1_000_000
                
                # Extract timestamp
                timestamp = float(row.get('dtime', 0))
                
                measurements.append({
                    'time': timestamp,
                    'throughput': throughput_mbps
                })
    
    except Exception as e:
        print(f"Error parsing {fcc_file}: {e}")
        return None
    
    return measurements


def normalize_trace(measurements):
    """
    Normalize to relative time starting from 0
    Resample to 0.5s intervals
    """
    
    if not measurements or len(measurements) < 2:
        return None
    
    # Sort by time
    measurements.sort(key=lambda x: x['time'])
    
    # Normalize to start at 0
    start_time = measurements[0]['time']
    for m in measurements:
        m['time'] -= start_time
    
    # Resample to 0.5s intervals
    duration = measurements[-1]['time']
    times = np.arange(0, duration, 0.5)
    
    resampled = []
    for t in times:
        # Find closest measurement
        closest = min(measurements, key=lambda x: abs(x['time'] - t))
        resampled.append((t, closest['throughput']))
    
    return resampled


def filter_trace(trace, min_throughput=0.2, max_throughput=6.0):
    """
    Filter traces like Pensieve:
    - Average throughput 0.2-6 Mbps
    - Duration > 60 seconds
    """
    
    if not trace or len(trace) < 120:  # < 60 seconds at 0.5s intervals
        return False
    
    throughputs = [t[1] for t in trace]
    avg_throughput = np.mean(throughputs)
    
    if avg_throughput < min_throughput or avg_throughput > max_throughput:
        return False
    
    return True


def save_trace(trace, output_file):
    """Save in Pensieve cooked format"""
    with open(output_file, 'w') as f:
        for time, throughput in trace:
            f.write(f"{time:.1f} {throughput:.2f}\n")


def process_fcc_directory(input_dir, output_dir, target_count=100):
    """
    Process FCC directory and create Pensieve traces
    """
    
    print("=" * 70)
    print("Converting FCC traces to Pensieve format")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = list(Path(input_dir).rglob('*.csv'))
    print(f"Found {len(csv_files)} CSV files")
    
    converted = 0
    processed = 0
    
    for csv_file in csv_files:
        if converted >= target_count:
            break
        
        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{len(csv_files)}, "
                  f"Converted: {converted}/{target_count}")
        
        # Parse
        measurements = parse_fcc_trace(csv_file)
        if measurements is None:
            continue
        
        # Normalize
        trace = normalize_trace(measurements)
        if trace is None:
            continue
        
        # Filter
        if not filter_trace(trace):
            continue
        
        # Save
        output_file = output_dir / f"fcc_trace_{converted:03d}"
        save_trace(trace, output_file)
        converted += 1
    
    print(f"\n✓ Converted {converted} traces")
    print(f"  Saved to: {output_dir}")
    
    return converted


def main():
    input_dir = Path('data/fcc_traces/data-raw-2016-jun')
    output_dir = Path('data/fcc_traces/cooked')
    
    count = process_fcc_directory(input_dir, output_dir, target_count=100)
    
    if count < 50:
        print("\n⚠ Warning: Less than 50 traces converted")
        print("  FCC format may have changed")
        print("  Consider using pre-converted Pensieve traces instead")
    else:
        print("\n✓ Ready for testing!")


if __name__ == '__main__':
    main()