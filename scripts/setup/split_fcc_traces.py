# scripts/setup/split_fcc_traces.py

import random
from pathlib import Path

def split_fcc_traces(trace_list_file, output_dir, 
                     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split FCC traces into train/val/test sets"""
    
    # Read traces
    with open(trace_list_file, 'r') as f:
        traces = [line.strip() for line in f if line.strip()]
    
    print(f"📋 Total FCC traces: {len(traces)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(traces)
    
    # Split
    n = len(traces)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_traces = traces[:train_end]
    val_traces = traces[train_end:val_end]
    test_traces = traces[val_end:]
    
    print(f"✅ Train: {len(train_traces)}")
    print(f"✅ Val: {len(val_traces)}")
    print(f"✅ Test: {len(test_traces)}")
    
    # Save splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'fcc_train.txt', 'w') as f:
        f.write('\n'.join(train_traces))
    
    with open(output_dir / 'fcc_val.txt', 'w') as f:
        f.write('\n'.join(val_traces))
    
    with open(output_dir / 'fcc_test.txt', 'w') as f:
        f.write('\n'.join(test_traces))
    
    print(f"\n💾 Splits saved to: {output_dir}")
    
    return train_traces, val_traces, test_traces

if __name__ == '__main__':
    split_fcc_traces(
        trace_list_file='data/network_traces/fcc/fcc_trace_list.txt',
        output_dir='data/network_traces/fcc/splits',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )