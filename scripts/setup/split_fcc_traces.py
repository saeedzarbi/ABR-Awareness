import os
import random
from pathlib import Path

def split_fcc_traces_from_directory(fcc_dir, output_dir, 
                                     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Automatically find all traces in directory and split them
    """
    
    print("="*60)
    print("🔍 Auto-detecting FCC traces from directory...")
    print("="*60)
    
    # Check if directory exists
    if not os.path.exists(fcc_dir):
        print(f"❌ Error: Directory not found: {fcc_dir}")
        return None, None, None
    
    # Get all trace files
    trace_files = []
    for file in os.listdir(fcc_dir):
        file_path = os.path.join(fcc_dir, file)
        # Only include files (not directories) and skip .txt files
        if os.path.isfile(file_path) and not file.endswith('.txt'):
            trace_files.append(file)
    
    if not trace_files:
        print(f"❌ Error: No trace files found in: {fcc_dir}")
        print(f"💡 Directory contents:")
        for item in os.listdir(fcc_dir)[:10]:
            print(f"   - {item}")
        return None, None, None
    
    print(f"✅ Found {len(trace_files)} trace files\n")
    
    # Show first few traces
    print("📋 Sample traces:")
    for i, trace in enumerate(sorted(trace_files)[:10]):
        print(f"   {i+1}. {trace}")
    if len(trace_files) > 10:
        print(f"   ... and {len(trace_files) - 10} more")
    print()
    
    # Shuffle
    random.seed(42)
    random.shuffle(trace_files)
    
    # Split
    n = len(trace_files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_traces = sorted(trace_files[:train_end])
    val_traces = sorted(trace_files[train_end:val_end])
    test_traces = sorted(trace_files[val_end:])
    
    print(f"📊 Split Statistics:")
    print(f"   Total:  {n} traces")
    print(f"   Train:  {len(train_traces)} traces ({len(train_traces)/n*100:.1f}%)")
    print(f"   Val:    {len(val_traces)} traces ({len(val_traces)/n*100:.1f}%)")
    print(f"   Test:   {len(test_traces)} traces ({len(test_traces)/n*100:.1f}%)")
    print()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_file = output_dir / 'fcc_train.txt'
    val_file = output_dir / 'fcc_val.txt'
    test_file = output_dir / 'fcc_test.txt'
    
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_traces))
    
    with open(val_file, 'w') as f:
        f.write('\n'.join(val_traces))
    
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_traces))
    
    print(f"💾 Splits saved to:")
    print(f"   {train_file}")
    print(f"   {val_file}")
    print(f"   {test_file}")
    print()
    
    # Save full list too
    full_list_file = output_dir / 'fcc_all_traces.txt'
    with open(full_list_file, 'w') as f:
        f.write('\n'.join(sorted(trace_files)))
    print(f"📝 Full trace list saved to:")
    print(f"   {full_list_file}")
    
    print("\n" + "="*60)
    print("✅ FCC dataset split complete!")
    print("="*60)
    
    return train_traces, val_traces, test_traces


if __name__ == '__main__':
    import sys
    
    # Get FCC directory from command line or use default
    if len(sys.argv) > 1:
        fcc_directory = sys.argv[1]
    else:
        fcc_directory = 'abr-content-aware/data/fcc_traces'
    
    print(f"\n📂 FCC Traces Directory: {fcc_directory}\n")
    
    # Output to project data directory
    output_directory = 'data/network_traces/fcc/splits'
    
    # Run split
    train, val, test = split_fcc_traces_from_directory(
        fcc_dir=fcc_directory,
        output_dir=output_directory,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    if train:
        print("\n🎉 Ready to train!")
        print("   Next step: python scripts/training/train_fcc_from_scratch.py")