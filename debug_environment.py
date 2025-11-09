"""
Debug environment - Find root cause
"""
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader
import numpy as np

print("="*80)
print("DEBUGGING ENVIRONMENT")
print("="*80)

# Create environment
fcc_loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

env = ContentAwareEnvFCC(
    fcc_trace_loader=fcc_loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='val'
)

print("\n1. Testing Network Trace Loading:")
print("-" * 80)

# Sample a trace
trace = fcc_loader.get_trace(mode='val')
print(f"Trace shape: {trace.shape}")
print(f"First 10 throughput values:")
for i in range(min(10, len(trace))):
    print(f"  t={trace[i,0]:.2f}s, throughput={trace[i,1]:.2f}")

# Check throughput range
throughputs = trace[:, 1]
print(f"\nThroughput statistics:")
print(f"  Min: {np.min(throughputs):.2f}")
print(f"  Max: {np.max(throughputs):.2f}")
print(f"  Mean: {np.mean(throughputs):.2f}")
print(f"  Median: {np.median(throughputs):.2f}")

# Check unit
if np.mean(throughputs) < 50:
    print(f"  ⚠️  WARNING: Throughput seems to be in Mbps (need conversion to kbps)")
else:
    print(f"  ✅ Throughput seems to be in kbps")

print("\n2. Testing Download Simulation:")
print("-" * 80)

state = env.reset()

# Test lowest bitrate (should have minimal rebuffering)
print("\nTesting LOWEST bitrate (300 kbps):")
for i in range(5):
    state, reward, done, info = env.step(0)
    print(f"  Step {i+1}: bitrate=300kbps, "
          f"rebuffer={info['rebuffer_time']:.2f}s, "
          f"download={info['download_time']:.2f}s, "
          f"throughput={info['throughput']:.1f}kbps, "
          f"reward={reward:.2f}")
    if done:
        break

print("\n3. Testing Download Time Calculation:")
print("-" * 80)

# Reset
state = env.reset()

# Manually check one download
selected_bitrate = 300  # kbps
chunk_duration = 4.0  # seconds
chunk_size_kbit = selected_bitrate * chunk_duration  # 1200 kbit

print(f"Chunk size: {chunk_size_kbit} kbit")
print(f"Expected download time at 1000 kbps: {chunk_size_kbit/1000:.2f}s")
print(f"Expected download time at 500 kbps: {chunk_size_kbit/500:.2f}s")
print(f"Expected download time at 100 kbps: {chunk_size_kbit/100:.2f}s")

# Now step
state, reward, done, info = env.step(0)
print(f"\nActual download time: {info['download_time']:.2f}s")
print(f"Actual throughput: {info['throughput']:.1f} kbps")
print(f"Rebuffering: {info['rebuffer_time']:.2f}s")

if info['download_time'] > 10:
    print("\n⚠️  PROBLEM: Download time > 10s for 300kbps!")
    print("This suggests:")
    print("  1. Throughput is in Mbps but treated as kbps (need × 1000)")
    print("  2. OR network trace has very low throughput values")

print("\n4. Testing Reward Calculation:")
print("-" * 80)

# Check reward components
vmaf = info.get('vmaf', 50)
rebuffer = info['rebuffer_time']
expected_reward_approx = vmaf - 4.3 * rebuffer

print(f"VMAF: {vmaf:.1f}")
print(f"Rebuffer: {rebuffer:.2f}s")
print(f"Expected reward (approx): {vmaf} - 4.3 × {rebuffer:.2f} = {expected_reward_approx:.2f}")
print(f"Actual reward: {reward:.2f}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

if np.mean(throughputs) < 50:
    print("❌ Throughput in traces is in Mbps but code expects kbps")
    print("   FIX: Multiply throughput by 1000 in trace loading")
elif info['download_time'] > 10:
    print("❌ Download times are too long - check conversion logic")
else:
    print("✅ Environment seems OK - high rebuffering may be expected")
    print("   Consider reducing rebuffer_penalty from 4.3 to 2.0")

print("="*80)