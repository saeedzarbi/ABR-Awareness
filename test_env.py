import sys
sys.path.insert(0, '.')

from models.fcc_trace_loader import FCCTraceLoader
from models.content_aware_env_fcc import ContentAwareEnvFCC

print("Testing environment...")

loader = FCCTraceLoader(
    fcc_trace_dir='data/fcc_traces',
    train_file='data/network_traces/fcc/splits/fcc_train.txt',
    val_file='data/network_traces/fcc/splits/fcc_val.txt',
    test_file='data/network_traces/fcc/splits/fcc_test.txt'
)

env = ContentAwareEnvFCC(
    fcc_trace_loader=loader,
    features_file='data/features/si_ti_features.json',
    vmaf_file='data/vmaf/vmaf_table.json',
    video_dir='data/videos',
    mode='train'
)

# تست reset
print("Testing reset...")
state = env.reset()

if state is None:
    print("❌ Reset returned None!")
else:
    print(f"✅ Reset OK")
    print(f"   Network: {state['network'].shape}")
    print(f"   Content: {state['content'].shape}")
    print(f"   VMAF: {state['vmaf'].shape}")

# تست step
print("\nTesting step...")
try:
    state, reward, done, info = env.step(0)
    print(f"✅ Step OK")
    print(f"   Reward: {reward:.2f}")
    print(f"   Rebuffer: {info['rebuffer_time']:.2f}s")
    print(f"   Bitrate: {info['bitrate']} kbps")
except Exception as e:
    print(f"❌ Step failed: {e}")