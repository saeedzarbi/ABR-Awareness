"""
Test FCC setup - CORRECTED version
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("🧪 Testing FCC Setup")
print("=" * 70)

# 1. Test FCCTraceLoader
print("\n1️⃣ Testing FCCTraceLoader...")
try:
    from models.fcc_trace_loader import FCCTraceLoader
    
    loader = FCCTraceLoader(
        fcc_trace_dir='data/fcc_traces',
        train_file='data/network_traces/fcc/splits/fcc_train.txt',
        val_file='data/network_traces/fcc/splits/fcc_val.txt',
        test_file='data/network_traces/fcc/splits/fcc_test.txt'
    )
    
    # Test getting a trace
    trace = loader.get_trace('train')
    print(f"   ✅ TraceLoader OK! Got trace with shape: {trace.shape}")
except Exception as e:
    print(f"   ❌ TraceLoader Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. Test ContentAwareEnvFCC
print("\n2️⃣ Testing ContentAwareEnvFCC...")
try:
    from models.content_aware_env_fcc import ContentAwareEnvFCC
    
    env = ContentAwareEnvFCC(
        fcc_trace_loader=loader,
        features_file='data/features/si_ti_features.json',  # ✅ اسم صحیح
        vmaf_file='data/vmaf/vmaf_table.json',              # ✅ اسم صحیح
        video_dir='data/videos',
        mode='train'
    )
    
    print(f"   ✅ Environment created!")
    
    # Test reset
    state = env.reset()
    print(f"   ✅ Reset OK! State keys: {list(state.keys())}")
    print(f"      Network shape: {state['network'].shape}")
    print(f"      Content shape: {state['content'].shape}")
    print(f"      VMAF shape: {state['vmaf'].shape}")
    
    # Test step
    next_state, reward, done, info = env.step(2)
    print(f"   ✅ Step OK!")
    print(f"      Reward: {reward:.2f}")
    print(f"      Bitrate: {info['bitrate']} kbps")
    print(f"      Buffer: {info['buffer']:.2f}s")
    
except Exception as e:
    print(f"   ❌ Environment Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Test Model
print("\n3️⃣ Testing Model...")
try:
    import torch
    from models.content_aware_model import ContentAwareActor
    
    model = ContentAwareActor(
        state_dim=(6, 8),
        action_dim=6,
        content_dim=2
    )
    
    # Test forward pass
    network_state = torch.FloatTensor(state['network']).unsqueeze(0)
    content_features = torch.FloatTensor(state['content']).unsqueeze(0)
    vmaf_features = torch.FloatTensor(state['vmaf']).unsqueeze(0)
    
    action_probs, value = model(network_state, content_features, vmaf_features)
    
    print(f"   ✅ Model OK!")
    print(f"      Action probs shape: {action_probs.shape}")
    print(f"      Value shape: {value.shape}")
    
except Exception as e:
    print(f"   ❌ Model Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Full episode test
print("\n4️⃣ Running full test episode (10 steps)...")
try:
    state = env.reset()
    total_reward = 0
    
    for step in range(10):
        action = 2  # Medium bitrate
        
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"   Step {step+1}: action={action}, reward={reward:+6.2f}, "
              f"buffer={info['buffer']:5.2f}s, bitrate={info['bitrate']:4d}kbps")
        
        if done:
            break
        
        state = next_state
    
    print(f"\n   ✅ Episode complete! Total reward: {total_reward:+7.2f}")
    
except Exception as e:
    print(f"   ❌ Episode Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED! System is ready!")
print("=" * 70)
print("\n🚀 Next step:")
print("   python scripts/training/train_fcc_from_scratch.py")