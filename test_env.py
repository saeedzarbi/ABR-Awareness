# test_env_output.py
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.fcc_trace_loader import FCCTraceLoader

# Load environment
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

# Test one step
state = env.reset()
print("State keys:", state.keys())
print("VMAF predictions:", state['vmaf'])

next_state, reward, done, info = env.step(action=3)
print("\nInfo keys:", info.keys())
print("Info content:", info)

