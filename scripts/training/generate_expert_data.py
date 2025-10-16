# scripts/training/generate_expert_data.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import json
from tqdm import tqdm

from models.expert_model import ExpertAgent
from models.fcc_trace_loader import FCCTraceLoader
from models.content_aware_env_fcc import ContentAwareEnvFCC

def generate_data():
    print("="*60)
    print("👩‍🏫 Generating Expert Data for Imitation Learning (Comyco-style)")
    print("="*60)
    
    # ۱. لود کردن محیط و داده‌های آموزشی
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
    
    # ۲. ساخت متخصص
    expert = ExpertAgent(bitrate_levels=env.bitrate_levels)
    
    # ۳. اجرای متخصص روی داده‌ها و ذخیره تصمیمات
    expert_data = []
    num_episodes = len(loader.train_traces) # به تعداد تمام فایل‌های آموزشی
    
    for ep in tqdm(range(num_episodes), desc="Generating Expert Trajectories"):
        state = env.reset(video_id=np.random.randint(1, 7))
        done = False
        while not done:
            action = expert.select_action(state)
            
            # ذخیره state و action متخصص
            expert_data.append({
                'state': {k: v.tolist() for k, v in state.items()},
                'action': action
            })
            
            state, _, done, _ = env.step(action)
            
    # ۴. ذخیره داده‌ها در فایل JSON
    output_path = 'data/expert_data.json'
    with open(output_path, 'w') as f:
        json.dump(expert_data, f)
        
    print(f"\n✅ Expert data generated and saved to '{output_path}'")
    print(f"   Total samples: {len(expert_data):,}")
    print("="*60)

if __name__ == '__main__':
    generate_data()