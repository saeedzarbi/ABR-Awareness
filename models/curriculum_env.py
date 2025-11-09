"""
Curriculum Environment
ContentAwareEnv with curriculum learning support
"""

import numpy as np
import random
from models.content_aware_env_fcc import ContentAwareEnvFCC
from models.curriculum_loader import CurriculumTraceLoader
from models.improved_reward import ImprovedReward
from models.trace_loader import NetworkTrace


class CurriculumEnvironment(ContentAwareEnvFCC):
    """
    Environment with curriculum learning and improved reward
    """
    
    def __init__(self, fcc_trace_loader, features_file, vmaf_file, video_dir, mode='train'):
        """
        Initialize curriculum environment
        """
        # Call parent init
        super().__init__(
            fcc_trace_loader=fcc_trace_loader,
            features_file=features_file,
            vmaf_file=vmaf_file,
            video_dir=video_dir,
            mode=mode
        )
        
        # Replace reward function with improved version
        self.reward_func = ImprovedReward(
            rebuffer_penalty=2.0,
            smoothness_penalty=1.0,
            bitrate_levels=self.bitrate_levels
        )
        
        # Setup curriculum (only for training)
        if mode == 'train':
            print("\nSetting up curriculum...")
            self.curriculum_loader = CurriculumTraceLoader(
                fcc_trace_loader=fcc_trace_loader,
                n_samples=100
            )
            self.use_curriculum = True
        else:
            self.curriculum_loader = None
            self.use_curriculum = False
        
        self.current_difficulty = 0.0
    
    def set_difficulty(self, difficulty: float):
        """
        Set curriculum difficulty
        
        Args:
            difficulty: 0.0 = easy, 1.0 = hard
        """
        self.current_difficulty = np.clip(difficulty, 0.0, 1.0)
    
    def reset(self, video_id=None, split=None):
        """
        Reset with curriculum trace
        """
        # Use self.mode if split not specified
        if split is None:
            split = self.mode
        
        # Random video
        self.video_id = random.randint(1, self.num_videos) if video_id is None else video_id
        self.chunk_idx = 0
        self.buffer = 0.0
        
        # Get trace (curriculum for training, random for val/test)
        if self.use_curriculum and split == 'train':
            trace_data = self.curriculum_loader.get_curriculum_trace(self.current_difficulty)
        else:
            trace_data = self.fcc_trace_loader.get_trace(mode=split)
        
        # Create trace object
        self.current_trace = NetworkTrace(
            trace_id=f"{split}_difficulty_{self.current_difficulty:.2f}",
            timestamps=trace_data[:, 0],
            throughputs=trace_data[:, 1],
            metadata={'difficulty': self.current_difficulty, 'mode': split}
        )
        self.trace_time = 0.0
        
        # Reset history
        self.past_throughput = []
        self.past_download_time = []
        self.past_bitrates = []
        self.past_errors = []
        
        return self.get_state()
    
    def compute_reward(self, action, rebuffer_time):
        """
        Compute reward using improved reward function
        """
        current_bitrate = self.bitrate_levels[action]
        last_bitrate = self.past_bitrates[-1] if len(self.past_bitrates) > 0 else 0
        
        # Use improved reward with buffer awareness
        reward = self.reward_func.compute_reward(
            current_bitrate=current_bitrate,
            rebuffer_time=rebuffer_time,
            last_bitrate=last_bitrate,
            buffer_level=self.buffer
        )
        
        return float(reward)
    
    def get_reward_breakdown(self, action, rebuffer_time):
        """
        Get detailed reward breakdown (for debugging)
        """
        current_bitrate = self.bitrate_levels[action]
        last_bitrate = self.past_bitrates[-1] if len(self.past_bitrates) > 0 else 0
        
        return self.reward_func.get_reward_breakdown(
            current_bitrate=current_bitrate,
            rebuffer_time=rebuffer_time,
            last_bitrate=last_bitrate,
            buffer_level=self.buffer
        )


if __name__ == '__main__':
    from models.fcc_trace_loader import FCCTraceLoader
    
    print("="*60)
    print("Testing Curriculum Environment")
    print("="*60)
    
    fcc_loader = FCCTraceLoader(
        fcc_trace_dir='data/fcc_traces',
        train_file='data/network_traces/fcc/splits/fcc_train.txt',
        val_file='data/network_traces/fcc/splits/fcc_val.txt',
        test_file='data/network_traces/fcc/splits/fcc_test.txt'
    )
    
    env = CurriculumEnvironment(
        fcc_trace_loader=fcc_loader,
        features_file='data/features/si_ti_features.json',
        vmaf_file='data/vmaf/vmaf_table.json',
        video_dir='data/videos',
        mode='train'
    )
    
    print("\n" + "="*60)
    print("Testing at different difficulties:")
    print("="*60)
    
    for difficulty in [0.0, 0.5, 1.0]:
        env.set_difficulty(difficulty)
        state = env.reset()
        
        print(f"\nDifficulty {difficulty:.1f}:")
        
        # Test 3 steps
        for i in range(3):
            action = 2  # Medium bitrate
            state, reward, done, info = env.step(action)
            
            breakdown = env.get_reward_breakdown(action, info['rebuffer_time'])
            
            print(f"  Step {i+1}: Reward = {reward:+.2f}")
            print(f"    Quality:  {breakdown['quality']:+.2f}")
            print(f"    Bonuses:  {breakdown['bitrate_bonus'] + breakdown['perfect_bonus'] + breakdown['buffer_bonus']:+.2f}")
            print(f"    Penalties: {breakdown['rebuffer_penalty'] + breakdown['smoothness_penalty']:+.2f}")
            
            if done:
                break
    
    print("\n✓ Curriculum environment tests passed!")