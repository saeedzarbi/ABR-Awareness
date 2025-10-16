"""
Pensieve Actor-Critic Model (Compatible Version)
Accepts content/vmaf inputs but ignores them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PensieveActorCompatible(nn.Module):
    """
    Actor-Critic network (Pensieve original)
    Accepts 3 inputs to be compatible with the trainer,
    but only uses network_state.
    """
    
    def __init__(self, state_dim=(6, 8), action_dim=6, content_dim=2):
        super(PensieveActorCompatible, self).__init__()
        
        print("🧠 Initializing PensieveActorCompatible (Content-BLIND Model)")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # ============================================
        # Network State Encoder (from Pensieve)
        # ============================================
        self.conv1 = nn.Conv1d(
            in_channels=state_dim[0],
            out_channels=128,
            kernel_size=4
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=4
        )
        
        # Calculate conv output size: (8 - 4 + 1) -> 5, (5 - 4 + 1) -> 2
        conv_out_size = 128 * 2  # 256
        
        # ============================================
        # Fusion Layer (Only network features)
        # ============================================
        fusion_input_size = conv_out_size  # 256
        
        self.fusion_fc = nn.Linear(fusion_input_size, 128)
        
        # ============================================
        # Output Heads
        # ============================================
        self.actor_head = nn.Linear(128, action_dim)  # Policy
        self.critic_head = nn.Linear(128, 1)          # Value
        
    def forward(self, network_state, content_features, vmaf_predictions):
        """
        Forward pass
        IGNORES content_features and vmaf_predictions
        """
        
        # ============================================
        # 1. Encode Network State
        # ============================================
        x = F.relu(self.conv1(network_state))  # (batch, 128, 5)
        x = F.relu(self.conv2(x))              # (batch, 128, 2)
        x = x.view(x.size(0), -1)              # (batch, 256)
        
        # ============================================
        # 2. Fusion (ONLY uses network state 'x')
        # ============================================
        fused = F.relu(self.fusion_fc(x))        # (batch, 128)
        
        # ============================================
        # 3. Output Heads
        # ============================================
        action_logits = self.actor_head(fused)          # (batch, 6)
        action_prob = F.softmax(action_logits, dim=1)   # (batch, 6)
        
        state_value = self.critic_head(fused)           # (batch, 1)
        
        return action_prob, state_value