"""
Content-Aware Actor-Critic Model for ABR
Extends Pensieve with content features and VMAF predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentAwareActor(nn.Module):
    """
    Actor-Critic network with content awareness
    
    Inputs:
        - network_state: (batch, 6, 8) - network observations (same as Pensieve)
        - content_features: (batch, 2) - SI, TI
        - vmaf_predictions: (batch, 6) - predicted VMAF for each bitrate
    
    Outputs:
        - action_prob: (batch, 6) - probability distribution over bitrates
        - state_value: (batch, 1) - value estimate
    """
    
    def __init__(self, state_dim=(6, 8), action_dim=6, content_dim=2):
        super(ContentAwareActor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.content_dim = content_dim
        
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
        
        # Calculate conv output size
        # After conv1: (8 - 4 + 1) = 5
        # After conv2: (5 - 4 + 1) = 2
        conv_out_size = 128 * 2  # 256
        
        # ============================================
        # Content Feature Encoder (NEW)
        # ============================================
        self.content_fc1 = nn.Linear(content_dim, 32)
        self.content_fc2 = nn.Linear(32, 64)
        
        # ============================================
        # VMAF Prediction Encoder (NEW)
        # ============================================
        self.vmaf_fc = nn.Linear(action_dim, 32)
        
        # ============================================
        # Fusion Layer
        # ============================================
        # Concatenate: conv(256) + content(64) + vmaf(32) = 352
        fusion_input_size = conv_out_size + 64 + 32
        
        self.fusion_fc = nn.Linear(fusion_input_size, 128)
        
        # ============================================
        # Output Heads
        # ============================================
        self.actor_head = nn.Linear(128, action_dim)  # Policy
        self.critic_head = nn.Linear(128, 1)          # Value
        
    def forward(self, network_state, content_features, vmaf_predictions):
        """
        Forward pass
        
        Args:
            network_state: (batch, 6, 8)
            content_features: (batch, 2)
            vmaf_predictions: (batch, 6)
        
        Returns:
            action_prob: (batch, 6)
            state_value: (batch, 1)
        """
        
        # ============================================
        # 1. Encode Network State
        # ============================================
        # Input: (batch, 6, 8)
        x = F.relu(self.conv1(network_state))  # (batch, 128, 5)
        x = F.relu(self.conv2(x))              # (batch, 128, 2)
        x = x.view(x.size(0), -1)              # (batch, 256)
        
        # ============================================
        # 2. Encode Content Features (NEW)
        # ============================================
        # Input: (batch, 2) - [SI, TI]
        c = F.relu(self.content_fc1(content_features))  # (batch, 32)
        c = F.relu(self.content_fc2(c))                 # (batch, 64)
        
        # ============================================
        # 3. Encode VMAF Predictions (NEW)
        # ============================================
        # Input: (batch, 6) - VMAF for each bitrate
        v = F.relu(self.vmaf_fc(vmaf_predictions))      # (batch, 32)
        
        # ============================================
        # 4. Fusion
        # ============================================
        # Concatenate all features
        combined = torch.cat([x, c, v], dim=1)          # (batch, 352)
        
        # Fused representation
        fused = F.relu(self.fusion_fc(combined))        # (batch, 128)
        
        # ============================================
        # 5. Output Heads
        # ============================================
        # Actor: policy distribution over actions
        action_logits = self.actor_head(fused)          # (batch, 6)
        action_prob = F.softmax(action_logits, dim=1)   # (batch, 6)
        
        # Critic: value estimate
        state_value = self.critic_head(fused)           # (batch, 1)
        
        return action_prob, state_value
    
    def select_action(self, network_state, content_features, vmaf_predictions):
        """
        Select action using current policy
        
        Returns:
            action: selected action (0-5)
            action_prob: probability of selected action
            state_value: value estimate
        """
        with torch.no_grad():
            action_probs, state_value = self.forward(
                network_state, 
                content_features, 
                vmaf_predictions
            )
            
            # Sample action from distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            return action.item(), action_probs[0, action].item(), state_value.item()


# ============================================
# Helper function to create model
# ============================================
def create_content_aware_model(state_dim=(6, 8), action_dim=6, content_dim=2):
    """Create and initialize content-aware model"""
    model = ContentAwareActor(
        state_dim=state_dim,
        action_dim=action_dim,
        content_dim=content_dim
    )
    
    # Initialize weights (Xavier initialization)
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    return model


# ============================================
# Test function
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Content-Aware Model")
    print("=" * 60)
    
    # Create model
    model = create_content_aware_model()
    print(f"✓ Model created")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    
    # Dummy inputs
    network_state = torch.randn(batch_size, 6, 8)
    content_features = torch.randn(batch_size, 2)  # SI, TI
    vmaf_predictions = torch.randn(batch_size, 6)  # VMAF for 6 bitrates
    
    print("\nTest forward pass:")
    print(f"  Network state: {network_state.shape}")
    print(f"  Content features: {content_features.shape}")
    print(f"  VMAF predictions: {vmaf_predictions.shape}")
    
    # Forward
    action_prob, state_value = model(network_state, content_features, vmaf_predictions)
    
    print(f"\nOutputs:")
    print(f"  Action probabilities: {action_prob.shape}")
    print(f"  State value: {state_value.shape}")
    
    # Check probabilities sum to 1
    prob_sum = action_prob.sum(dim=1)
    print(f"\n✓ Probability sums: {prob_sum}")
    assert torch.allclose(prob_sum, torch.ones(batch_size)), "Probabilities don't sum to 1!"
    
    # Test action selection
    action, action_prob_val, value = model.select_action(
        network_state[0:1], 
        content_features[0:1], 
        vmaf_predictions[0:1]
    )
    print(f"\nTest action selection:")
    print(f"  Selected action: {action}")
    print(f"  Action probability: {action_prob_val:.4f}")
    print(f"  State value: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
