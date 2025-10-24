"""
Content-Aware Actor-Critic Model for ABR (IMPROVED VERSION)
✅ اضافه شده: Dropout, BatchNorm
✅ بهبود: جلوگیری از Overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentAwareActorImproved(nn.Module):
    """
    Actor-Critic network با قابلیت‌های anti-overfitting
    
    بهبودها نسبت به نسخه قبلی:
    - ✅ Dropout برای regularization
    - ✅ BatchNorm برای پایداری training
    - ✅ Smaller hidden dims برای کاهش parameters
    
    Inputs:
        - network_state: (batch, 6, 8) - network observations
        - content_features: (batch, 2) - SI, TI
        - vmaf_predictions: (batch, 6) - predicted VMAF for each bitrate
    
    Outputs:
        - action_prob: (batch, 6) - probability distribution over bitrates
        - state_value: (batch, 1) - value estimate
    """
    
    def __init__(self, 
                 state_dim=(6, 8), 
                 action_dim=6, 
                 content_dim=2,
                 hidden_dim=128,      # قابل تنظیم
                 dropout_rate=0.2,    # نرخ Dropout
                 use_batchnorm=True): # استفاده از BatchNorm
        super(ContentAwareActorImproved, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        
        # ============================================
        # Network State Encoder (from Pensieve)
        # ============================================
        self.conv1 = nn.Conv1d(
            in_channels=state_dim[0],
            out_channels=hidden_dim,
            kernel_size=4
        )
        
        # ✅ BatchNorm بعد از Conv1
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=4
        )
        
        # ✅ BatchNorm بعد از Conv2
        if self.use_batchnorm:
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # ✅ Dropout بعد از Conv layers
        self.dropout_conv = nn.Dropout(dropout_rate)
        
        # Calculate conv output size
        # After conv1: (8 - 4 + 1) = 5
        # After conv2: (5 - 4 + 1) = 2
        conv_out_size = hidden_dim * 2
        
        # ============================================
        # Content Feature Encoder (NEW)
        # ============================================
        self.content_fc1 = nn.Linear(content_dim, 32)
        self.content_fc2 = nn.Linear(32, 64)
        
        # ✅ Dropout برای content encoder
        self.dropout_content = nn.Dropout(dropout_rate)
        
        # ============================================
        # VMAF Prediction Encoder (NEW)
        # ============================================
        self.vmaf_fc = nn.Linear(action_dim, 32)
        
        # ✅ Dropout برای VMAF encoder
        self.dropout_vmaf = nn.Dropout(dropout_rate)
        
        # ============================================
        # Fusion Layer
        # ============================================
        # Concatenate: conv(hidden_dim*2) + content(64) + vmaf(32)
        fusion_input_size = conv_out_size + 64 + 32
        
        self.fusion_fc = nn.Linear(fusion_input_size, hidden_dim)
        
        # ✅ Dropout برای fusion layer
        self.dropout_fusion = nn.Dropout(dropout_rate)
        
        # ============================================
        # Output Heads (بدون Dropout)
        # ============================================
        self.actor_head = nn.Linear(hidden_dim, action_dim)  # Policy
        self.critic_head = nn.Linear(hidden_dim, 1)          # Value
        
    def forward(self, network_state, content_features, vmaf_predictions):
        """
        Forward pass با Dropout و BatchNorm
        
        Args:
            network_state: (batch, 6, 8)
            content_features: (batch, 2)
            vmaf_predictions: (batch, 6)
        
        Returns:
            action_prob: (batch, 6)
            state_value: (batch, 1)
        """
        
        # ============================================
        # 1. Encode Network State با BatchNorm و Dropout
        # ============================================
        # Input: (batch, 6, 8)
        x = self.conv1(network_state)  # (batch, hidden_dim, 5)
        
        # ✅ BatchNorm
        if self.use_batchnorm:
            x = self.bn1(x)
        
        x = F.relu(x)
        
        # ✅ Dropout
        x = self.dropout_conv(x)
        
        x = self.conv2(x)              # (batch, hidden_dim, 2)
        
        # ✅ BatchNorm
        if self.use_batchnorm:
            x = self.bn2(x)
        
        x = F.relu(x)
        
        # ✅ Dropout
        x = self.dropout_conv(x)
        
        x = x.view(x.size(0), -1)      # (batch, hidden_dim*2)
        
        # ============================================
        # 2. Encode Content Features با Dropout
        # ============================================
        # Input: (batch, 2) - [SI, TI]
        c = F.relu(self.content_fc1(content_features))  # (batch, 32)
        c = self.dropout_content(c)                     # ✅ Dropout
        
        c = F.relu(self.content_fc2(c))                 # (batch, 64)
        c = self.dropout_content(c)                     # ✅ Dropout
        
        # ============================================
        # 3. Encode VMAF Predictions با Dropout
        # ============================================
        # Input: (batch, 6) - VMAF for each bitrate
        v = F.relu(self.vmaf_fc(vmaf_predictions))      # (batch, 32)
        v = self.dropout_vmaf(v)                        # ✅ Dropout
        
        # ============================================
        # 4. Fusion با Dropout
        # ============================================
        # Concatenate all features
        combined = torch.cat([x, c, v], dim=1)          # (batch, fusion_input_size)
        
        # Fused representation
        fused = F.relu(self.fusion_fc(combined))        # (batch, hidden_dim)
        fused = self.dropout_fusion(fused)              # ✅ Dropout
        
        # ============================================
        # 5. Output Heads (بدون Dropout)
        # ============================================
        # Actor: policy distribution over actions
        action_logits = self.actor_head(fused)          # (batch, action_dim)
        action_prob = F.softmax(action_logits, dim=1)   # (batch, action_dim)
        
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
# Helper function to create improved model
# ============================================
def create_improved_model(
    state_dim=(6, 8), 
    action_dim=6, 
    content_dim=2,
    hidden_dim=128,
    dropout_rate=0.2,
    use_batchnorm=True
):
    """
    Create and initialize improved content-aware model
    
    Args:
        state_dim: Network state dimensions
        action_dim: Number of actions (bitrates)
        content_dim: Content feature dimensions (SI, TI)
        hidden_dim: Hidden layer size (default 128, کوچکتر = less overfitting)
        dropout_rate: Dropout probability (0.0 - 0.5)
        use_batchnorm: استفاده از BatchNorm
    
    Returns:
        model: Initialized model
    """
    model = ContentAwareActorImproved(
        state_dim=state_dim,
        action_dim=action_dim,
        content_dim=content_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm
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
    print("=" * 80)
    print("🧪 Testing IMPROVED Content-Aware Model")
    print("=" * 80)
    print()
    
    # ============================================
    # Test 1: مدل استاندارد
    # ============================================
    print("Test 1: Standard Model (dropout=0.2, batchnorm=True)")
    print("-" * 80)
    
    model = create_improved_model(
        dropout_rate=0.2,
        use_batchnorm=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print()
    
    # Test forward pass
    batch_size = 4
    network_state = torch.randn(batch_size, 6, 8)
    content_features = torch.randn(batch_size, 2)
    vmaf_predictions = torch.randn(batch_size, 6)
    
    # Training mode
    model.train()
    action_prob, state_value = model(network_state, content_features, vmaf_predictions)
    
    print(f"✅ Forward pass (training mode):")
    print(f"   Action probabilities: {action_prob.shape}")
    print(f"   State value: {state_value.shape}")
    print(f"   Prob sums: {action_prob.sum(dim=1)}")
    print()
    
    # Eval mode
    model.eval()
    action_prob_eval, state_value_eval = model(network_state, content_features, vmaf_predictions)
    
    print(f"✅ Forward pass (eval mode):")
    print(f"   Action probabilities: {action_prob_eval.shape}")
    
    # نتایج باید متفاوت باشند (به خاطر Dropout)
    diff = torch.abs(action_prob - action_prob_eval).mean().item()
    print(f"   Difference from training mode: {diff:.6f}")
    if diff > 0.001:
        print("   ✅ Dropout is working! (outputs differ between train/eval)")
    print()
    
    # ============================================
    # Test 2: مدل بدون BatchNorm
    # ============================================
    print("Test 2: Model without BatchNorm")
    print("-" * 80)
    
    model_no_bn = create_improved_model(
        dropout_rate=0.2,
        use_batchnorm=False
    )
    
    params_no_bn = sum(p.numel() for p in model_no_bn.parameters())
    print(f"✅ Model without BN created")
    print(f"   Parameters: {params_no_bn:,} (vs {total_params:,} with BN)")
    print()
    
    # ============================================
    # Test 3: مدل کوچکتر (less overfitting)
    # ============================================
    print("Test 3: Smaller Model (hidden_dim=64)")
    print("-" * 80)
    
    model_small = create_improved_model(
        hidden_dim=64,
        dropout_rate=0.3,  # Dropout بیشتر برای مدل کوچکتر
        use_batchnorm=True
    )
    
    params_small = sum(p.numel() for p in model_small.parameters())
    print(f"✅ Smaller model created")
    print(f"   Parameters: {params_small:,} (vs {total_params:,} standard)")
    print(f"   Reduction: {(1 - params_small/total_params)*100:.1f}%")
    print()
    
    # ============================================
    # Test 4: Action Selection
    # ============================================
    print("Test 4: Action Selection")
    print("-" * 80)
    
    model.eval()
    action, action_prob_val, value = model.select_action(
        network_state[0:1], 
        content_features[0:1], 
        vmaf_predictions[0:1]
    )
    
    print(f"✅ Action selection:")
    print(f"   Selected action: {action}")
    print(f"   Action probability: {action_prob_val:.4f}")
    print(f"   State value: {value:.4f}")
    print()
    
    # ============================================
    # مقایسه سه مدل
    # ============================================
    print("=" * 80)
    print("📊 Model Comparison Summary")
    print("=" * 80)
    print()
    print(f"{'Model':<25} {'Parameters':<15} {'Notes':<30}")
    print("-" * 80)
    print(f"{'Standard (dropout=0.2)':<25} {total_params:>13,}   {'Balanced':<30}")
    print(f"{'No BatchNorm':<25} {params_no_bn:>13,}   {'Fewer params, less stable':<30}")
    print(f"{'Small (hidden=64)':<25} {params_small:>13,}   {'Less overfitting risk':<30}")
    print()
    
    print("=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print()
    
    print("💡 توصیه‌ها:")
    print("   1. برای training اولیه: Standard model با dropout=0.2")
    print("   2. اگر overfitting دیدید: Small model (hidden=64) با dropout=0.3")
    print("   3. برای stability بیشتر: use_batchnorm=True")
    print()
