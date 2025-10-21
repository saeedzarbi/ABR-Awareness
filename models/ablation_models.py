"""
models/ablation_models.py
==========================
Modified models for ablation study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentAwareActor(nn.Module):
    """
    Full model - با همه features
    (این رو از قبل دارید)
    """
    def __init__(self, state_dim=(6, 8), action_dim=6, content_dim=2):
        super().__init__()
        
        # Network state encoder (1D-CNN)
        self.conv1 = nn.Conv1d(state_dim[0], 128, kernel_size=4, stride=1)
        self.fc_network = nn.Linear(128 * (state_dim[1] - 3), 128)
        
        # Content features encoder (SI/TI)
        self.fc_content = nn.Linear(content_dim, 64)
        
        # VMAF predictions encoder
        self.fc_vmaf = nn.Linear(action_dim, 64)
        
        # Combined layers
        self.fc1 = nn.Linear(128 + 64 + 64, 128)  # network + content + vmaf
        self.fc_actor = nn.Linear(128, action_dim)
        self.fc_critic = nn.Linear(128, 1)
    
    def forward(self, network_state, content_state, vmaf_state):
        # Network encoding
        x = F.relu(self.conv1(network_state))
        x = x.view(x.size(0), -1)
        x_net = F.relu(self.fc_network(x))
        
        # Content encoding
        x_cont = F.relu(self.fc_content(content_state))
        
        # VMAF encoding
        x_vmaf = F.relu(self.fc_vmaf(vmaf_state))
        
        # Combine all
        x = torch.cat([x_net, x_cont, x_vmaf], dim=1)
        x = F.relu(self.fc1(x))
        
        # Actor & Critic
        action_probs = F.softmax(self.fc_actor(x), dim=-1)
        state_value = self.fc_critic(x)
        
        return action_probs, state_value


class AblatedActor_NoSITI(nn.Module):
    """
    Ablation 1: بدون SI/TI features
    Network + VMAF فقط
    """
    def __init__(self, state_dim=(6, 8), action_dim=6, content_dim=2):
        super().__init__()
        
        # Network state encoder
        self.conv1 = nn.Conv1d(state_dim[0], 128, kernel_size=4, stride=1)
        self.fc_network = nn.Linear(128 * (state_dim[1] - 3), 128)
        
        # VMAF encoder (keep)
        self.fc_vmaf = nn.Linear(action_dim, 64)
        
        # Combined (network + vmaf only)
        self.fc1 = nn.Linear(128 + 64, 128)  # NO content!
        self.fc_actor = nn.Linear(128, action_dim)
        self.fc_critic = nn.Linear(128, 1)
    
    def forward(self, network_state, content_state, vmaf_state):
        # Network encoding
        x = F.relu(self.conv1(network_state))
        x = x.view(x.size(0), -1)
        x_net = F.relu(self.fc_network(x))
        
        # VMAF encoding
        x_vmaf = F.relu(self.fc_vmaf(vmaf_state))
        
        # Combine (ignore content_state)
        x = torch.cat([x_net, x_vmaf], dim=1)
        x = F.relu(self.fc1(x))
        
        # Actor & Critic
        action_probs = F.softmax(self.fc_actor(x), dim=-1)
        state_value = self.fc_critic(x)
        
        return action_probs, state_value


class AblatedActor_NoVMAF(nn.Module):
    """
    Ablation 2: بدون VMAF predictions
    Network + SI/TI فقط
    """
    def __init__(self, state_dim=(6, 8), action_dim=6, content_dim=2):
        super().__init__()
        
        # Network state encoder
        self.conv1 = nn.Conv1d(state_dim[0], 128, kernel_size=4, stride=1)
        self.fc_network = nn.Linear(128 * (state_dim[1] - 3), 128)
        
        # Content encoder (keep)
        self.fc_content = nn.Linear(content_dim, 64)
        
        # Combined (network + content only)
        self.fc1 = nn.Linear(128 + 64, 128)  # NO vmaf!
        self.fc_actor = nn.Linear(128, action_dim)
        self.fc_critic = nn.Linear(128, 1)
    
    def forward(self, network_state, content_state, vmaf_state):
        # Network encoding
        x = F.relu(self.conv1(network_state))
        x = x.view(x.size(0), -1)
        x_net = F.relu(self.fc_network(x))
        
        # Content encoding
        x_cont = F.relu(self.fc_content(content_state))
        
        # Combine (ignore vmaf_state)
        x = torch.cat([x_net, x_cont], dim=1)
        x = F.relu(self.fc1(x))
        
        # Actor & Critic
        action_probs = F.softmax(self.fc_actor(x), dim=-1)
        state_value = self.fc_critic(x)
        
        return action_probs, state_value


class AblatedActor_NetworkOnly(nn.Module):
    """
    Ablation 3: بدون هیچ content feature
    Network state فقط (مثل Pensieve)
    """
    def __init__(self, state_dim=(6, 8), action_dim=6, content_dim=2):
        super().__init__()
        
        # Network state encoder only
        self.conv1 = nn.Conv1d(state_dim[0], 128, kernel_size=4, stride=1)
        self.fc_network = nn.Linear(128 * (state_dim[1] - 3), 128)
        
        # Direct to output (no content, no vmaf)
        self.fc1 = nn.Linear(128, 128)
        self.fc_actor = nn.Linear(128, action_dim)
        self.fc_critic = nn.Linear(128, 1)
    
    def forward(self, network_state, content_state, vmaf_state):
        # Network encoding only
        x = F.relu(self.conv1(network_state))
        x = x.view(x.size(0), -1)
        x_net = F.relu(self.fc_network(x))
        
        # Direct processing (ignore content and vmaf)
        x = F.relu(self.fc1(x_net))
        
        # Actor & Critic
        action_probs = F.softmax(self.fc_actor(x), dim=-1)
        state_value = self.fc_critic(x)
        
        return action_probs, state_value


# ============================================
# Helper: Load appropriate model
# ============================================
def load_ablated_model(ablation_type='full', checkpoint_path=None, device='cpu'):
    """
    Load ablated model
    
    Args:
        ablation_type: 'full', 'no_siti', 'no_vmaf', 'network_only'
        checkpoint_path: Path to checkpoint (if loading pretrained)
        device: torch device
    
    Returns:
        model
    """
    if ablation_type == 'full':
        model = ContentAwareActor(state_dim=(6,8), action_dim=6, content_dim=2)
    elif ablation_type == 'no_siti':
        model = AblatedActor_NoSITI(state_dim=(6,8), action_dim=6, content_dim=2)
    elif ablation_type == 'no_vmaf':
        model = AblatedActor_NoVMAF(state_dim=(6,8), action_dim=6, content_dim=2)
    elif ablation_type == 'network_only':
        model = AblatedActor_NetworkOnly(state_dim=(6,8), action_dim=6, content_dim=2)
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
    
    model = model.to(device)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    
    return model


# ============================================
# Test
# ============================================
if __name__ == '__main__':
    print("="*80)
    print("Testing Ablation Models")
    print("="*80)
    
    batch_size = 4
    network = torch.randn(batch_size, 6, 8)
    content = torch.randn(batch_size, 2)
    vmaf = torch.randn(batch_size, 6)
    
    models = {
        'Full Model': ContentAwareActor(),
        'No SI/TI': AblatedActor_NoSITI(),
        'No VMAF': AblatedActor_NoVMAF(),
        'Network Only': AblatedActor_NetworkOnly()
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        probs, value = model(network, content, vmaf)
        print(f"   Output shape: probs={probs.shape}, value={value.shape}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "="*80)
    print("✅ All ablation models work correctly!")
    print("="*80)