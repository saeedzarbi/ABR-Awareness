"""
Policy Wrapper با Safety Rules
"""

import torch
import numpy as np


class BufferAwarePolicy:
    """
    Wrap trained model با buffer-aware safety rules
    """
    
    def __init__(self, model, bitrate_levels=[300, 750, 1850, 2850, 4300, 6000]):
        self.model = model
        self.bitrate_levels = bitrate_levels
        self.model.eval()
        
        # Tracking
        self.recent_rebuffers = []
        self.last_action = None
    
    def select_action(self, state, buffer, recent_rebuffer_time=0):
        """
        Select action با safety overrides
        
        Args:
            state: environment state dict
            buffer: current buffer level (seconds)
            recent_rebuffer_time: rebuffer in last chunk (seconds)
        
        Returns:
            action: selected action (0-5)
        """
        
        # Get model prediction
        network_state = torch.FloatTensor(state['network']).unsqueeze(0)
        content_features = torch.FloatTensor(state['content']).unsqueeze(0)
        vmaf_predictions = torch.FloatTensor(state['vmaf']).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, _ = self.model(network_state, content_features, vmaf_predictions)
            model_action = action_probs.argmax(dim=1).item()
        
        # Track recent rebuffering
        self.recent_rebuffers.append(recent_rebuffer_time)
        if len(self.recent_rebuffers) > 5:
            self.recent_rebuffers.pop(0)
        
        total_recent_rebuffer = sum(self.recent_rebuffers)
        
        # Safety rules
        action = self._apply_safety_rules(
            model_action, 
            buffer, 
            total_recent_rebuffer
        )
        
        self.last_action = action
        return action
    
    def _apply_safety_rules(self, model_action, buffer, total_recent_rebuffer):
        """
        Apply safety overrides
        """
        
        # Rule 1: Emergency - very low buffer
        if buffer < 2.0:
            return 0  # Force lowest bitrate
        
        # Rule 2: Critical - low buffer
        if buffer < 5.0 and model_action > 1:
            return min(model_action, 1)  # Max 750 kbps
        
        # Rule 3: Recent heavy rebuffering
        if total_recent_rebuffer > 8.0 and model_action > 2:
            # Be more conservative
            return max(0, model_action - 1)
        
        if total_recent_rebuffer > 15.0:
            # Very heavy rebuffering - go safe
            return min(model_action, 1)
        
        # Rule 4: Good buffer - can be more aggressive
        if buffer > 35.0 and total_recent_rebuffer < 2.0:
            # Good conditions - can try higher
            if model_action < 4:
                return min(5, model_action + 1)
        
        # Rule 5: Excellent buffer - allow highest
        if buffer > 45.0 and total_recent_rebuffer == 0:
            if model_action < 5:
                return min(5, model_action + 1)
        
        # No override needed
        return model_action
    
    def reset(self):
        """Reset tracking"""
        self.recent_rebuffers = []
        self.last_action = None


class SmoothPolicy:
    """
    Wrap policy با bitrate smoothing
    """
    
    def __init__(self, base_policy, max_jump=2):
        self.base_policy = base_policy
        self.max_jump = max_jump
        self.last_action = None
    
    def select_action(self, state, buffer, recent_rebuffer_time=0):
        """
        Select action با smoothing constraint
        """
        
        # Get base policy action
        action = self.base_policy.select_action(state, buffer, recent_rebuffer_time)
        
        # Apply smoothing
        if self.last_action is not None:
            jump = abs(action - self.last_action)
            
            if jump > self.max_jump:
                # Limit jump size
                if action > self.last_action:
                    action = self.last_action + self.max_jump
                else:
                    action = self.last_action - self.max_jump
        
        self.last_action = action
        return action
    
    def reset(self):
        """Reset tracking"""
        self.last_action = None
        if hasattr(self.base_policy, 'reset'):
            self.base_policy.reset()


# Test
if __name__ == '__main__':
    print("Policy Wrapper module loaded!")
    print("\nUsage:")
    print("  buffer_policy = BufferAwarePolicy(model)")
    print("  smooth_policy = SmoothPolicy(buffer_policy, max_jump=2)")
