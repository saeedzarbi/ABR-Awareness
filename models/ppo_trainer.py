"""
PPO Trainer - FIXED VERSION
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def __len__(self):
        return len(self.states)
    
    def get(self):
        return {
            'states': self.states,
            'actions': torch.tensor(self.actions, dtype=torch.long),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'values': torch.tensor(self.values, dtype=torch.float32),
            'log_probs': torch.tensor(self.log_probs, dtype=torch.float32),
            'dones': torch.tensor(self.dones, dtype=torch.float32)
        }


class PPOTrainer:
    def __init__(self, model, env, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, n_epochs=4, batch_size=64):
        self.model = model
        self.env = env
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.external_logger = None
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        returns = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_val = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_val * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)
    
    def collect_rollout(self, n_steps=2048, video_ids=[1,2,3,4,5,6]):
        buffer = RolloutBuffer()
        state = self.env.reset(video_id=np.random.choice(video_ids), split='train')
        
        episode_reward = 0
        episode_length = 0
        episode_rebuffer = 0
        episode_bitrates = []
        
        for step in range(n_steps):
            network_state = torch.FloatTensor(state['network']).unsqueeze(0)
            content_features = torch.FloatTensor(state['content']).unsqueeze(0)
            vmaf_predictions = torch.FloatTensor(state['vmaf']).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, value = self.model(network_state, content_features, vmaf_predictions)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            next_state, reward, done, info = self.env.step(action.item())
            reward = float(np.clip(reward, -50.0, 50.0))

            buffer.add(state, action.item(), reward, value.item(), log_prob.item(), done)
            
            episode_reward += reward
            episode_length += 1
            episode_rebuffer += info['rebuffer_time']
            episode_bitrates.append(info['bitrate'])
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                if self.external_logger is not None:
                    self.external_logger.log_episode(
                        reward=episode_reward,
                        length=episode_length,
                        rebuffer=episode_rebuffer,
                        avg_bitrate=np.mean(episode_bitrates) if episode_bitrates else 0
                    )
                
                state = self.env.reset(video_id=np.random.choice(video_ids), split='train')
                episode_reward = 0
                episode_length = 0
                episode_rebuffer = 0
                episode_bitrates = []
            else:
                state = next_state
        
        return buffer
    
    def update_policy(self, buffer):
        data = buffer.get()
        states = data['states']
        actions = data['actions']
        old_log_probs = data['log_probs']
        rewards = data['rewards']
        values = data['values']
        dones = data['dones']
        
        with torch.no_grad():
            last_state = states[-1]
            network_state = torch.FloatTensor(last_state['network']).unsqueeze(0)
            content_features = torch.FloatTensor(last_state['content']).unsqueeze(0)
            vmaf_predictions = torch.FloatTensor(last_state['vmaf']).unsqueeze(0)
            _, next_value = self.model(network_state, content_features, vmaf_predictions)
            next_value = next_value.item()
        
        advantages, returns = self.compute_gae(
            rewards.tolist(), values.tolist(), dones.tolist(), next_value
        )
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                if len(batch_indices) < self.batch_size // 2:
                    continue
                
                batch_states = [states[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                network_states = torch.stack([torch.FloatTensor(s['network']) for s in batch_states])
                content_features = torch.stack([torch.FloatTensor(s['content']) for s in batch_states])
                vmaf_predictions = torch.stack([torch.FloatTensor(s['vmaf']) for s in batch_states])
                
                action_probs, values = self.model(network_states, content_features, vmaf_predictions)
                
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                values = values.squeeze()
                value_loss = nn.MSELoss()(values, batch_returns)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        return {
            'policy_loss': total_policy_loss / n_updates if n_updates > 0 else 0,
            'value_loss': total_value_loss / n_updates if n_updates > 0 else 0,
            'entropy': total_entropy / n_updates if n_updates > 0 else 0,
        }
    
    def train(self, total_timesteps=100000, rollout_length=2048, log_interval=10):
        print("=" * 70)
        print("PPO Training")
        print("=" * 70)
        
        timesteps = 0
        update = 0
        
        while timesteps < total_timesteps:
            buffer = self.collect_rollout(n_steps=rollout_length)
            timesteps += len(buffer)
            
            stats = self.update_policy(buffer)
            update += 1
            
            if update % log_interval == 0:
                recent = self.episode_rewards[-20:] if len(self.episode_rewards) >= 20 else self.episode_rewards
                avg_reward = np.mean(recent) if recent else 0
                
                print(f"Update {update:4d} | Steps: {timesteps:6d} | Reward: {avg_reward:7.2f}")
        
        print("\n✓ Training complete!")
        return self.episode_rewards


if __name__ == '__main__':
    print("PPO Trainer loaded!")
