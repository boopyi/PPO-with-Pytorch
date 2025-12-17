import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

# 1. Setup Environment
env = gym.make("Reacher-v5", render_mode="human")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Re-define Actor (Must match your training architecture)
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(env.observation_space.shape[0],64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU()
            )
        self.mu_head=nn.Linear(64,env.action_space.shape[0])
        self.log_std = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        x=self.net(x)
        mu=self.mu_head(x)
        mu=torch.tanh(mu)*torch.tensor(env.action_space.high).to(device)
        std = torch.exp(self.log_std).expand_as(mu)
        
        return mu,std
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(env.observation_space.shape[0],64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)
            )
    def forward(self,x):
        return self.net(x)

# 3. Load Checkpoint
checkpoint = torch.load("ppo_pendulum_final.pth", map_location=device)

policy = Actor().to(device)
policy.load_state_dict(checkpoint['model_state_dict'])
policy.eval()

# 4. Reconstruct the Normalizer using saved stats
saved_mean = checkpoint['obs_mean']
saved_m2 = checkpoint['obs_m2']
saved_k = checkpoint['obs_k']

def normalize_test_obs(obs, mean, m2, k):
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    variance = m2 / k
    std = torch.sqrt(variance + 1e-8)
    normed = (obs - mean) / std
    return torch.clamp(normed, -5, 5)

# 5. Visualization Loop
obs, _ = env.reset()
for _ in range(1000):
    # Normalize using the FROZEN training stats
    norm_obs = normalize_test_obs(obs, saved_mean, saved_m2, saved_k)
    
    with torch.no_grad():
        # Use mu (mean) for deterministic testing
        action_tensor,_ = policy(norm_obs)
        action_numpy = action_tensor.cpu().numpy().flatten()
    
    obs, reward, terminated, truncated, _ = env.step(action_numpy)
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()