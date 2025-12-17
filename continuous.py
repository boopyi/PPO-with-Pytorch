import torch
import torch.nn as nn
import numpy as np


import gymnasium as gym
num_envs = 8
env_name="InvertedDoublePendulum-v5"

env = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
class obs_normalizer:
    def __init__(self):
        self.k=1e-4
        self.mean=torch.zeros(num_envs,env.single_observation_space.shape[0]).to(device)
        self.m2=torch.zeros(num_envs,env.single_observation_space.shape[0]).to(device)
    def update(self,obs):
        with torch.no_grad():
            self.k+=1
            old_mean=self.mean
            self.mean+=(obs-old_mean)/self.k
            self.m2+=(obs-self.mean)*(obs-old_mean)
            variance=self.m2/self.k
            std=torch.sqrt(variance+1e-8)
            normed_obs=(obs-self.mean)/std
            return torch.clamp(normed_obs, -5, 5)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(env.single_observation_space.shape[0],512 ),
            nn.Tanh(),
            nn.Linear(512 ,512 ),
            nn.Tanh(),
            nn.Linear(512 ,512 ),
            nn.Tanh()
            )
        self.mu_head=nn.Linear(512 ,env.single_action_space.shape[0])
        self.log_std = nn.Parameter(torch.zeros(env.single_action_space.shape[0]))
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
            nn.Linear(env.single_observation_space.shape[0],512 ),
            nn.ReLU(),
            nn.Linear(512 ,512 ),
            nn.ReLU(),
            nn.Linear(512 ,1)
            )
    def forward(self,x):
        return self.net(x)
def compute_advanteges(values,rewards,terminateds,trunkateds,last_value,gamma=0.99,lam=0.95):
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
    advanteges=torch.zeros_like(rewards).to(device)
    advantege=0
    values = torch.cat([values, last_value.unsqueeze(0)])
    terminateds=torch.from_numpy(np.array(terminateds)).to(torch.float32).to(device)
    for t in reversed(range(len(values)-1)):
        done=terminateds[t].float()
        
        delta=rewards[t]+gamma*values[t+1]*(1-done)-values[t]
        advantege=delta+gamma*lam*(1-done)*advantege
        advanteges[t]=advantege
    return advanteges

policy=Actor()
value_func=Critic()
normilizer=obs_normalizer()

policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
value_optimizer = torch.optim.Adam(value_func.parameters(), lr=1e-3)
policy.to(device)
value_func.to(device)
def update_policy(states, actions, log_probs_old, returns, advantages,batch_size=1024,epochs=10,clip_ratio=0.2,c1=0.5,c2=0.02):
    dataset_size = len(states)
    for _ in range(epochs):
        indices = torch.randperm(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end=start+batch_size
            batch_idx = indices[start:end]
            s_batch = states[batch_idx]
            a_batch = actions[batch_idx]
            old_log_prob_batch=log_probs_old[batch_idx]
            r_batch=returns[batch_idx]
            ad_batch=advantages[batch_idx]

            mu,std = policy(s_batch)
            dist = torch.distributions.Normal(mu,std)
            logp = dist.log_prob(a_batch).sum(dim=-1)
            ratio=torch.exp(logp-old_log_prob_batch)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * ad_batch
            policy_loss = -torch.min(ratio * ad_batch, clip_adv).mean()
            entropy = dist.entropy().mean()
            
            policy_loss -= c2 * entropy
            value_pred = value_func(s_batch).squeeze()
            value_loss = c1 * (r_batch - value_pred).pow(2).mean()
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
            
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

num_episodes=500

task=input("train or test?\n")

if  task=="train":
    try:
        episode_count = 0
        for _ in range(num_episodes):
            states=[]
            actions=[]
            log_probs=[]
            rewards=[]
            terminateds=[]
            trunkateds=[]
            values=[]

            obs,_=env.reset()
            episode_reward=torch.zeros(num_envs,dtype=torch.float32)
            low = torch.tensor(env.action_space.low).to(device)
            high = torch.tensor(env.action_space.high).to(device)
            for i in range(4096):
                with torch.no_grad():
                    obs=torch.from_numpy(obs).float().to(device=device)
                    obs=normilizer.update(obs)
                    mu,std=policy.forward(obs)
                    dist=torch.distributions.Normal(mu,std)
                    action=dist.sample()
                    
                    
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    value=value_func.forward(obs)
                    states.append(obs)
                    actions.append(action.cpu().numpy())
                    log_probs.append(log_prob.detach())
                    values.append(value.squeeze().detach())
                    action_env = torch.clamp(action, low, high).cpu().numpy()
                    obs, reward, terminated, truncated, _=env.step(action_env)
                    
                    rewards.append(reward)
                    episode_reward+=reward
                    trunkateds.append(truncated)
                    terminateds.append(terminated)

            episode_count+=1
            
            print(f"episode: {episode_count}, reward: {episode_reward.mean()/num_envs}")
            episode_reward=torch.zeros(num_envs,dtype=torch.float32)
            values = torch.stack(values)

            with torch.no_grad():
                final_obs_tensor = torch.from_numpy(obs).float().to(device)
                variance = normilizer.m2 / normilizer.k
                std = torch.sqrt(variance + 1e-8)
                normed_last_obs = torch.clamp((final_obs_tensor - normilizer.mean) / std, -5, 5)
                last_value = value_func(normed_last_obs).squeeze()
            
            advantages=compute_advanteges(values,rewards,terminateds,trunkateds,last_value)
            returns = advantages + values
            advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
            states=torch.stack(states)
            actions = torch.from_numpy(np.array(actions)).to(torch.float32).to(device)
            log_probs=torch.stack(log_probs)
            policy.to(device)
            value_func.to(device)
            states = states.to(device)
            actions = actions.to(device)
            log_probs = log_probs.to(device)
            returns = returns.to(device)
            advantages = advantages.to(device)
            advantages = advantages.detach()
            returns = returns.detach()


            update_policy(states,actions,log_probs,returns,advantages)
        
        torch.save({
            'model_state_dict': policy.state_dict(),
            'obs_mean': normilizer.mean,
            'obs_m2': normilizer.m2,
            'obs_k': normilizer.k
        }, "ppo_continous_final.pth")
    except KeyboardInterrupt:
        if input("save?(y/n)\n")=="y":
            torch.save({
                'model_state_dict': policy.state_dict(),
                'obs_mean': normilizer.mean,
                'obs_m2': normilizer.m2,
                'obs_k': normilizer.k
            }, "ppo_continous_final.pth")
            print("saved")
elif task=="test":
    env=gym.make(env_name,render_mode="human")
    checkpoint = torch.load("ppo_continous_final.pth", map_location=device)

    policy = policy.to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    mean=checkpoint["obs_mean"]
    m2=checkpoint["obs_m2"]
    k=checkpoint["obs_k"]
    variance=m2/k
    std=torch.sqrt(variance+1e-8)
    obs, _ = env.reset()

    
    for _ in range(1000):
        obs=torch.tensor(obs,dtype=torch.float32).to(device)
        norm_obs=(obs-mean)/std
        norm_obs=torch.clamp(norm_obs, -5, 5)
        
        with torch.no_grad():
            action_tensor,_ = policy(norm_obs)
            action_numpy = action_tensor.cpu().numpy().flatten()
        
        obs, reward, terminated, truncated, _ = env.step(action_numpy)
        
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    