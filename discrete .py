import torch
import torch.nn as nn
import numpy as np


import gymnasium as gym
env_name="CartPole-v1"
env=gym.make(env_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(env.observation_space.shape[0],64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,env.action_space.n),
            nn.Softmax(dim=-1)
            )
    def forward(self,x):
        return self.net(x)
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
def compute_advanteges(values,rewards,terminateds,trunkateds,last_value,gamma=0.99,lam=0.95):
    advanteges=[]
    advantege=0
    values = torch.cat([values, last_value.unsqueeze(0)])
    for t in reversed(range(len(values)-1)):
        
        delta=rewards[t]+gamma*values[t+1]*(1-terminateds[t])-values[t]
        done=terminateds[t]or trunkateds[t]
        advantege=delta+gamma*lam*(1-done)*advantege
        advanteges.insert(0,advantege)
    return torch.tensor(advanteges,dtype=torch.float32).to(device=device)


def update_policy(states, actions, log_probs_old, returns, advantages,batch_size=64,epochs=5,clip_ratio=0.2,c1=0.5,c2=0.01):
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

            probs = policy(s_batch)
            dist = torch.distributions.Categorical(probs)
            logp = dist.log_prob(a_batch)
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
policy=Actor()
value_func=Critic()
policy.to(device)
value_func.to(device)
task=input("train or test?")
if  task=="train":
    num_episodes=100


    episode_count = 0
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    value_optimizer = torch.optim.Adam(value_func.parameters(), lr=1e-3)
    for _ in range(num_episodes):
        states=[]
        actions=[]
        log_probs=[]
        rewards=[]
        terminateds=[]
        trunkateds=[]
        values=[]

        obs,_=env.reset()
        episode_reward = 0

        for i in range(1024):
            obs=torch.tensor(obs,dtype=torch.float32).to(device=device)
            probs=policy.forward(obs)
            dist=torch.distributions.Categorical(probs)
            action=dist.sample()
            action_prob_log=dist.log_prob(action)
            value=value_func.forward(obs)
            states.append(obs)
            actions.append(action.item())
            log_probs.append(action_prob_log.detach())
            values.append(value.squeeze().detach())
            obs, reward, terminated, truncated, _=env.step(action.item())
            
            rewards.append(reward)
            episode_reward+=reward
            trunkateds.append(truncated)
            terminateds.append(terminated)
            if terminated or truncated:
                
                episode_count += 1
                
                obs,_=env.reset()
                if episode_count%20==0:
                    print(f"Episode {episode_count} reward: {episode_reward}")
                episode_reward = 0


        values = torch.stack(values)
        last_obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            last_value = value_func(last_obs_tensor).squeeze()

        advantages=compute_advanteges(values,rewards,terminateds,trunkateds,last_value)
        returns = advantages + values
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        states=torch.stack(states)
        actions=torch.tensor(actions,dtype=torch.long)
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
    torch.save(policy.state_dict(), "ppo_discrete_final.pth")
elif task=="test":
    env=gym.make(env_name,render_mode="human")
    state_dict=torch.load("ppo_discrete_final.pth")
    policy.load_state_dict(state_dict)
    policy.eval()
    obs, _ = env.reset()
    
    for _ in range(1000):
        obs=torch.tensor(obs,dtype=torch.float32).to(device)
        
        with torch.no_grad():
            probs = policy(obs)
            action=probs.argmax().item()
            
            
            
            
            
        
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

