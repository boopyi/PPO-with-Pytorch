import torch
import torch.nn as nn
import numpy as np
import time
import arm_env
import gymnasium as gym
import cv2
num_envs = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    env = gym.vector.AsyncVectorEnv([lambda: arm_env.Env() for _ in range(num_envs)])
    print(device)
class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channel):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, self.width),
            torch.linspace(-1, 1, self.height),
            indexing='ij'
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, x):

        x = x.view(x.size(0), x.size(1), -1)
        softmax_attention = torch.softmax(x, dim=-1)
        
        expected_x = torch.sum(softmax_attention * self.pos_x, dim=-1)
        expected_y = torch.sum(softmax_attention * self.pos_y, dim=-1)
        
        return torch.cat([expected_x, expected_y], dim=-1)
class obs_normalizer:
    def __init__(self,shape):
        self.k=1e-4
        self.mean = torch.zeros(shape).to(device)
        self.m2 = torch.zeros(shape).to(device)
    def update(self,obs):
        with torch.no_grad():
            batch_mean = obs.mean(dim=0)
            self.k += 1
            old_mean = self.mean
            self.mean += (batch_mean - old_mean) / self.k
            self.m2 += (batch_mean - self.mean) * (batch_mean - old_mean)
            
            variance = self.m2 / self.k
            std = torch.sqrt(variance + 1e-8)
            normed_obs = (obs - self.mean) / std
            return torch.clamp(normed_obs, -5, 5)
class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.spatial_softmax = SpatialSoftmax(8, 8, 64)
        self.repr_dim = 128 

    def forward(self, x):
        x = x.float()
        x = self.conv(x)
        return self.spatial_softmax(x)
class Actor(nn.Module):
    def __init__(self,encoder):
        super().__init__()

        self.encoder=encoder
        self.net=nn.Sequential(
            nn.Linear(env.single_observation_space["obs"].shape[0]+self.encoder.repr_dim,512 ),
            nn.Tanh(),
            nn.Linear(512 ,512 ),
            nn.Tanh(),
            nn.Linear(512 ,512 ),
            nn.Tanh()
            )
        self.mu_head=nn.Linear(512 ,env.single_action_space.shape[0])
        self.log_std = nn.Parameter(torch.zeros(env.single_action_space.shape[0]))
        high=torch.tensor(env.single_action_space.high,dtype=torch.float32).to(device)
        self.register_buffer("high",high)
    def forward(self,obs,image):
        
        image=(image.float()).to(device)
        
        obs=obs.to(device)
        
        x=self.net(torch.cat((self.encoder(image),obs),dim=1))
        mu=self.mu_head(x)
        mu=torch.tanh(mu)*self.high
        std = torch.exp(self.log_std).expand_as(mu)
        
        return mu,std
class Critic(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder=encoder
        self.net=nn.Sequential(
            nn.Linear(env.single_observation_space["obs"].shape[0]+self.encoder.repr_dim,512 ),
            nn.ReLU(),
            nn.Linear(512 ,512 ),
            nn.ReLU(),
            nn.Linear(512 ,1)
            )
    def forward(self,obs,image):
        image=(image.float()).to(device)
        
        obs=obs.to(device)
        visual_features = self.encoder(image)
        return self.net(torch.cat((visual_features, obs), dim=1))
if __name__ == '__main__':
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
    shared_encoder = VisualEncoder().to(device)
    policy = Actor(shared_encoder).to(device)
    value_func = Critic(shared_encoder).to(device)
    normilizer = obs_normalizer(env.single_observation_space["obs"].shape)
    optimizer_actor = torch.optim.Adam(policy.parameters(), lr=5e-5)
    optimizer_critic = torch.optim.Adam(value_func.parameters(), lr=1e-4)

    def update_policy(s_image, s_obs, actions, log_probs_old, returns, advantages, batch_size=512, epochs=10):
        dataset_size = len(actions)
        
        for epoch in range(epochs):
            indices = torch.randperm(dataset_size,device=device)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                img_b = s_image[batch_idx]
                obs_b = s_obs[batch_idx]
                a_b = actions[batch_idx]
                old_logp_b = log_probs_old[batch_idx]
                ret_b = returns[batch_idx]
                adv_b = advantages[batch_idx]


                mu, std = policy(obs_b, img_b)
                dist = torch.distributions.Normal(mu, std)
                logp = dist.log_prob(a_b).sum(dim=-1)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 0.9, 1.1) * adv_b
                
                policy_loss = -torch.min(surr1, surr2).mean()

                actor_total_loss = policy_loss - (0.01 * entropy) 

                optimizer_actor.zero_grad(set_to_none=True)
                actor_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer_actor.step()

            
                value_pred = value_func(obs_b, img_b).squeeze()

                value_loss = torch.nn.functional.huber_loss(value_pred, ret_b, delta=1.0)

                optimizer_critic.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(value_func.parameters(), 0.3) 
                optimizer_critic.step()
                

                
        print(f"Final Epoch Entropy: {entropy.item():.4f}, Value Loss: {value_loss.item():.4f}")

    num_episodes=500
    good_reward=100

    task=input("train or test?\n")

    if  task=="train":
        try:
            episode_count = 0
            total_steps = 0
            storage_images = torch.zeros((1024, num_envs, 4, 64, 64), dtype=torch.uint8, device=device)
            storage_obs = torch.zeros((1024, num_envs, 8), device=device)
            for _ in range(num_episodes):
                total_successes = 0
                ep_start_time=time.time()
                obses=[]
                images=[]
                actions=[]
                log_probs=[]
                rewards=[]
                terminateds=[]
                trunkateds=[]
                values=[]

                obs,_=env.reset()
                episode_reward=torch.zeros(num_envs,dtype=torch.float32).to(device)
                low = torch.tensor(env.action_space.low).to(device)
                high = torch.tensor(env.action_space.high).to(device)
                for i in range(1024):
                    storage_images[i] = torch.from_numpy(obs["image"]).to(device, dtype=torch.uint8)
                    storage_obs[i] = torch.from_numpy(obs["obs"]).to(device, dtype=torch.float32)
                    if i % 100 == 0:
                        print(f"Step {i}/1024...", end="\r")
                    with torch.no_grad():
                        curr_obs = storage_obs[i]

                        normed_obs = normilizer.update(curr_obs)
                        curr_img = storage_images[i]

                        mu, std = policy.forward(normed_obs, curr_img)
                        dist = torch.distributions.Normal(mu, std)
                        action = dist.sample()
                        
                        
                        log_prob = dist.log_prob(action).sum(dim=-1)
                        value = value_func.forward(normed_obs, curr_img)
                        obses.append(normed_obs.cpu())
                        images.append(curr_img.cpu())
                        actions.append(action.cpu().numpy())
                        log_probs.append(log_prob.detach())
                        values.append(value.squeeze().detach())
                        low = torch.tensor(env.action_space.low).to(device)
                        high = torch.tensor(env.action_space.high).to(device)
                        action_env = torch.clamp(action, low, high).cpu().numpy()
                        obs, reward, terminated, truncated, _ = env.step(action_env)
                        
                        rewards.append(reward)
                        episode_reward += torch.from_numpy(reward).to(device)
                        terminateds.append(terminated)
                obs["image"]=torch.from_numpy(obs["image"]).to(torch.uint8)
                episode_count+=1
                curr_steps=1024*num_envs
                total_steps+=curr_steps
                ep_end_time=time.time()
                sps=curr_steps/(ep_end_time-ep_start_time)
                total_successes += np.sum(terminated)
                success_rate = total_successes / num_envs
                print(f"episode: {episode_count}, steps: {total_steps}, sps: {sps}, reward: {episode_reward.mean()}, Goal Hits: {total_successes}")

                if episode_reward.mean()>good_reward and total_successes>0:
                    good_reward+=100
                    torch.save({
                        'model_state_dict': policy.state_dict(),
                        'obs_mean': normilizer.mean,
                        'obs_m2': normilizer.m2,
                        'obs_k': normilizer.k
                    }, fr"C:\Users\User\Desktop\vs_code_projects\learn-ppo\models\ppo_continous_step{total_steps}_reward{episode_reward.mean()}.pth")
                episode_reward=torch.zeros(num_envs,dtype=torch.float32)
                values = torch.stack(values)

                with torch.no_grad():
                    final_obs_tensor = torch.from_numpy(obs["obs"]).float().to(device)
                    variance = normilizer.m2 / normilizer.k
                    std = torch.sqrt(variance + 1e-8)
                    normed_last_obs = torch.clamp((final_obs_tensor - normilizer.mean) / std, -5, 5)
                    final_img = obs["image"]
                    if not isinstance(final_img, torch.Tensor):
                        final_img = torch.from_numpy(final_img)
                    final_img = final_img.to(device)
                    last_value = value_func.forward(normed_last_obs, final_img).squeeze()
                
                
                advantages = compute_advanteges(values, rewards, terminateds, trunkateds, last_value)
                returns = advantages + values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                
                obses_tensor = torch.stack(obses).to(device).view(-1, env.single_observation_space["obs"].shape[0])
                actions_tensor = torch.from_numpy(np.array(actions)).to(device, dtype=torch.float32).view(-1, env.single_action_space.shape[0])
                log_probs_tensor = torch.stack(log_probs).to(device).view(-1)
               
                images_batch = storage_images.view(-1, 4, 64, 64)
                
                returns_flat = returns.view(-1).detach()
                advantages_flat = advantages.view(-1).detach()

                update_policy(images_batch, obses_tensor, actions_tensor, log_probs_tensor, returns_flat, advantages_flat)
                
                obses, images, actions, log_probs = [], [], [], []
                rewards, terminateds, values = [], [], []
                storage_images.zero_()
                storage_obs.zero_()
            
            torch.save({
                'model_state_dict': policy.state_dict(),
                'obs_mean': normilizer.mean,
                'obs_m2': normilizer.m2,
                'obs_k': normilizer.k
            }, r"C:\Users\User\Desktop\vs_code_projects\learn-ppo\models\ppo_continous_final.pth")
        except KeyboardInterrupt:
            if input("save?(y/n)\n")=="y":
                torch.save({
                    'model_state_dict': policy.state_dict(),
                    'obs_mean': normilizer.mean,
                    'obs_m2': normilizer.m2,
                    'obs_k': normilizer.k
                }, r"C:\Users\User\Desktop\vs_code_projects\learn-ppo\models\ppo_continous_final.pth")
                print("saved")
    elif task=="test":
        arm_env.p.disconnect()
        frame_duration = 15 * (1/240)
        env=arm_env.Env(render_mode="human")
        checkpoint = torch.load(r"C:\Users\User\Desktop\vs_code_projects\learn-ppo\models\ppo_continous_final.pth", map_location=device,weights_only=True)

        policy = policy.to(device)
        policy.load_state_dict(checkpoint['model_state_dict'])
        policy.eval()
        mean = checkpoint["obs_mean"].cpu().flatten()
        m2 = checkpoint["obs_m2"].cpu().flatten()
        k = checkpoint["obs_k"]
        std = torch.sqrt(m2 / k + 1e-8).flatten()
        variance=m2/k
        if std.dim() > 1:
            std = std[0]
        obs, _ = env.reset()

        low = torch.tensor(env.action_space.low).to(device)
        high = torch.tensor(env.action_space.high).to(device)
        for i in range(1000):
            
            time.sleep(frame_duration)
            obs["obs"]=torch.from_numpy(obs["obs"]).float()

            obs["image"]=torch.from_numpy(obs["image"]).to(torch.uint8).unsqueeze(dim=0)

            obs["obs"]=(obs["obs"]-mean)/std
            obs["obs"]=torch.clamp(obs["obs"], -5, 5).unsqueeze(dim=0)
            
            with torch.no_grad():
                
                mu,stdd = policy(obs["obs"],obs["image"])
                dist = torch.distributions.Normal(mu,stdd)
                
                action=dist.sample()
                
                env_action = torch.clamp(action, low, high).cpu().numpy().flatten()
                
                
            obs, reward, terminated, truncated, _ = env.step(env_action)
            
            if terminated or truncated:
                print("touch")
                obs, _ = env.reset()

        env.close()
        