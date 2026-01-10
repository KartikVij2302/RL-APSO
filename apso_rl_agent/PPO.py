import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class Normalizer:
    def __init__(self, n_inputs):
        self.n = 0
        self.mean = np.zeros(n_inputs)
        self.mean_diff = np.zeros(n_inputs)
        self.var = np.ones(n_inputs)

    def observe(self, x):
        self.n += 1
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        if self.n > 1: self.var = self.mean_diff / (self.n - 1)
    
    def normalize(self, inputs):
        obs_std = np.sqrt(self.var + 1e-8)
        return (inputs - self.mean) / obs_std

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # MLP Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Output [-1, 1]
        )
        
        # MLP Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)

    def act(self, state):
        action_mean = self.actor(state)
        std = self.log_std.exp()
        dist = Normal(action_mean, std)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).sum(dim=-1).detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        state_values = self.critic(state)
        std = self.log_std.exp()
        dist = Normal(action_mean, std)
        return dist.log_prob(action).sum(dim=-1), state_values, dist.entropy().sum(dim=-1)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = []
        self.K_epochs = 10
        self.eps_clip = 0.2
        self.gamma = 0.99

    def select_action(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            action, logprob = self.policy_old.act(state)
        return action.cpu().numpy(), logprob

    def store(self, s, a, lp, r, d):
        self.buffer.append((s, a, lp, r, d))

    def update(self):
        if not self.buffer: return
        
        # Convert buffer to tensors
        s = torch.tensor(np.array([t[0] for t in self.buffer]), dtype=torch.float32)
        a = torch.tensor(np.array([t[1] for t in self.buffer]), dtype=torch.float32)
        lp = torch.tensor(np.array([t[2] for t in self.buffer]), dtype=torch.float32)
        r = [t[3] for t in self.buffer]
        d = [t[4] for t in self.buffer]
        
        rewards = []
        disc_r = 0
        for reward, done in zip(reversed(r), reversed(d)):
            if done: disc_r = 0
            disc_r = reward + (self.gamma * disc_r)
            rewards.insert(0, disc_r)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        if rewards.std() > 1e-5:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        else:
            rewards = rewards - rewards.mean()

        for _ in range(self.K_epochs):
            logprobs, values, dist_entropy = self.policy.evaluate(s, a)
            values = values.squeeze()
            
            ratios = torch.exp(logprobs - lp)
            adv = rewards - values.detach()
            
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * adv
            
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = []