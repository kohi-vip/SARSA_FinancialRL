import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from collections import deque
import random


class PolicyNetwork(nn.Module):
    """Neural network for policy gradient agent"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class PolicyGradientAgent:
    """
    Policy Gradient Agent for Financial Trading
    Based on REINFORCE algorithm with baseline
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device

        # Policy network
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Experience storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

        # For baseline (value function approximation)
        self.baseline_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        self.baseline_optimizer = optim.Adam(self.baseline_net.parameters(), lr=lr)

    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.policy_net(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def store_transition(self, state, action, reward, log_prob):
        """Store transition for later policy update"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def compute_baseline(self, states):
        """Compute baseline values for variance reduction"""
        states_tensor = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            baselines = self.baseline_net(states_tensor).squeeze().cpu().numpy()
        return baselines

    def update_policy(self):
        """Update policy using REINFORCE with baseline"""
        if len(self.states) == 0:
            return

        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = np.array(self.rewards)
        log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Compute returns and baseline
        returns = self.compute_returns(rewards)
        baselines = self.compute_baseline(self.states)

        # Advantage function
        returns = np.array(returns)
        advantages = returns - baselines

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Policy loss
        policy_loss = -(log_probs * advantages).mean()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Update baseline
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        baseline_loss = F.mse_loss(self.baseline_net(states).squeeze(), returns_tensor)

        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()

        # Clear experience
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def save_model(self, path):
        """Save model parameters"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'baseline_net': self.baseline_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'baseline_optimizer': self.baseline_optimizer.state_dict()
        }, path)

    def load_model(self, path):
        """Load model parameters"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.baseline_net.load_state_dict(checkpoint['baseline_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.baseline_optimizer.load_state_dict(checkpoint['baseline_optimizer'])


class ExperienceDataset:
    """Dataset for batch training"""

    def __init__(self, states, actions, rewards, log_probs):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.log_probs = log_probs

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'log_prob': self.log_probs[idx]
        }


def train_policy_gradient_agent(env, agent, num_episodes=1000, max_steps=1000,
                               update_freq=10, save_path=None):
    """
    Train Policy Gradient agent

    Args:
        env: Trading environment
        agent: PolicyGradientAgent instance
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        update_freq: Update policy every N episodes
        save_path: Path to save model (optional)
    """
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, log_prob)

            state = next_state
            episode_reward += reward
            step += 1

        # Update policy periodically
        if (episode + 1) % update_freq == 0:
            agent.update_policy()

        episode_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

    # Final update
    agent.update_policy()

    if save_path:
        agent.save_model(save_path)

    return episode_rewards


def evaluate_policy_gradient_agent(env, agent, num_episodes=10, max_steps=1000):
    """
    Evaluate trained Policy Gradient agent

    Args:
        env: Trading environment
        agent: Trained PolicyGradientAgent
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode

    Returns:
        List of episode rewards
    """
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            action, _ = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            state = next_state
            episode_reward += reward
            step += 1

        episode_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}: Reward = {episode_reward:.2f}")

    avg_reward = np.mean(episode_rewards)
    print(f"Average Evaluation Reward: {avg_reward:.2f}")

    return episode_rewards