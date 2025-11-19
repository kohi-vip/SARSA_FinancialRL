"""
Deep SARSA Agent for Stock Trading
Based on Yang et al. (2020) implementation
Key difference from DQN: Uses next action selected by policy (on-policy)
instead of max Q-value (off-policy)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    """Deep Q-Network"""

    def __init__(self, state_dim, n_actions, hidden_dims=[128, 128, 64]):
        super(DQN, self).__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, n_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience Replay Buffer for DQN"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DeepSARSAAgent:
    """
    Deep SARSA Agent

    Key difference from DQN: Uses next action selected by policy (on-policy)
    instead of max Q-value (off-policy)

    Hyperparameters as per Yang et al. (2020):
    - lr: 1e-5 (neural network learning rate)
    - alpha: 0.7 (Q-function update learning rate for smooth update)
    - gamma: 0.6 (discount factor)
    - epsilon: 0.8 → 0.2 with decay 0.9
    """

    def __init__(self, state_dim, n_actions, lr=1e-5, gamma=0.6, alpha=0.7,
                 epsilon_start=0.8, epsilon_end=0.2, epsilon_decay=0.9,
                 buffer_capacity=10000, batch_size=64, target_update=10):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha  # Learning rate for Q-function smooth update
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Q-networks
        self.q_network = DQN(state_dim, n_actions).to(self.device)
        self.target_network = DQN(state_dim, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.update_counter = 0
        self.loss_history = []

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, next_action, done, buffer_capacity=10000):
        """Store SARSA transition (includes next_action)"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        # Store next_action separately for SARSA update
        if not hasattr(self, 'next_actions'):
            self.next_actions = deque(maxlen=buffer_capacity)
        self.next_actions.append(next_action)

    def train_step(self):
        """Perform one SARSA training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Get corresponding next_actions
        next_actions_batch = random.sample(list(self.next_actions), self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_actions = torch.LongTensor(next_actions_batch).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values: Q(s, a)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # SARSA Target with smooth update (as per paper):
        # Q_target = (1 - α) * Q_current + α * [r + γ * Q(s', a')]
        with torch.no_grad():
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            td_target = rewards + (1 - dones) * self.gamma * next_q_values
            # Smooth update: blend current Q-value with TD target
            target_q_values = (1 - self.alpha) * current_q_values + self.alpha * td_target

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.loss_history.append(loss.item())
        return loss.item()

    def save(self, filepath):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']