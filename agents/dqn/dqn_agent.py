"""
Deep Q-Network (DQN) Agent for Stock Trading
Based on Yang et al. (2020) implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    """Deep Q-Network matching the reference implementation"""

    def __init__(self, input_size=7, num_classes=11):
        super(DQN, self).__init__()
        self.fc_liner = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.fc_liner(x)


class ReplayBuffer:
    """Experience Replay Buffer for DQN"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Learning Agent"""

    def __init__(self, state_dim, n_actions, lr=0.00001, gamma=0.6,
                 epsilon_start=0.8, epsilon_end=0.2, epsilon_decay=0.9,
                 buffer_capacity=10000, batch_size=64, target_update=10, lr_update=0.7):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.lr_update = lr_update

        # Q-network
        self.q_network = DQN(state_dim, n_actions).to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_func = nn.HuberLoss()
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

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values using same network (no target network)
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Update rule: (1 - lr_update) * current_q + lr_update * target_q
        target_q_full = (1 - self.lr_update) * current_q + self.lr_update * target_q

        # Compute loss
        loss = self.loss_func(current_q, target_q_full)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network (though not used in this implementation)
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            pass  # No target network update

        # Note: Epsilon decay moved to training loop

        self.loss_history.append(loss.item())
        return loss.item()

    def save(self, filepath):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']