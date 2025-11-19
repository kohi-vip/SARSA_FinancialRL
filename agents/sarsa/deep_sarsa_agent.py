"""
Deep SARSA Agent for Stock Trading with FinRL Environment
Integrates the Deep SARSA algorithm with FinRL's StockTradingEnv
"""
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm


class QNetwork(nn.Module):
    """Deep Q-Network for approximating Q(s,a) values"""
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[128, 64]):
        """
        Args:
            state_dim: dimension of state space
            action_dim: dimension of action space (number of discrete actions)
            hidden_sizes: list of hidden layer sizes
        """
        super().__init__()
        
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: state tensor of shape [batch_size, state_dim]
        Returns:
            Q-values tensor of shape [batch_size, action_dim]
        """
        return self.network(x)


class ExperienceDataset(Dataset):
    """Dataset for storing trajectory experiences"""
    
    def __init__(self, states, next_states, rewards, actions):
        """
        Args:
            states: array of states [T, state_dim]
            next_states: array of next states [T, state_dim]
            rewards: array of rewards [T]
            actions: array of action indices [T]
        """
        self.states = torch.FloatTensor(states)
        self.next_states = torch.FloatTensor(next_states)
        self.rewards = torch.FloatTensor(rewards)
        self.actions = torch.LongTensor(actions)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'next_state': self.next_states[idx],
            'reward': self.rewards[idx],
            'action': self.actions[idx]
        }


class DeepSARSAAgent:
    """
    Deep SARSA Agent for Stock Trading
    
    Attributes:
        env: FinRL StockTradingEnv
        q_network: neural network for Q(s,a) approximation
        optimizer: optimizer for training
        device: 'cpu' or 'cuda'
        gamma: discount factor
        action_dim: number of discrete actions
    """
    
    def __init__(self, 
                 env,
                 state_dim=None,
                 action_dim=10,
                 hidden_sizes=[128, 64],
                 learning_rate=1e-4,
                 gamma=0.99,
                 device='cpu'):
        """
        Initialize Deep SARSA Agent
        
        Args:
            env: StockTradingEnv from FinRL
            state_dim: dimension of state (auto-detect if None)
            action_dim: number of discrete actions to discretize continuous action space
            hidden_sizes: hidden layer sizes for Q-network
            learning_rate: learning rate for optimizer
            gamma: discount factor
            device: 'cpu' or 'cuda'
        """
        self.env = env
        self.device = torch.device(device)
        self.gamma = gamma
        self.action_dim = action_dim
        
        # Auto-detect state dimension from environment
        if state_dim is None:
            test_state, _ = env.reset()
            self.state_dim = len(test_state)
        else:
            self.state_dim = state_dim
        
        # Get continuous action space dimension from environment
        self.continuous_action_dim = env.action_space.shape[0]
        
        # Create Q-network
        self.q_network = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_sizes=hidden_sizes
        ).to(self.device)
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.HuberLoss()
        
        # Training history
        self.training_history = {
            'losses': [],
            'episode_rewards': [],
            'episode_lengths': []
        }
        
        print(f"DeepSARSAAgent initialized:")
        print(f"  State dimension: {self.state_dim}")
        print(f"  Discrete action dimension: {self.action_dim}")
        print(f"  Continuous action dimension: {self.continuous_action_dim}")
        print(f"  Device: {self.device}")
        print(f"  Network architecture: {self.state_dim} -> {hidden_sizes} -> {self.action_dim}")
    
    def discrete_to_continuous_action(self, discrete_action):
        """
        Convert discrete action index to continuous action vector for environment
        
        Args:
            discrete_action: integer in [0, action_dim-1]
        
        Returns:
            continuous_action: array of shape [continuous_action_dim]
        """
        # Simple mapping: discretize each stock action into bins
        # Map action_idx to a value in range for each stock
        
        # Strategy: divide discrete actions across stocks
        # For simplicity, map to {-1, 0, 1} * scale for each stock
        action_per_stock = (discrete_action % 3) - 1  # -1, 0, or 1
        action_vector = np.ones(self.continuous_action_dim) * action_per_stock * 0.1
        
        return action_vector
    
    def select_action(self, state, epsilon=0.0, greedy=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: current state
            epsilon: exploration probability
            greedy: if True, use greedy policy (ignore epsilon)
        
        Returns:
            discrete_action: integer action index
        """
        # Epsilon-greedy exploration
        if not greedy and np.random.rand() < epsilon:
            return np.random.randint(0, self.action_dim)
        
        # Greedy action selection
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def collect_trajectory(self, epsilon=0.2, max_steps=None):
        """
        Collect a trajectory by interacting with environment
        
        Args:
            epsilon: exploration probability
            max_steps: maximum steps per episode (None for no limit)
        
        Returns:
            states: list of states
            next_states: list of next states
            rewards: list of rewards
            actions: list of discrete actions
            total_reward: cumulative reward
        """
        states, next_states, rewards, actions = [], [], [], []
        
        state, _ = self.env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        while not done:
            # Select discrete action
            discrete_action = self.select_action(state, epsilon=epsilon, greedy=False)
            
            # Convert to continuous action for environment
            continuous_action = self.discrete_to_continuous_action(discrete_action)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = self.env.step(continuous_action)
            done = terminated or truncated
            
            # Store transition
            states.append(state)
            next_states.append(next_state)
            rewards.append(reward)
            actions.append(discrete_action)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if max_steps and steps >= max_steps:
                break
        
        return (np.array(states), np.array(next_states), 
                np.array(rewards), np.array(actions), total_reward)
    
    def train_on_trajectory(self, trajectory, lr_update=0.7, batch_size=64, epochs=5):
        """
        Train Q-network on a collected trajectory using SARSA update
        
        Args:
            trajectory: tuple of (states, next_states, rewards, actions, total_reward)
            lr_update: learning rate for TD update (mixing old and new Q-values)
            batch_size: batch size for training
            epochs: number of epochs to train on this trajectory
        
        Returns:
            avg_loss: average loss over all batches
        """
        states, next_states, rewards, actions, _ = trajectory
        
        # Create dataset and dataloader
        dataset = ExperienceDataset(states, next_states, rewards, actions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        epoch_losses = []
        
        for epoch in range(epochs):
            for batch in dataloader:
                # Move batch to device
                state_batch = batch['state'].to(self.device)
                next_state_batch = batch['next_state'].to(self.device)
                reward_batch = batch['reward'].to(self.device)
                action_batch = batch['action'].to(self.device)
                
                # Get current Q-values
                q_values = self.q_network(state_batch)
                current_q = q_values.gather(1, action_batch.unsqueeze(1)).squeeze()
                
                # SARSA: get next Q-value using policy's action
                with torch.no_grad():
                    next_q_values = self.q_network(next_state_batch)
                    # Use greedy action for next state (or could use epsilon-greedy)
                    next_actions = next_q_values.argmax(dim=1)
                    next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
                    
                    # TD target
                    td_target = reward_batch + self.gamma * next_q
                    
                    # Smoothed target (as in original DSARSA)
                    target = (1 - lr_update) * current_q + lr_update * td_target
                
                # Compute loss
                loss = self.loss_fn(current_q, target)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        return avg_loss
    
    def train(self, 
              num_episodes=100,
              epsilon_start=0.9,
              epsilon_end=0.1,
              epsilon_decay=0.95,
              lr_update=0.7,
              batch_size=64,
              epochs_per_trajectory=5,
              max_steps_per_episode=None,
              verbose=True):
        """
        Train the agent over multiple episodes
        
        Args:
            num_episodes: number of training episodes
            epsilon_start: initial exploration rate
            epsilon_end: minimum exploration rate
            epsilon_decay: decay factor for epsilon per episode
            lr_update: learning rate for TD update
            batch_size: batch size for training
            epochs_per_trajectory: epochs to train on each trajectory
            max_steps_per_episode: max steps per episode
            verbose: whether to print progress
        
        Returns:
            training_history: dict with losses, rewards, lengths
        """
        epsilon = epsilon_start
        
        pbar = tqdm(range(num_episodes), desc="Training Deep SARSA")
        
        for episode in pbar:
            # Collect trajectory
            trajectory = self.collect_trajectory(
                epsilon=max(epsilon, epsilon_end),
                max_steps=max_steps_per_episode
            )
            states, next_states, rewards, actions, total_reward = trajectory
            
            # Train on trajectory
            avg_loss = self.train_on_trajectory(
                trajectory,
                lr_update=lr_update,
                batch_size=batch_size,
                epochs=epochs_per_trajectory
            )
            
            # Update epsilon
            epsilon *= epsilon_decay
            
            # Store history
            self.training_history['losses'].append(avg_loss)
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['episode_lengths'].append(len(states))
            
            # Update progress bar
            pbar.set_postfix({
                'reward': f'{total_reward:.2f}',
                'loss': f'{avg_loss:.4f}',
                'eps': f'{epsilon:.3f}',
                'steps': len(states)
            })
        
        if verbose:
            self.plot_training_history()
        
        return self.training_history
    
    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluate the trained agent
        
        Args:
            num_episodes: number of episodes to evaluate
            render: whether to render (not used for now)
        
        Returns:
            eval_rewards: list of episode rewards
            eval_stats: dict with mean, std, min, max
        """
        eval_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 10000:
                # Use greedy policy
                discrete_action = self.select_action(state, greedy=True)
                continuous_action = self.discrete_to_continuous_action(discrete_action)
                
                next_state, reward, terminated, truncated, _ = self.env.step(continuous_action)
                done = terminated or truncated
                
                total_reward += reward
                state = next_state
                steps += 1
            
            eval_rewards.append(total_reward)
        
        eval_stats = {
            'mean': np.mean(eval_rewards),
            'std': np.std(eval_rewards),
            'min': np.min(eval_rewards),
            'max': np.max(eval_rewards)
        }
        
        print(f"\nEvaluation Results ({num_episodes} episodes):")
        print(f"  Mean Reward: {eval_stats['mean']:.2f} ± {eval_stats['std']:.2f}")
        print(f"  Min Reward: {eval_stats['min']:.2f}")
        print(f"  Max Reward: {eval_stats['max']:.2f}")
        
        return eval_rewards, eval_stats
    
    
    def save_model(self, filepath='deep_sarsa_agent.pth'):
        """Save agent's Q-network and training state"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'continuous_action_dim': self.continuous_action_dim,
            'gamma': self.gamma
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='deep_sarsa_agent.pth'):
        """Load agent's Q-network and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        print(f"Model loaded from {filepath}")
        print(f"  Previous training episodes: {len(self.training_history['episode_rewards'])}")
