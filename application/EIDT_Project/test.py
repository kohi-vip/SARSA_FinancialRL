# UCB_2_pro_fixed.py
# Bản sửa lỗi hoàn chỉnh cho chương trình so sánh UCB-VAE và Epsilon-Greedy

import os
import sys
import random
import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# Cấu hình thiết bị đồng nhất
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STATE_DIM = 7

# ================================================================================
# KHU VỰC 1: MÔ HÌNH CHIẾN LƯỢC UCB-VAE (JOINT VAE TRAINING)
# ================================================================================

class NumpyStandardScaler:
    """Scaler numpy đơn giản, fit trên train states để tránh data leakage."""
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray) -> "NumpyStandardScaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0) + self.eps
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler chưa được fit.")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

def initial_state_from_first_row(series: pd.DataFrame, mdp: Any) -> List[float]:
    """State ban đầu khi training: lấy giá/indicator từ dòng đầu train."""
    first_row = series.iloc[0]
    return [
        float(first_row['close']),
        float(mdp.balance_init),
        0.0,
        float(first_row['MACD']),
        float(first_row['RSI']),
        float(first_row['CCI']),
        float(first_row['ADX']),
    ]

def initial_state_for_test(train_series: pd.DataFrame, mdp: Any) -> List[float]:
    """State đầu test dùng đúng dòng cuối train để tránh lệch thời gian."""
    prev_row = train_series.iloc[-1]
    return [
        float(prev_row['close']),
        float(mdp.balance_init),
        0.0,
        float(prev_row['MACD']),
        float(prev_row['RSI']),
        float(prev_row['CCI']),
        float(prev_row['ADX']),
    ]

def action_to_index(action: int, mdp: Any) -> int:
    """Ánh xạ action thật trong [-k, k] sang index trong [0, 2k]."""
    return int(action) + int(mdp.k)

def actions_to_onehot(actions: Sequence[int], mdp: Any) -> np.ndarray:
    idx = np.array([action_to_index(a, mdp) for a in actions], dtype=np.int64)
    onehot = np.zeros((len(idx), len(mdp.A)), dtype=np.float32)
    onehot[np.arange(len(idx)), idx] = 1.0
    return onehot

def normalize_u_ep_batch(u_ep_values: torch.Tensor) -> torch.Tensor:
    """Chuẩn hóa động u_ep để ép vào [0, 1] dựa trên max của batch."""
    max_u = torch.max(u_ep_values) + 1e-8
    return u_ep_values / max_u

def pi_random_factory(mdp: Any):
    """Tạo random policy cho bootstrap VAE và baseline."""
    def pi_random(state, greedy=False, eps=1.0):
        return int(np.random.choice(mdp.A))
    return pi_random

class EpistemicVAE(nn.Module):
    """VAE học phân phối joint (state, action_onehot) để ước lượng epistemic uncertainty."""
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 16):
        super().__init__()
        input_dim = state_dim + action_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([s, a], dim=-1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def compute_u_ep(self, s: torch.Tensor, a_onehot: torch.Tensor, delta: float = 1.0, lam: float = 1.0) -> torch.Tensor:
        """Tính u_ep(s, a) cho một batch state/action đã scale."""
        x = torch.cat([s, a_onehot], dim=-1)
        mu, logvar = self.encode(x)
        # SỬA LỖI SCALE: Sử dụng torch.mean thay vì torch.sum để đồng bộ với vae_loss
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        std = torch.exp(0.5 * logvar)
        z_95 = mu + 1.96 * std
        x_recon_95 = self.decode(z_95)
        recon_err = torch.norm(x - x_recon_95, p=2, dim=-1)
        return delta * kl_div + lam * recon_err

# SỬA LỖI UNPACKING: Định nghĩa rõ ràng hàm vae_loss trả về tuple 3 giá trị (loss, recon_loss, kld)
def vae_loss(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta_kl: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta_kl * kld
    return loss, recon_loss, kld

class QsaUCB(nn.Module):
    """Critic MLP dùng chung cho UCB-VAE và Epsilon-Greedy để đối chứng công bằng."""
    def __init__(self, input_size: int = STATE_DIM, num_classes: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def collect_vae_samples(train_series: pd.DataFrame, mdp: Any, num_trajectories: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap nhẹ bằng random trajectories trên train, tuyệt đối không dùng test."""
    all_states, all_actions = [], []
    state_init = initial_state_from_first_row(train_series, mdp)
    series = train_series.iloc[1:].reset_index(drop=True)
    pi_random = pi_random_factory(mdp)
    for _ in range(num_trajectories):
        states, rewards, actions = mdp.simulate(series, state_init, pi_random, greedy=False)
        all_states.extend(states[:-1])
        all_actions.extend(actions)
    return np.asarray(all_states, dtype=np.float32), np.asarray(all_actions, dtype=np.int64)

class VAEReplayBuffer:
    """Replay buffer lưu state/action để cập nhật VAE online cùng Deep SARSA."""
    def __init__(self, capacity: int = 50_000):
        self.capacity = int(capacity)
        self.states = []
        self.actions = []

    def __len__(self) -> int:
        return len(self.actions)

    def add(self, states: np.ndarray, actions: np.ndarray):
        self.states.extend(states.tolist())
        self.actions.extend(actions.tolist())
        overflow = len(self.states) - self.capacity
        if overflow > 0:
            self.states = self.states[overflow:]
            self.actions = self.actions[overflow:]

    def add_trajectory(self, states: List[Any], actions: List[int]):
        self.add(np.asarray(states[:-1]), np.asarray(actions))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(self) == 0:
            raise ValueError("VAEReplayBuffer rỗng, chưa thể sample.")
        size = min(int(batch_size), len(self))
        idx = np.random.choice(len(self), size=size, replace=False)
        states = np.asarray([self.states[i] for i in idx], dtype=np.float32)
        actions = np.asarray([self.actions[i] for i in idx], dtype=np.int64)
        return states, actions

def fit_scaler_from_bootstrap(train_series: pd.DataFrame, mdp: Any, num_trajectories: int = 4, scaler: Optional[NumpyStandardScaler] = None) -> Tuple[NumpyStandardScaler, np.ndarray, np.ndarray]:
    if scaler is None:
        scaler = NumpyStandardScaler()
    states_np, actions_np = collect_vae_samples(train_series, mdp, num_trajectories)
    scaler.fit(states_np)
    return scaler, states_np, actions_np

# SỬA LỖI UNPACKING: Hàm update_vae_from_replay được đồng bộ hoàn toàn với vae_loss 3 đầu ra
def update_vae_from_replay(
    vae: EpistemicVAE,
    vae_optimizer: optim.Optimizer,
    replay_buffer: VAEReplayBuffer,
    scaler: NumpyStandardScaler,
    mdp: Any,
    batch_size: int,
    vae_beta_kl: float = 1e-3,
) -> Optional[Dict[str, float]]:
    """Sample replay buffer và update VAE một bước trên đúng DEVICE."""
    if len(replay_buffer) == 0:
        return None
    states_np, actions_np = replay_buffer.sample(batch_size)
    states_scaled = scaler.transform(states_np)
    actions_onehot = actions_to_onehot(actions_np, mdp)
    
    s_batch = torch.as_tensor(states_scaled, dtype=torch.float32, device=DEVICE)
    a_batch = torch.as_tensor(actions_onehot, dtype=torch.float32, device=DEVICE)
    
    vae.train()
    x_recon, mu, logvar = vae(s_batch, a_batch)
    x_true = torch.cat([s_batch, a_batch], dim=-1)
    
    loss, recon_loss, kld = vae_loss(x_recon, x_true, mu, logvar, beta_kl=vae_beta_kl)
    
    vae_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    vae_optimizer.step()
    return {
        "loss": float(loss.detach().cpu().item()),
        "recon_loss": float(recon_loss.detach().cpu().item()),
        "kld": float(kld.detach().cpu().item())
    }

# SỬA LỖI DEVICE: Đưa VAE lên DEVICE ngay khi khởi tạo
def initialize_joint_vae_components(
    train_series: pd.DataFrame,
    mdp: Any,
    latent_dim: int = 16,
    vae_lr: float = 1e-3,
    replay_capacity: int = 50_000,
    bootstrap_trajectories: int = 4,
    bootstrap_vae_updates: int = 100,
    vae_batch_size: int = 128,
    vae_beta_kl: float = 1e-3,
    verbose: bool = False,
) -> Tuple[EpistemicVAE, optim.Optimizer, NumpyStandardScaler, VAEReplayBuffer, List[float]]:
    """Khởi tạo VAE + scaler + replay buffer cho joint training."""
    scaler, states_np, actions_np = fit_scaler_from_bootstrap(
        train_series, mdp, num_trajectories=bootstrap_trajectories
    )
    replay_buffer = VAEReplayBuffer(capacity=replay_capacity)
    replay_buffer.add(states_np, actions_np)
    
    # SỬA LỖI DEVICE: Thêm .to(DEVICE)
    vae = EpistemicVAE(state_dim=STATE_DIM, action_dim=len(mdp.A), latent_dim=latent_dim).to(DEVICE)
    vae_optimizer = optim.Adam(vae.parameters(), lr=vae_lr)
    bootstrap_losses = []
    
    for _ in range(int(bootstrap_vae_updates)):
        stats = update_vae_from_replay(
            vae, vae_optimizer, replay_buffer, scaler, mdp,
            batch_size=vae_batch_size, vae_beta_kl=vae_beta_kl
        )
        if stats is not None:
            bootstrap_losses.append(stats["loss"])
            
    if verbose and bootstrap_losses:
        plt.figure(figsize=(8, 4))
        plt.plot(bootstrap_losses)
        plt.title('EpistemicVAE Bootstrap Warm-up Loss')
        plt.xlabel('Update')
        plt.ylabel('Loss')
        plt.grid(alpha=0.3)
        plt.show()
        
    return vae, vae_optimizer, scaler, replay_buffer, bootstrap_losses

class UCBVAEPolicy:
    """Policy chọn a = argmax_a [Q(s, a) + beta * u_ep(s, a)]"""
    def __init__(self, qsa, mdp, vae, scaler, beta=0.5, delta=1.0, lam=1.0, beta_decay=0.97, beta_min=0.01, episode_max=100):
        self.qsa = qsa
        self.mdp = mdp
        self.vae = vae
        self.scaler = scaler
        self.beta_init = float(beta)
        self.beta = float(beta)
        self.beta_decay = float(beta_decay)
        self.beta_min = float(beta_min)
        self.delta = float(delta)
        self.lam = float(lam)
        self.episode_max = episode_max

    def update_beta_decay(self, episode: int) -> None:
        """Cập nhật beta theo cơ chế giảm dần theo episode."""
        self.beta = max(self.beta_min, self.beta_init * (self.beta_decay ** int(episode)))

    def scaled_state_tensor(self, state: Sequence[float]) -> torch.Tensor:
        s_np = np.asarray(state, dtype=np.float32).reshape(1, -1)
        s_scaled = self.scaler.transform(s_np)
        return torch.as_tensor(s_scaled, dtype=torch.float32, device=DEVICE)

    def __call__(self, state: Sequence[float], greedy: bool = False, eps: float = 0.0) -> int:
        self.qsa.eval()
        self.vae.eval()
        with torch.no_grad():
            s_tensor = self.scaled_state_tensor(state)
            q_values = self.qsa(s_tensor).squeeze(0)
            if greedy:
                best_action_idx = int(torch.argmax(q_values).item())
                return int(self.mdp.A[best_action_idx])
            
            num_actions = len(self.mdp.A)
            s_batch = s_tensor.repeat(num_actions, 1)
            a_onehot_batch = torch.eye(num_actions, dtype=torch.float32, device=DEVICE)
            
            u_ep_tensor = self.vae.compute_u_ep(s_batch, a_onehot_batch, delta=self.delta, lam=self.lam)
            u_ep_normalized = normalize_u_ep_batch(u_ep_tensor)
            
            ucb_tensor = q_values + self.beta * u_ep_normalized
            best_action_idx = int(torch.argmax(ucb_tensor).item())
            return int(self.mdp.A[best_action_idx])

class ScaledStatesDataset(Dataset):
    """Dataset trajectory đã scale state để update SARSA bằng mini-batch."""
    def __init__(self, states, rewards, actions, scaler):
        # SỬA LỖI CHUẨN HÓA: Ép kiểu list thành np.array trước khi transform
        self.states = torch.tensor(scaler.transform(np.asarray(states[:-1])), dtype=torch.float32)
        self.states_next = torch.tensor(scaler.transform(np.asarray(states[1:])), dtype=torch.float32)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)
        self.actions = list(actions)

    def __len__(self) -> int:
        return len(self.rewards)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "states": self.states[idx],
            "states_next": self.states_next[idx],
            "rewards": self.rewards[idx],
            "actions": self.actions[idx]
        }

def _sarsa_update_epoch(
    qsa: QsaUCB,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    mdp: Any,
    gamma: float,
    alpha: float,
    loss_func: nn.Module,
) -> List[float]:
    """Update Q-network theo On-policy SARSA target cho trajectory đã sinh."""
    qsa.train()
    losses = []
    for data_pack in dataloader:
        s_batch = data_pack["states"].to(DEVICE)
        sn_batch = data_pack["states_next"].to(DEVICE)
        r_batch = data_pack["rewards"].to(DEVICE)
        actions_tensor = data_pack["actions"].to(DEVICE)
        
        # Với DataLoader shuffle=False, next action của SARSA chính là action kế tiếp
        next_actions = torch.cat([actions_tensor[1:], actions_tensor[-1:]], dim=0)
        next_action_indices = next_actions + int(mdp.k)
        
        q_values = qsa(s_batch)
        action_indices = actions_tensor + int(mdp.k)
        current_q = q_values.gather(1, action_indices.long().view(-1, 1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = qsa(sn_batch)
            next_q = next_q_values.gather(1, next_action_indices.long().view(-1, 1)).squeeze(1)
            td_target = r_batch + gamma * next_q
            target_tensor = (1.0 - alpha) * current_q.detach() + alpha * td_target
            
        loss = loss_func(current_q, target_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return losses

def train_deep_sarsa_ucb_vae(
    mdp: Any,
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    vae: EpistemicVAE,
    vae_optimizer: optim.Optimizer,
    vae_buffer: VAEReplayBuffer,
    scaler: NumpyStandardScaler,
    episodes: int = 35,
    gamma: float = 0.95,
    alpha: float = 0.6,
    nn_epochs: int = 1,
    nn_lr: float = 2.5e-5,
    beta: float = 0.2,
    delta: float = 1.0,
    lam: float = 1.0,
    batch_size: int = 128,
    vae_batch_size: int = 128,
    vae_updates_per_q_batch: int = 1,
    vae_beta_kl: float = 1e-3,
    beta_decay: float = 0.97,
    beta_min: float = 0.01,
    verbose: bool = False,
) -> Tuple[UCBVAEPolicy, QsaUCB, List[float], List[float]]:
    """Huấn luyện Deep SARSA UCB-VAE với joint VAE training online."""
    qsa = QsaUCB(input_size=STATE_DIM, num_classes=len(mdp.A)).to(DEVICE)
    optimizer = optim.Adam(qsa.parameters(), lr=nn_lr)
    loss_func = nn.HuberLoss()
    policy = UCBVAEPolicy(qsa, mdp, vae, scaler, beta=beta, delta=delta, lam=lam, beta_decay=beta_decay, beta_min=beta_min, episode_max=episodes)
    
    state_init = initial_state_from_first_row(train_series, mdp)
    series = train_series.iloc[1:].reset_index(drop=True)
    learning_curve, vae_losses = [], []
    
    iterator = tqdm(range(int(episodes)), desc="Training Deep SARSA-UCB-VAE", disable=not verbose)
    for epi in iterator:
        policy.update_beta_decay(epi)
        
        states, rewards, actions = mdp.simulate(series, state_init, policy, greedy=False)
        
        vae_buffer.add_trajectory(states, actions)
        
        dataset = ScaledStatesDataset(states, rewards, actions, scaler)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Train Q-network
        q_losses = _sarsa_update_epoch(qsa, optimizer, dataloader, mdp, gamma, alpha, loss_func)
        
        # Train VAE song song
        for _ in range(vae_updates_per_q_batch):
            stats = update_vae_from_replay(vae, vae_optimizer, vae_buffer, scaler, mdp, vae_batch_size, vae_beta_kl)
            if stats is not None:
                vae_losses.append(stats["loss"])
                
        profit = mdp.interact_test(policy, train_series=train_series, test_series=test_series, verbose=False)
        learning_curve.append(float(profit))
        
    return policy, qsa, learning_curve, vae_losses

# ================================================================================
# KHU VỰC 2: MÔ HÌNH CHIẾN LƯỢC EPSILON-GREEDY DEEP SARSA
# ================================================================================

class EpsilonGreedyPolicy:
    """Policy epsilon-greedy truyền thống dùng cùng critic QsaUCB."""
    def __init__(self, qsa: QsaUCB, mdp: Any, scaler: NumpyStandardScaler, epsilon_init=1.0, epsilon_decay=0.95, epsilon_min=0.05):
        self.qsa = qsa
        self.mdp = mdp
        self.scaler = scaler
        self.epsilon_init = float(epsilon_init)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.epsilon = float(epsilon_init)

    def update_epsilon(self, episode: int) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon_init * (self.epsilon_decay ** int(episode)))

    def scaled_state_tensor(self, state: Sequence[float]) -> torch.Tensor:
        s_np = np.asarray(state, dtype=np.float32).reshape(1, -1)
        s_scaled = self.scaler.transform(s_np)
        return torch.tensor(s_scaled, dtype=torch.float32, device=DEVICE)

    def __call__(self, state: Sequence[float], greedy: bool = False, eps: float = 0.0) -> int:
        self.qsa.eval()
        if (not greedy) and (np.random.rand() < self.epsilon):
            return int(np.random.choice(self.mdp.A))
        with torch.no_grad():
            q_values = self.qsa(self.scaled_state_tensor(state)).squeeze(0)
            return int(self.mdp.A[int(torch.argmax(q_values).item())])

def fit_scaler_from_random_train(train_series: pd.DataFrame, mdp: Any, num_trajectories: int = 5) -> NumpyStandardScaler:
    """Fit scaler cho epsilon-greedy bằng train trajectories/bootstrap."""
    scaler = NumpyStandardScaler()
    _ = fit_scaler_from_bootstrap(train_series, mdp, num_trajectories=num_trajectories, scaler=scaler)
    return scaler

def train_deep_sarsa_epsilon_greedy(
    mdp: Any,
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    scaler: Optional[NumpyStandardScaler] = None,
    episodes: int = 35,
    gamma: float = 0.95,
    alpha: float = 0.6,
    nn_epochs: int = 1,
    nn_lr: float = 2.5e-5,
    batch_size: int = 128,
    epsilon_init: float = 1.0,
    epsilon_decay: float = 0.95,
    epsilon_min: float = 0.05,
    scaler_bootstrap_trajectories: int = 5,
    verbose: bool = False,
) -> Tuple[EpsilonGreedyPolicy, QsaUCB, List[float], List[float]]:
    """Huấn luyện Deep SARSA epsilon-greedy đối chứng."""
    if scaler is None:
        scaler = fit_scaler_from_random_train(train_series, mdp, num_trajectories=scaler_bootstrap_trajectories)
        
    qsa = QsaUCB(input_size=STATE_DIM, num_classes=len(mdp.A)).to(DEVICE)
    optimizer = optim.Adam(qsa.parameters(), lr=nn_lr)
    loss_func = nn.HuberLoss()
    policy = EpsilonGreedyPolicy(qsa, mdp, scaler, epsilon_init, epsilon_decay, epsilon_min)
    
    state_init = initial_state_from_first_row(train_series, mdp)
    series = train_series.iloc[1:].reset_index(drop=True)
    learning_curve, q_losses = [], []
    
    iterator = tqdm(range(int(episodes)), desc="Training Deep SARSA-EpsilonGreedy", disable=not verbose)
    for epi in iterator:
        policy.update_epsilon(epi)
        
        states, rewards, actions = mdp.simulate(series, state_init, policy, greedy=False)
        
        dataset = ScaledStatesDataset(states, rewards, actions, scaler)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        losses = _sarsa_update_epoch(qsa, optimizer, dataloader, mdp, gamma, alpha, loss_func)
        q_losses.extend(losses)
        
        profit = mdp.interact_test(policy, train_series=train_series, test_series=test_series, verbose=False)
        learning_curve.append(float(profit))
        
    return policy, qsa, learning_curve, q_losses

# ================================================================================
# KHU VỰC 3: TRỰC QUAN HÓA SO SÁNH (VISUALIZATION) & LƯU MÔ HÌNH
# ================================================================================

def collect_portfolio_history(mdp: Any, pi: Any, train_series: pd.DataFrame, test_series: pd.DataFrame) -> np.ndarray:
    prev_row = train_series.iloc[-1]
    state_init = [
        float(prev_row['close']),
        float(mdp.balance_init),
        0.0,
        float(prev_row['MACD']),
        float(prev_row['RSI']),
        float(prev_row['CCI']),
        float(prev_row['ADX']),
    ]
    series = test_series.reset_index(drop=True)
    states, rewards, actions = mdp.simulate(series, state_init, pi, greedy=True)
    portfolio = np.array([s[1] + s[0] * s[2] for s in states], dtype=np.float32)
    return portfolio

def calculate_volatility(portfolio_history: np.ndarray) -> float:
    if len(portfolio_history) < 2: return 0.0
    returns = np.diff(portfolio_history) / np.maximum(portfolio_history[:-1], 1e-8)
    return float(np.std(returns) * np.sqrt(252) * 100)

def calculate_sharpe_ratio(portfolio_history: np.ndarray, risk_free_rate: float = 2.0) -> float:
    if len(portfolio_history) < 2: return 0.0
    returns = np.diff(portfolio_history) / np.maximum(portfolio_history[:-1], 1e-8)
    annual_return = np.mean(returns) * 252 * 100
    volatility = np.std(returns) * np.sqrt(252) * 100
    if volatility <= 1e-8: return 0.0
    return float((annual_return - risk_free_rate) / volatility)

def calculate_max_drawdown(portfolio_history: np.ndarray) -> float:
    if len(portfolio_history) == 0: return 0.0
    peak = np.maximum.accumulate(portfolio_history)
    drawdown = (portfolio_history - peak) / np.maximum(peak, 1e-8)
    return float(abs(drawdown.min()) * 100)

def agent_annual_return(initial_capital: float, final_portfolio: float, start_date: str, end_date: str) -> Tuple[float, float]:
    total_return = (final_portfolio / initial_capital) - 1.0
    years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    if years <= 0: return 0.0, 0.0
    annual = (1.0 + total_return) ** (1.0 / years) - 1.0
    return annual * 100, total_return * 100

def _run_single_ucb(train_series: pd.DataFrame, test_series: pd.DataFrame, mdp: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    vae, vae_optimizer, scaler, vae_buffer, bootstrap_losses = initialize_joint_vae_components(
        train_series, mdp,
        latent_dim=config.get("vae_latent_dim", 16),
        vae_lr=config.get("vae_lr", 1e-3),
        vae_beta_kl=config.get("vae_beta_kl", 1e-3),
        vae_batch_size=config.get("vae_batch_size", 128),
        bootstrap_trajectories=config.get("bootstrap_trajectories", 4),
        bootstrap_vae_updates=config.get("bootstrap_vae_updates", 100),
        verbose=False,
    )
    policy, qsa, learning_curve, vae_losses = train_deep_sarsa_ucb_vae(
        mdp=mdp,
        train_series=train_series,
        test_series=test_series,
        vae=vae,
        vae_optimizer=vae_optimizer,
        vae_buffer=vae_buffer,
        scaler=scaler,
        episodes=config["episodes"],
        gamma=config["gamma"],
        alpha=config["alpha"],
        nn_epochs=config["nn_epochs"],
        nn_lr=config["nn_lr"],
        beta=config["beta"],
        delta=config["delta"],
        lam=config["lam"],
        batch_size=config.get("batch_size", 128),
        vae_batch_size=config.get("vae_batch_size", 128),
        vae_updates_per_q_batch=config.get("vae_updates_per_q_batch", 1),
        vae_beta_kl=config.get("vae_beta_kl", 1e-3),
        beta_decay=config["beta_decay"],
        beta_min=config["beta_min"],
        verbose=False,
    )
    final_profit = mdp.interact_test(policy, train_series=train_series, test_series=test_series, verbose=False)
    portfolio_history = collect_portfolio_history(mdp, policy, train_series, test_series)
    return {
        "policy": policy,
        "qsa": qsa,
        "vae": vae,
        "scaler": scaler,
        "final_profit": final_profit,
        "learning_curve": learning_curve,
        "portfolio_history": portfolio_history,
        "vae_losses": vae_losses,
        "bootstrap_vae_losses": bootstrap_losses
    }

def _run_single_eps(train_series: pd.DataFrame, test_series: pd.DataFrame, mdp: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    policy, qsa, learning_curve, q_losses = train_deep_sarsa_epsilon_greedy(
        mdp=mdp,
        train_series=train_series,
        test_series=test_series,
        episodes=config["episodes"],
        gamma=config["gamma"],
        alpha=config["alpha"],
        nn_epochs=config["nn_epochs"],
        nn_lr=config["nn_lr"],
        batch_size=config.get("batch_size", 128),
        epsilon_init=config.get("epsilon_init", 1.0),
        epsilon_decay=config.get("epsilon_decay", 0.95),
        epsilon_min=config.get("epsilon_min", 0.05),
        verbose=False,
    )
    final_profit = mdp.interact_test(policy, train_series=train_series, test_series=test_series, verbose=False)
    portfolio_history = collect_portfolio_history(mdp, policy, train_series, test_series)
    return {
        "policy": policy,
        "qsa": qsa,
        "scaler": policy.scaler,
        "final_profit": final_profit,
        "learning_curve": learning_curve,
        "portfolio_history": portfolio_history,
        "q_losses": q_losses
    }

def _aggregate_runs(agent_name: str, runs: List[Dict[str, Any]], test_series: pd.DataFrame, mdp: Any) -> Dict[str, Any]:
    profits = np.array([r["final_profit"] for r in runs])
    curves = np.array([r["learning_curve"] for r in runs])
    portfolios = np.array([r["portfolio_history"] for r in runs])
    best_run = max(runs, key=lambda r: r["final_profit"])
    
    start_date = test_series.iloc[0]["time"]
    end_date = test_series.iloc[-1]["time"]
    final_val = mdp.balance_init + np.mean(profits)
    arr, roi = agent_annual_return(mdp.balance_init, final_val, start_date, end_date)
    
    mean_portfolio = np.mean(portfolios, axis=0)
    vol = calculate_volatility(mean_portfolio)
    sr = calculate_sharpe_ratio(mean_portfolio)
    mdd = calculate_max_drawdown(mean_portfolio)
    
    return {
        "agent": agent_name,
        "final_profit": float(np.mean(profits)),
        "std_final_profit": float(np.std(profits)),
        "all_final_profits": profits.tolist(),
        "learning_curve": np.mean(curves, axis=0).tolist(),
        "std_learning_curve": np.std(curves, axis=0).tolist(),
        "portfolio_history": mean_portfolio.tolist(),
        "trained_agent": best_run["qsa"],
        "best_policy": best_run["policy"],
        "trained_vae": best_run.get("vae", None),
        "vae_scaler": best_run.get("scaler", None),
        "arr": arr,
        "roi": roi,
        "volatility": vol,
        "sharpe_ratio": sr,
        "max_drawdown": mdd
    }

def run_comparative_experiment(
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    config_ucb: Dict[str, Any],
    config_eps: Dict[str, Any],
    num_runs: int = 20,
    mdp: Optional[Any] = None,
    seed: int = 42,
    title_suffix: str = "HPG",
    save_models: bool = True,
    plot: bool = True,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    ucb_runs = []
    eps_runs = []
    
    print(f"=== BẮT ĐẦU CHẠY THỰC NGHIỆM ĐỐI CHỨNG GIỮA UCB-VAE VÀ EPSILON-GREEDY ({title_suffix}) ===")
    for run in range(num_runs):
        print(f"--- Lượt chạy [{run+1}/{num_runs}] ---")
        run_seed = seed + run
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        
        ucb_res = _run_single_ucb(train_series, test_series, mdp, config_ucb)
        eps_res = _run_single_eps(train_series, test_series, mdp, config_eps)
        
        ucb_runs.append(ucb_res)
        eps_runs.append(eps_res)
        
    result_ucb = _aggregate_runs("Deep SARSA-UCB-VAE", ucb_runs, test_series, mdp)
    result_eps = _aggregate_runs("Deep SARSA-Epsilon-Greedy", eps_runs, test_series, mdp)
    
    summary_df = pd.DataFrame([
        {
            "Strategy": result_ucb["agent"],
            "Average Final Profit": f"{result_ucb['final_profit']:.2f} ± {result_ucb['std_final_profit']:.2f}",
            "ARR (%)": f"{result_ucb['arr']:.2f}%",
            "ROI (%)": f"{result_ucb['roi']:.2f}%",
            "Volatility (%)": f"{result_ucb['volatility']:.2f}%",
            "Sharpe Ratio": f"{result_ucb['sharpe_ratio']:.4f}",
            "Max Drawdown (%)": f"{result_ucb['max_drawdown']:.2f}%"
        },
        {
            "Strategy": result_eps["agent"],
            "Average Final Profit": f"{result_eps['final_profit']:.2f} ± {result_eps['std_final_profit']:.2f}",
            "ARR (%)": f"{result_eps['arr']:.2f}%",
            "ROI (%)": f"{result_eps['roi']:.2f}%",
            "Volatility (%)": f"{result_eps['volatility']:.2f}%",
            "Sharpe Ratio": f"{result_eps['sharpe_ratio']:.4f}",
            "Max Drawdown (%)": f"{result_eps['max_drawdown']:.2f}%"
        }
    ])
    print("\nBẢNG KẾT QUẢ ĐỐI CHỨNG:")
    print(summary_df.to_markdown(index=False))
    
    if save_models:
        models_dir = Path("./models")
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result_ucb["trained_agent"].state_dict(), models_dir / "sarsa_ucb_vae_best.pth")
        torch.save(result_eps["trained_agent"].state_dict(), models_dir / "sarsa_epsilon_greedy_best.pth")
        print(f"\n[Saved] Đã lưu model state_dict xuất sắc nhất vào {models_dir}/")
        
    return {"ucb": result_ucb, "eps": result_eps}
