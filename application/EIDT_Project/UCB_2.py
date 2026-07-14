"""
Deep SARSA trong tài chính: so sánh UCB-VAE và Epsilon-Greedy trên HPG GOOD/BAD.

File này là nguồn Python đồng bộ với notebook `UCB_2.ipynb` để dễ kiểm tra syntax,
tránh phải nhồi code dài vào terminal/notebook. Notebook được chia thành 3 khu vực lớn:

1. MÔ HÌNH CHIẾN LƯỢC UCB-VAE (JOINT VAE TRAINING)
2. MÔ HÌNH CHIẾN LƯỢC EPSILON-GREEDY DEEP SARSA
3. TRỰC QUAN HÓA SO SÁNH & LƯU MÔ HÌNH

Các thiết kế chính:
- Cả hai chiến lược dùng chung critic `QsaUCB(input_size=7, num_classes=11)`.
- Scaler chỉ fit trên train trajectory/bootstrap, tuyệt đối không dùng test để tránh leakage.
- Tất cả tensor được đưa về `DEVICE` nhất quán để chạy tốt trên CPU/GPU Kaggle.
"""

# %% [markdown]
# # Thiết lập môi trường, import và xử lý dữ liệu HPG

import os
import sys
import random
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


def find_project_root() -> Path:
    """Tìm project root robust cho VS Code, Kaggle và Jupyter."""
    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd / "SARSA_FinancialRL", *cwd.parents]
    for p in candidates:
        if (p / "data").exists() and (p / "environments").exists() and (p / "agents").exists():
            return p
    raise FileNotFoundError("Không tìm thấy project root chứa data/, environments/, agents/.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from data.data_processor.feature_engineer import engineer_stat as es
    from environments.stock_trading_env.mdp import StockTradingMDP as stockMDP
except Exception as exc:  # pragma: no cover - giúp py_compile/test.py không phụ thuộc runtime env đầy đủ
    es = None
    stockMDP = None
    _IMPORT_ERROR = exc


SEED = 42
STATE_DIM = 7


def set_global_seed(seed: int = SEED) -> None:
    """Cố định seed để các run độc lập nhưng có thể tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_global_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_hpg_good_bad_data(project_root: Path = PROJECT_ROOT) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nạp HPG, thêm indicator và split GOOD/BAD đồng nhất cho hai chiến lược."""
    if es is None:
        raise ImportError(f"Không import được engineer_stat/StockTradingMDP: {_IMPORT_ERROR}")

    csv_path = project_root / "data" / "data_storer" / "data_research" / "HPG_data.csv"
    df = pd.read_csv(csv_path)
    start_date = pd.to_datetime("01-01-2013", format="%d-%m-%Y")

    price_history = es.add_technical_indicators(
        df,
        start_date=start_date,
        auto_adjust_start_date=True,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        rsi_period=14,
        cci_period=20,
        adx_period=14,
    )

    price_history["time"] = pd.to_datetime(price_history["time"], format="%d/%m/%Y", errors="coerce")
    price_history = price_history.dropna(subset=["time", "close", "MACD", "RSI", "CCI", "ADX"]).reset_index(drop=True)

    good_train_hpg = price_history[(price_history["time"] >= "2013-01-01") & (price_history["time"] <= "2018-12-31")].reset_index(drop=True)
    good_test_hpg = price_history[(price_history["time"] >= "2019-01-01") & (price_history["time"] <= "2021-12-31")].reset_index(drop=True)
    bad_train_hpg = price_history[(price_history["time"] >= "2013-01-01") & (price_history["time"] <= "2021-12-31")].reset_index(drop=True)
    bad_test_hpg = price_history[(price_history["time"] >= "2022-01-01") & (price_history["time"] <= "2023-12-31")].reset_index(drop=True)
    return price_history, good_train_hpg, good_test_hpg, bad_train_hpg, bad_test_hpg


# %% [markdown]
# # KHU VỰC 1: MÔ HÌNH CHIẾN LƯỢC UCB-VAE (JOINT VAE TRAINING)


class NumpyStandardScaler:
    """Scaler numpy đơn giản, fit trên train states để tránh data leakage."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "NumpyStandardScaler":
        x = np.asarray(x, dtype=np.float32)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0) + self.eps
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler chưa được fit.")
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def initial_state_from_first_row(series: pd.DataFrame, mdp: Any) -> List[float]:
    """State ban đầu khi training: lấy giá/indicator từ dòng đầu train."""
    first_row = series.iloc[0]
    return [
        float(first_row["close"]),
        float(mdp.balance_init),
        0.0,
        float(first_row["MACD"]),
        float(first_row["RSI"]),
        float(first_row["CCI"]),
        float(first_row["ADX"]),
    ]


def initial_state_for_test(train_series: pd.DataFrame, mdp: Any) -> List[float]:
    """State đầu test dùng đúng dòng cuối train như `interact_test`, tránh lệch thời gian."""
    prev_row = train_series.iloc[-1]
    return [
        float(prev_row["close"]),
        float(mdp.balance_init),
        0.0,
        float(prev_row["MACD"]),
        float(prev_row["RSI"]),
        float(prev_row["CCI"]),
        float(prev_row["ADX"]),
    ]


def action_to_index(action: int, mdp: Any) -> int:
    """Ánh xạ action thật trong [-k, k] sang index [0, 2k]."""
    return int(action) + int(mdp.k)


def actions_to_onehot(actions: Sequence[int], mdp: Any) -> np.ndarray:
    idx = np.asarray([action_to_index(a, mdp) for a in actions], dtype=np.int64)
    onehot = np.zeros((len(idx), len(mdp.A)), dtype=np.float32)
    onehot[np.arange(len(idx)), idx] = 1.0
    return onehot


def normalize_u_ep_batch(u_ep_values: torch.Tensor) -> torch.Tensor:
    """Chuẩn hóa u_ep theo max batch, giữ nguyên device CPU/GPU của tensor đầu vào."""
    return u_ep_values / (torch.max(u_ep_values) + 1e-8)


def pi_random_factory(mdp: Any):
    """Tạo random policy cho bootstrap VAE và baseline."""

    def pi_random(state, greedy: bool = False, eps: float = 1.0):
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
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
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

    def forward(self, s: torch.Tensor, a_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([s, a_onehot], dim=-1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def compute_u_ep(self, s: torch.Tensor, a_onehot: torch.Tensor, delta: float = 1.0, lam: float = 1.0) -> torch.Tensor:
        """u_ep(s,a) = delta * KL + lam * reconstruction_error_at_95_percentile."""
        x = torch.cat([s, a_onehot], dim=-1)
        mu, logvar = self.encode(x)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        std = torch.exp(0.5 * logvar)
        z_95 = mu + 1.96 * std
        x_recon_95 = self.decode(z_95)
        recon_err = torch.norm(x - x_recon_95, p=2, dim=-1)
        return delta * kl_div + lam * recon_err


def vae_loss(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta_kl: float = 1e-3):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta_kl * kld, recon_loss.detach(), kld.detach()


class VAEReplayBuffer:
    """Replay buffer lưu state/action để VAE được joint training online cùng SARSA."""

    def __init__(self, capacity: int = 50_000):
        self.capacity = int(capacity)
        self.states: List[List[float]] = []
        self.actions: List[int] = []

    def __len__(self) -> int:
        return len(self.actions)

    def add(self, states: Sequence[Sequence[float]], actions: Sequence[int]) -> None:
        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.int64)
        if len(states_np) != len(actions_np):
            raise ValueError(f"len(states)={len(states_np)} phải bằng len(actions)={len(actions_np)}")
        self.states.extend(states_np.tolist())
        self.actions.extend(actions_np.tolist())
        overflow = len(self.actions) - self.capacity
        if overflow > 0:
            self.states = self.states[overflow:]
            self.actions = self.actions[overflow:]

    def add_trajectory(self, states: Sequence[Sequence[float]], actions: Sequence[int]) -> None:
        # mdp.simulate trả len(states) = len(actions) + 1
        self.add(states[:-1], actions)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(self) == 0:
            raise ValueError("VAEReplayBuffer rỗng, chưa thể sample.")
        size = min(int(batch_size), len(self))
        idx = np.random.choice(len(self), size=size, replace=False)
        states = np.asarray([self.states[i] for i in idx], dtype=np.float32)
        actions = np.asarray([self.actions[i] for i in idx], dtype=np.int64)
        return states, actions


class QsaUCB(nn.Module):
    """Critic MLP dùng chung cho UCB-VAE và Epsilon-Greedy để đối chứng công bằng."""

    def __init__(self, input_size: int = STATE_DIM, num_classes: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collect_vae_samples(train_series: pd.DataFrame, mdp: Any, num_trajectories: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap nhẹ bằng random trajectories trên train, tuyệt đối không dùng test."""
    all_states, all_actions = [], []
    state_init = initial_state_from_first_row(train_series, mdp)
    series = train_series.iloc[1:].reset_index(drop=True)
    pi_random = pi_random_factory(mdp)
    for _ in range(int(num_trajectories)):
        states, rewards, actions = mdp.simulate(series, state_init, pi_random, greedy=False, eps=1.0)
        all_states.extend(states[:-1])
        all_actions.extend(actions)
    return np.asarray(all_states, dtype=np.float32), np.asarray(all_actions, dtype=np.int64)


def fit_scaler_from_bootstrap(train_series: pd.DataFrame, mdp: Any, num_trajectories: int = 4) -> Tuple[NumpyStandardScaler, np.ndarray, np.ndarray]:
    states_np, actions_np = collect_vae_samples(train_series, mdp, num_trajectories=num_trajectories)
    scaler = NumpyStandardScaler().fit(states_np)
    return scaler, states_np, actions_np


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
    return {"loss": float(loss.detach().cpu()), "recon_loss": float(recon_loss.cpu()), "kld": float(kld.cpu())}


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
    scaler, states_np, actions_np = fit_scaler_from_bootstrap(train_series, mdp, num_trajectories=bootstrap_trajectories)
    replay_buffer = VAEReplayBuffer(capacity=replay_capacity)
    replay_buffer.add(states_np, actions_np)

    vae = EpistemicVAE(state_dim=STATE_DIM, action_dim=len(mdp.A), latent_dim=latent_dim).to(DEVICE)
    vae_optimizer = optim.Adam(vae.parameters(), lr=vae_lr)
    bootstrap_losses: List[float] = []
    for _ in range(int(bootstrap_vae_updates)):
        stats = update_vae_from_replay(
            vae, vae_optimizer, replay_buffer, scaler, mdp,
            batch_size=vae_batch_size,
            vae_beta_kl=vae_beta_kl,
        )
        if stats is not None:
            bootstrap_losses.append(stats["loss"])

    if verbose and bootstrap_losses:
        plt.figure(figsize=(8, 4))
        plt.plot(bootstrap_losses)
        plt.title("EpistemicVAE Bootstrap Warm-up Loss")
        plt.xlabel("Update")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        plt.show()
    return vae, vae_optimizer, scaler, replay_buffer, bootstrap_losses


class UCBVAEPolicy:
    """Policy chọn a = argmax_a [Q(s,a) + beta * u_ep(s,a)]."""

    def __init__(
        self,
        qsa: QsaUCB,
        mdp: Any,
        vae: EpistemicVAE,
        scaler: NumpyStandardScaler,
        beta: float = 0.5,
        delta: float = 1.0,
        lam: float = 1.0,
        beta_decay: float = 0.97,
        beta_min: float = 0.01,
    ):
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

    def update_beta_decay(self, episode: int) -> None:
        self.beta = max(self.beta_min, self.beta_init * (self.beta_decay ** int(episode)))

    def scaled_state_tensor(self, state: Sequence[float]) -> torch.Tensor:
        s_np = np.asarray(state, dtype=np.float32).reshape(1, -1)
        return torch.as_tensor(self.scaler.transform(s_np), dtype=torch.float32, device=DEVICE)

    def __call__(self, state: Sequence[float], greedy: bool = False, eps: float = 0.0) -> int:
        self.qsa.eval()
        self.vae.eval()
        with torch.no_grad():
            s_tensor = self.scaled_state_tensor(state)
            q_values = self.qsa(s_tensor).squeeze(0)
            if greedy:
                return int(self.mdp.A[int(torch.argmax(q_values).item())])

            num_actions = len(self.mdp.A)
            s_batch = s_tensor.repeat(num_actions, 1)
            a_onehot_batch = torch.eye(num_actions, dtype=torch.float32, device=DEVICE)
            u_ep = self.vae.compute_u_ep(s_batch, a_onehot_batch, delta=self.delta, lam=self.lam)
            ucb_values = q_values + self.beta * normalize_u_ep_batch(u_ep)
            return int(self.mdp.A[int(torch.argmax(ucb_values).item())])


class ScaledStatesDataset(Dataset):
    """Dataset trajectory đã scale state cho update SARSA bằng mini-batch."""

    def __init__(self, states: Sequence[Sequence[float]], rewards: Sequence[float], actions: Sequence[int], scaler: NumpyStandardScaler):
        self.states = torch.as_tensor(scaler.transform(np.asarray(states[:-1], dtype=np.float32)), dtype=torch.float32)
        self.states_next = torch.as_tensor(scaler.transform(np.asarray(states[1:], dtype=np.float32)), dtype=torch.float32)
        self.rewards = torch.as_tensor(np.asarray(rewards, dtype=np.float32), dtype=torch.float32)
        self.actions = torch.as_tensor(np.asarray(actions, dtype=np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.rewards)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "states": self.states[idx],
            "states_next": self.states_next[idx],
            "rewards": self.rewards[idx],
            "actions": self.actions[idx],
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
    losses: List[float] = []
    for data_pack in dataloader:
        s_batch = data_pack["states"].to(DEVICE)
        sn_batch = data_pack["states_next"].to(DEVICE)
        r_batch = data_pack["rewards"].to(DEVICE)
        actions_tensor = data_pack["actions"].to(DEVICE)
        action_indices = actions_tensor + int(mdp.k)

        # Với DataLoader shuffle=False, next action của SARSA chính là action kế tiếp trong cùng trajectory.
        # Phần tử cuối dùng lại action cuối để giữ shape ổn định.
        next_actions = torch.cat([actions_tensor[1:], actions_tensor[-1:]], dim=0)
        next_action_indices = next_actions + int(mdp.k)

        q_values = qsa(s_batch)
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
        losses.append(float(loss.detach().cpu()))
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
    alpha: float = 1.0,
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
    policy = UCBVAEPolicy(qsa, mdp, vae, scaler, beta=beta, delta=delta, lam=lam, beta_decay=beta_decay, beta_min=beta_min)

    state_init = initial_state_from_first_row(train_series, mdp)
    series = train_series.iloc[1:].reset_index(drop=True)
    learning_curve, q_losses, vae_losses = [], [], []

    iterator = tqdm(range(int(episodes)), desc="Training Deep SARSA-UCB-VAE", disable=not verbose)
    for epi in iterator:
        policy.update_beta_decay(epi)
        states, rewards, actions = mdp.simulate(series, state_init, policy, greedy=False, eps=0.0)
        vae_buffer.add_trajectory(states, actions)

        dataset = ScaledStatesDataset(states, rewards, actions, scaler)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for _ in range(int(nn_epochs)):
            q_losses.extend(_sarsa_update_epoch(qsa, optimizer, dataloader, mdp, gamma, alpha, loss_func))
            for _vae_step in range(int(vae_updates_per_q_batch)):
                stats = update_vae_from_replay(vae, vae_optimizer, vae_buffer, scaler, mdp, vae_batch_size, vae_beta_kl)
                if stats is not None:
                    vae_losses.append(stats["loss"])

        profit = mdp.interact_test(policy, train_series=train_series, test_series=test_series, series_name="test", verbose=False)
        learning_curve.append(float(profit))
    return policy, qsa, learning_curve, vae_losses


# %% [markdown]
# # KHU VỰC 2: MÔ HÌNH CHIẾN LƯỢC EPSILON-GREEDY DEEP SARSA


class EpsilonGreedyPolicy:
    """Policy epsilon-greedy truyền thống dùng cùng critic QsaUCB."""

    def __init__(
        self,
        qsa: QsaUCB,
        mdp: Any,
        scaler: NumpyStandardScaler,
        epsilon_init: float = 1.0,
        epsilon_decay: float = 0.95,
        epsilon_min: float = 0.05,
    ):
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
        return torch.as_tensor(self.scaler.transform(s_np), dtype=torch.float32, device=DEVICE)

    def __call__(self, state: Sequence[float], greedy: bool = False, eps: float = 0.0) -> int:
        self.qsa.eval()
        if (not greedy) and (np.random.rand() < self.epsilon):
            return int(np.random.choice(self.mdp.A))
        with torch.no_grad():
            q_values = self.qsa(self.scaled_state_tensor(state)).squeeze(0)
            return int(self.mdp.A[int(torch.argmax(q_values).item())])


def fit_scaler_from_random_train(train_series: pd.DataFrame, mdp: Any, num_trajectories: int = 4) -> NumpyStandardScaler:
    """Fit scaler cho epsilon-greedy bằng train random trajectories, cùng dữ liệu đầu vào với UCB."""
    scaler, _, _ = fit_scaler_from_bootstrap(train_series, mdp, num_trajectories=num_trajectories)
    return scaler


def train_deep_sarsa_epsilon_greedy(
    mdp: Any,
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    scaler: Optional[NumpyStandardScaler] = None,
    episodes: int = 35,
    gamma: float = 0.95,
    alpha: float = 1.0,
    nn_epochs: int = 1,
    nn_lr: float = 2.5e-5,
    batch_size: int = 128,
    epsilon_init: float = 1.0,
    epsilon_decay: float = 0.95,
    epsilon_min: float = 0.05,
    scaler_bootstrap_trajectories: int = 4,
    verbose: bool = False,
) -> Tuple[EpsilonGreedyPolicy, QsaUCB, List[float], List[float]]:
    """Huấn luyện Deep SARSA epsilon-greedy, không VAE/replay/u_ep."""
    if scaler is None:
        scaler = fit_scaler_from_random_train(train_series, mdp, num_trajectories=scaler_bootstrap_trajectories)

    qsa = QsaUCB(input_size=STATE_DIM, num_classes=len(mdp.A)).to(DEVICE)
    optimizer = optim.Adam(qsa.parameters(), lr=nn_lr)
    loss_func = nn.HuberLoss()
    policy = EpsilonGreedyPolicy(qsa, mdp, scaler, epsilon_init=epsilon_init, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)

    state_init = initial_state_from_first_row(train_series, mdp)
    series = train_series.iloc[1:].reset_index(drop=True)
    learning_curve, q_losses = [], []

    iterator = tqdm(range(int(episodes)), desc="Training Deep SARSA-EpsilonGreedy", disable=not verbose)
    for epi in iterator:
        policy.update_epsilon(epi)
        states, rewards, actions = mdp.simulate(series, state_init, policy, greedy=False, eps=policy.epsilon)
        dataset = ScaledStatesDataset(states, rewards, actions, scaler)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for _ in range(int(nn_epochs)):
            q_losses.extend(_sarsa_update_epoch(qsa, optimizer, dataloader, mdp, gamma, alpha, loss_func))
        profit = mdp.interact_test(policy, train_series=train_series, test_series=test_series, series_name="test", verbose=False)
        learning_curve.append(float(profit))
    return policy, qsa, learning_curve, q_losses


# %% [markdown]
# # KHU VỰC 3: TRỰC QUAN HÓA SO SÁNH (VISUALIZATION) & LƯU MÔ HÌNH


def collect_portfolio_history(mdp: Any, pi: Any, train_series: pd.DataFrame, test_series: pd.DataFrame) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """Thu portfolio value theo từng ngày test với greedy=True."""
    state_init = initial_state_for_test(train_series, mdp)
    states, rewards, actions = mdp.simulate(test_series, state_init, pi, greedy=True, eps=0.0)
    portfolio = np.asarray([s[1] + s[0] * s[2] for s in states], dtype=np.float32)
    return portfolio, states, actions


def calculate_volatility(portfolio_history: Sequence[float]) -> float:
    values = np.asarray(portfolio_history, dtype=np.float32)
    returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
    return float(np.std(returns) * np.sqrt(252) * 100) if len(returns) else 0.0


def calculate_sharpe_ratio(portfolio_history: Sequence[float], risk_free_rate: float = 2.0) -> float:
    values = np.asarray(portfolio_history, dtype=np.float32)
    returns = np.diff(values) / np.maximum(values[:-1], 1e-8)
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    annual_return = np.mean(returns) * 252 * 100
    volatility = np.std(returns) * np.sqrt(252) * 100
    return float((annual_return - risk_free_rate) / volatility) if volatility > 0 else 0.0


def calculate_max_drawdown(portfolio_history: Sequence[float]) -> float:
    values = np.asarray(portfolio_history, dtype=np.float32)
    if len(values) == 0:
        return 0.0
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / np.maximum(peak, 1e-8)
    return float(abs(drawdown.min()) * 100)


def agent_annual_return(initial_capital: float, final_portfolio: float, start_date: Any, end_date: Any) -> Tuple[float, float]:
    total_return = (final_portfolio / initial_capital) - 1.0
    years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    if years <= 0:
        return 0.0, total_return * 100
    annual = (1 + total_return) ** (1 / years) - 1 if (1 + total_return) > 0 else -1.0
    return float(annual * 100), float(total_return * 100)


def compute_metrics_from_runs(result: Dict[str, Any], mdp: Any, test_series: pd.DataFrame) -> Dict[str, float]:
    """Tính metrics trung bình/std từ nhiều runs."""
    profits = np.asarray(result["all_final_profits"], dtype=np.float32)
    mean_profit = float(np.mean(profits))
    std_profit = float(np.std(profits))
    final_portfolio = float(mdp.balance_init + mean_profit)
    arr, roi = agent_annual_return(mdp.balance_init, final_portfolio, test_series.iloc[0]["time"], test_series.iloc[-1]["time"])
    mean_portfolio = np.mean(np.asarray(result["all_portfolio_histories"], dtype=np.float32), axis=0)
    return {
        "Average Final Profit": mean_profit,
        "Std Final Profit": std_profit,
        "ARR (%)": arr,
        "ROI (%)": roi,
        "Volatility (%)": calculate_volatility(mean_portfolio),
        "Sharpe Ratio": calculate_sharpe_ratio(mean_portfolio),
        "Max Drawdown (%)": calculate_max_drawdown(mean_portfolio),
    }


def _run_single_ucb(train_series: pd.DataFrame, test_series: pd.DataFrame, mdp: Any, config: Dict[str, Any]):
    vae, vae_optimizer, scaler, vae_buffer, bootstrap_losses = initialize_joint_vae_components(
        train_series=train_series,
        mdp=mdp,
        latent_dim=config.get("vae_latent_dim", 16),
        vae_lr=config.get("vae_lr", 1e-3),
        replay_capacity=config.get("vae_replay_capacity", 50_000),
        bootstrap_trajectories=config.get("bootstrap_trajectories", 4),
        bootstrap_vae_updates=config.get("bootstrap_vae_updates", 100),
        vae_batch_size=config.get("vae_batch_size", 128),
        vae_beta_kl=config.get("vae_beta_kl", 1e-3),
        verbose=False,
    )
    policy, qsa, learning_curve, aux_losses = train_deep_sarsa_ucb_vae(
        mdp=mdp,
        train_series=train_series,
        test_series=test_series,
        vae=vae,
        vae_optimizer=vae_optimizer,
        vae_buffer=vae_buffer,
        scaler=scaler,
        episodes=config.get("episodes", 35),
        gamma=config.get("gamma", 0.95),
        alpha=config.get("alpha", 1.0),
        nn_epochs=config.get("nn_epochs", 1),
        nn_lr=config.get("nn_lr", 2.5e-5),
        beta=config.get("beta", 0.2),
        delta=config.get("delta", 1.0),
        lam=config.get("lam", 1.0),
        batch_size=config.get("batch_size", 128),
        vae_batch_size=config.get("vae_batch_size", 128),
        vae_updates_per_q_batch=config.get("vae_updates_per_q_batch", 1),
        vae_beta_kl=config.get("vae_beta_kl", 1e-3),
        beta_decay=config.get("beta_decay", 0.97),
        beta_min=config.get("beta_min", 0.01),
        verbose=False,
    )
    final_profit = float(mdp.interact_test(policy, train_series=train_series, test_series=test_series, series_name="test", verbose=False))
    portfolio, states, actions = collect_portfolio_history(mdp, policy, train_series, test_series)
    return {"policy": policy, "qsa": qsa, "vae": vae, "scaler": scaler, "learning_curve": learning_curve, "final_profit": final_profit, "portfolio_history": portfolio, "states": states, "actions": actions, "aux_losses": aux_losses, "bootstrap_losses": bootstrap_losses}


def _run_single_eps(train_series: pd.DataFrame, test_series: pd.DataFrame, mdp: Any, config: Dict[str, Any]):
    scaler = fit_scaler_from_random_train(train_series, mdp, num_trajectories=config.get("scaler_bootstrap_trajectories", config.get("bootstrap_trajectories", 4)))
    policy, qsa, learning_curve, aux_losses = train_deep_sarsa_epsilon_greedy(
        mdp=mdp,
        train_series=train_series,
        test_series=test_series,
        scaler=scaler,
        episodes=config.get("episodes", 35),
        gamma=config.get("gamma", 0.95),
        alpha=config.get("alpha", 1.0),
        nn_epochs=config.get("nn_epochs", 1),
        nn_lr=config.get("nn_lr", 2.5e-5),
        batch_size=config.get("batch_size", 128),
        epsilon_init=config.get("epsilon_init", 1.0),
        epsilon_decay=config.get("epsilon_decay", 0.95),
        epsilon_min=config.get("epsilon_min", 0.05),
        verbose=False,
    )
    final_profit = float(mdp.interact_test(policy, train_series=train_series, test_series=test_series, series_name="test", verbose=False))
    portfolio, states, actions = collect_portfolio_history(mdp, policy, train_series, test_series)
    return {"policy": policy, "qsa": qsa, "scaler": scaler, "learning_curve": learning_curve, "final_profit": final_profit, "portfolio_history": portfolio, "states": states, "actions": actions, "aux_losses": aux_losses}


def _aggregate_runs(agent_name: str, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = max(runs, key=lambda r: r["final_profit"])
    curves = np.asarray([r["learning_curve"] for r in runs], dtype=np.float32)
    portfolios = np.asarray([r["portfolio_history"] for r in runs], dtype=np.float32)
    return {
        "agent": agent_name,
        "runs": runs,
        "all_final_profits": [float(r["final_profit"]) for r in runs],
        "all_learning_curves": [r["learning_curve"] for r in runs],
        "learning_curve": np.mean(curves, axis=0),
        "std_learning_curve": np.std(curves, axis=0),
        "all_portfolio_histories": [r["portfolio_history"] for r in runs],
        "portfolio_history": np.mean(portfolios, axis=0),
        "best_portfolio_history": best["portfolio_history"],
        "best_policy": best["policy"],
        "trained_agent": best["qsa"],
        "best_run": best,
    }


def print_markdown_summary(summary_df: pd.DataFrame) -> None:
    """In bảng Markdown; fallback nếu môi trường thiếu tabulate."""
    try:
        print(summary_df.to_markdown(index=False, floatfmt=".4f"))
    except Exception:
        print(summary_df.to_string(index=False))


def plot_comparative_results(result_ucb: Dict[str, Any], result_eps: Dict[str, Any], test_series: pd.DataFrame, title_suffix: str = "HPG") -> None:
    """Vẽ 3 biểu đồ khoa học: learning curve, percentile, portfolio over time."""
    # 1. Learning curves
    plt.figure(figsize=(12, 5))
    for result, color in [(result_ucb, "tab:blue"), (result_eps, "tab:orange")]:
        x = np.arange(len(result["learning_curve"]))
        mean = np.asarray(result["learning_curve"], dtype=np.float32)
        std = np.asarray(result["std_learning_curve"], dtype=np.float32)
        plt.plot(x, mean, label=result["agent"], linewidth=2, color=color)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    plt.title(f"Learning Curves - {title_suffix}")
    plt.xlabel("Episode")
    plt.ylabel("Test Profit ($)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Percentile profit distributions
    levels = np.arange(0, 101, 5)
    plt.figure(figsize=(10, 6))
    plt.plot(levels, np.percentile(result_ucb["all_final_profits"], levels), marker="o", label="UCB-VAE", color="tab:blue")
    plt.plot(levels, np.percentile(result_eps["all_final_profits"], levels), marker="s", label="Epsilon-Greedy", color="tab:orange")
    plt.title(f"Percentile Profit Distributions - {title_suffix}")
    plt.xlabel("Percentile (%)")
    plt.ylabel("Final Profit ($)")
    plt.xticks(levels)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Portfolio value over time: dùng best run để trực quan kịch bản tốt nhất đạt được
    x_time = test_series["time"]
    plt.figure(figsize=(14, 6))
    plt.plot(x_time, result_ucb["best_portfolio_history"][1:], label="UCB-VAE best run", color="tab:blue", linewidth=2)
    plt.plot(x_time, result_eps["best_portfolio_history"][1:], label="Epsilon-Greedy best run", color="tab:orange", linewidth=2)
    plt.title(f"Portfolio Value Over Time - {title_suffix}")
    plt.xlabel("Trading day")
    plt.ylabel("Portfolio Value ($)")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_best_models(result_ucb: Dict[str, Any], result_eps: Dict[str, Any], models_dir: Optional[Path] = None) -> Path:
    """Lưu state_dict Q-network tốt nhất của hai chiến lược."""
    if models_dir is None:
        models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result_ucb["trained_agent"].state_dict(), models_dir / "sarsa_ucb_vae_best.pth")
    torch.save(result_eps["trained_agent"].state_dict(), models_dir / "sarsa_epsilon_greedy_best.pth")
    return models_dir


def run_comparative_experiment(
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    config_ucb: Dict[str, Any],
    config_eps: Dict[str, Any],
    num_runs: int = 20,
    mdp: Optional[Any] = None,
    seed: int = SEED,
    title_suffix: str = "HPG",
    save_models: bool = True,
    plot: bool = True,
) -> Dict[str, Any]:
    """Chạy đối chứng công bằng UCB-VAE vs Epsilon-Greedy trên cùng train/test HPG."""
    if mdp is None:
        if stockMDP is None:
            raise ImportError(f"Không import được StockTradingMDP: {_IMPORT_ERROR}")
        mdp = stockMDP(
            balance_init=config_ucb.get("balance_init", config_eps.get("balance_init", 1000)),
            k=config_ucb.get("k", config_eps.get("k", 5)),
            min_balance=config_ucb.get("min_balance", config_eps.get("min_balance", -100)),
        )

    ucb_runs, eps_runs = [], []
    for run in tqdm(range(int(num_runs)), desc=f"Comparative experiment {title_suffix}"):
        run_seed = seed + run
        set_global_seed(run_seed)
        ucb_runs.append(_run_single_ucb(train_series, test_series, mdp, config_ucb))
        set_global_seed(run_seed)  # cùng seed nền để đối chứng nhiễu random công bằng nhất có thể
        eps_runs.append(_run_single_eps(train_series, test_series, mdp, config_eps))
        print(f"Run {run + 1:02d}/{num_runs} | UCB={ucb_runs[-1]['final_profit']:.2f} | EPS={eps_runs[-1]['final_profit']:.2f}")

    result_ucb = _aggregate_runs("Deep SARSA-UCB-VAE (Joint VAE Training)", ucb_runs)
    result_eps = _aggregate_runs("Deep SARSA-Epsilon-Greedy", eps_runs)

    metrics_ucb = compute_metrics_from_runs(result_ucb, mdp, test_series)
    metrics_eps = compute_metrics_from_runs(result_eps, mdp, test_series)
    summary_df = pd.DataFrame([
        {"Strategy": "UCB-VAE", **metrics_ucb},
        {"Strategy": "Epsilon-Greedy", **metrics_eps},
    ])

    print("\n## Comparative Summary")
    print_markdown_summary(summary_df)

    if plot:
        plot_comparative_results(result_ucb, result_eps, test_series, title_suffix=title_suffix)

    models_dir = None
    if save_models:
        models_dir = save_best_models(result_ucb, result_eps)
        print(f"Saved best Q-networks to: {models_dir}")

    return {"ucb_vae": result_ucb, "epsilon_greedy": result_eps, "summary": summary_df, "models_dir": models_dir}


# %% [markdown]
# # Cấu hình mẫu và cách chạy trên HPG GOOD/BAD


DEFAULT_CONFIG_UCB = {
    "episodes": 45,
    "gamma": 0.95,
    "alpha": 1.0,
    "nn_epochs": 1,
    "nn_lr": 2.5e-5,
    "beta": 0.20,
    "delta": 1.0,
    "lam": 1.0,
    "beta_decay": 0.91,
    "beta_min": 0.01,
    "vae_latent_dim": 16,
    "vae_lr": 1e-3,
    "vae_beta_kl": 1e-2,
    "vae_batch_size": 128,
    "vae_updates_per_q_batch": 1,
    "vae_replay_capacity": 50_000,
    "bootstrap_trajectories": 5,
    "bootstrap_vae_updates": 100,
    "balance_init": 1000,
    "k": 5,
    "min_balance": -100,
}

DEFAULT_CONFIG_EPS = {
    "episodes": 45,
    "gamma": 0.95,
    "alpha": 1.0,
    "nn_epochs": 1,
    "nn_lr": 2.5e-5,
    "epsilon_init": 1.0,
    "epsilon_decay": 0.91,
    "epsilon_min": 0.01,
    "scaler_bootstrap_trajectories": 5,
    "balance_init": 1000,
    "k": 5,
    "min_balance": -100,
}


def dry_run_policy_components() -> None:
    """Kiểm tra nhẹ policy/Q/VAE trên một state giả, không chạy training dài."""
    if stockMDP is None:
        raise ImportError(f"Không import được StockTradingMDP: {_IMPORT_ERROR}")
    mdp = stockMDP(balance_init=1000, k=5, min_balance=-100)
    dummy_states = np.random.randn(64, STATE_DIM).astype(np.float32)
    scaler = NumpyStandardScaler().fit(dummy_states)
    qsa = QsaUCB(input_size=STATE_DIM, num_classes=len(mdp.A)).to(DEVICE)
    vae = EpistemicVAE(state_dim=STATE_DIM, action_dim=len(mdp.A), latent_dim=16).to(DEVICE)
    policy_ucb = UCBVAEPolicy(qsa, mdp, vae, scaler)
    policy_eps = EpsilonGreedyPolicy(qsa, mdp, scaler)
    sample_state = [100.0, 1000.0, 0.0, 0.1, 50.0, 120.0, 44.0]
    print("UCB action:", policy_ucb(sample_state), "| EPS greedy action:", policy_eps(sample_state, greedy=True), "| DEVICE:", DEVICE)


if __name__ == "__main__":
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("DEVICE =", DEVICE)
    dry_run_policy_components()
    print("File UCB_2.py đã sẵn sàng. Trong notebook, gọi load_hpg_good_bad_data() và run_comparative_experiment(...).")
