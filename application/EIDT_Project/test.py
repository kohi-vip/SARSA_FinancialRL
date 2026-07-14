import json
import re
from pathlib import Path
import pandas as pd
path = Path('SARSA_FinancialRL/application/EIDT_Project/thucnghiem.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

def src(i):
    s = nb['cells'][i].get('source', '')
    return ''.join(s) if isinstance(s, list) else s

def set_src(i, s):
    nb['cells'][i]['source'] = s

# ---------------- Cell 5: UCB-VAE raw-state pipeline ----------------
s = src(5)

new_update_vae = '''def update_vae_from_replay(
    vae: EpistemicVAE,
    vae_optimizer: optim.Optimizer,
    replay_buffer: VAEReplayBuffer,
    mdp: Any,
    batch_size: int,
    vae_beta_kl: float = 1e-3,
) -> Optional[Dict[str, float]]:
    """Sample replay buffer và update VAE một bước trên state thô, không chuẩn hóa."""
    if len(replay_buffer) == 0:
        return None
    states_np, actions_np = replay_buffer.sample(batch_size)
    actions_onehot = actions_to_onehot(actions_np, mdp)
    
    s_batch = torch.as_tensor(states_np, dtype=torch.float32, device=DEVICE)
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

'''
s = re.sub(r'# SỬA LỖI UNPACKING: Hàm update_vae_from_replay.*?\n\ndef initialize_joint_vae_components', new_update_vae + 'def initialize_joint_vae_components', s, flags=re.S)

new_init = '''def initialize_joint_vae_components(
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
) -> Tuple[EpistemicVAE, optim.Optimizer, VAEReplayBuffer, List[float]]:
    """Khởi tạo VAE + replay buffer cho joint training trên state thô, không chuẩn hóa."""
    states_np, actions_np = collect_vae_samples(
        train_series, mdp, num_trajectories=bootstrap_trajectories
    )
    replay_buffer = VAEReplayBuffer(capacity=replay_capacity)
    replay_buffer.add(states_np, actions_np)
    
    vae = EpistemicVAE(state_dim=STATE_DIM, action_dim=len(mdp.A), latent_dim=latent_dim).to(DEVICE)
    vae_optimizer = optim.Adam(vae.parameters(), lr=vae_lr)
    bootstrap_losses = []
    
    for _ in range(int(bootstrap_vae_updates)):
        stats = update_vae_from_replay(
            vae, vae_optimizer, replay_buffer, mdp,
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
        
    return vae, vae_optimizer, replay_buffer, bootstrap_losses

'''
s = re.sub(r'def initialize_joint_vae_components\(.*?\nclass UCBVAEPolicy', new_init + 'class UCBVAEPolicy', s, flags=re.S)

new_ucb_policy = '''class UCBVAEPolicy:
    """Policy chọn a = argmax_a [Q(s, a) + beta * u_ep(s, a)] trên state thô."""
    def __init__(self, qsa, mdp, vae, beta=0.5, delta=1.0, lam=1.0, beta_decay=0.97, beta_min=0.01, episode_max=100):
        self.qsa = qsa
        self.mdp = mdp
        self.vae = vae
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

    def state_tensor(self, state: Sequence[float]) -> torch.Tensor:
        s_np = np.asarray(state, dtype=np.float32).reshape(1, -1)
        return torch.as_tensor(s_np, dtype=torch.float32, device=DEVICE)

    def __call__(self, state: Sequence[float], greedy: bool = False, eps: float = 0.0) -> int:
        self.qsa.eval()
        self.vae.eval()
        with torch.no_grad():
            s_tensor = self.state_tensor(state)
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

'''
s = re.sub(r'class UCBVAEPolicy:.*?\nclass ScaledStatesDataset', new_ucb_policy + 'class ScaledStatesDataset', s, flags=re.S)

new_raw_dataset = '''class RawStatesDataset(Dataset):
    """Dataset trajectory dùng state thô để update SARSA bằng mini-batch."""
    def __init__(self, states, rewards, actions):
        self.states = torch.tensor(np.asarray(states[:-1]), dtype=torch.float32)
        self.states_next = torch.tensor(np.asarray(states[1:]), dtype=torch.float32)
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

'''
s = re.sub(r'class ScaledStatesDataset\(Dataset\):.*?\n\ndef _sarsa_update_epoch', new_raw_dataset + 'def _sarsa_update_epoch', s, flags=re.S)

new_train_ucb = '''def train_deep_sarsa_ucb_vae(
    mdp: Any,
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    vae: EpistemicVAE,
    vae_optimizer: optim.Optimizer,
    vae_buffer: VAEReplayBuffer,
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
    """Huấn luyện Deep SARSA UCB-VAE với joint VAE training online trên state thô."""
    qsa = QsaUCB(input_size=STATE_DIM, num_classes=len(mdp.A)).to(DEVICE)
    optimizer = optim.Adam(qsa.parameters(), lr=nn_lr)
    loss_func = nn.HuberLoss()
    policy = UCBVAEPolicy(qsa, mdp, vae, beta=beta, delta=delta, lam=lam, beta_decay=beta_decay, beta_min=beta_min, episode_max=episodes)
    
    state_init = initial_state_from_first_row(train_series, mdp)
    series = train_series.iloc[1:].reset_index(drop=True)
    learning_curve, vae_losses = [], []
    
    iterator = tqdm(range(int(episodes)), desc="Training Deep SARSA-UCB-VAE-Raw", disable=not verbose)
    for epi in iterator:
        policy.update_beta_decay(epi)
        
        states, rewards, actions = mdp.simulate(series, state_init, policy, greedy=False)
        
        vae_buffer.add_trajectory(states, actions)
        
        dataset = RawStatesDataset(states, rewards, actions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Train Q-network
        q_losses = _sarsa_update_epoch(qsa, optimizer, dataloader, mdp, gamma, alpha, loss_func)
        
        # Train VAE song song trên replay state thô
        for _ in range(vae_updates_per_q_batch):
            stats = update_vae_from_replay(vae, vae_optimizer, vae_buffer, mdp, vae_batch_size, vae_beta_kl)
            if stats is not None:
                vae_losses.append(stats["loss"])
                
        profit = mdp.interact_test(policy, train_series=train_series, test_series=test_series, verbose=False)
        learning_curve.append(float(profit))
        
    return policy, qsa, learning_curve, vae_losses
'''
s = re.sub(r'def train_deep_sarsa_ucb_vae\(.*', new_train_ucb, s, flags=re.S)
set_src(5, s)

# ---------------- Cell 7: Epsilon-Greedy raw-state pipeline ----------------
set_src(7, '''class EpsilonGreedyPolicy:
    """Policy epsilon-greedy truyền thống dùng state thô, không chuẩn hóa."""
    def __init__(self, qsa: QsaUCB, mdp: Any, epsilon_init=1.0, epsilon_decay=0.95, epsilon_min=0.05):
        self.qsa = qsa
        self.mdp = mdp
        self.epsilon_init = float(epsilon_init)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.epsilon = float(epsilon_init)

    def update_epsilon(self, episode: int) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon_init * (self.epsilon_decay ** int(episode)))

    def state_tensor(self, state: Sequence[float]) -> torch.Tensor:
        s_np = np.asarray(state, dtype=np.float32).reshape(1, -1)
        return torch.tensor(s_np, dtype=torch.float32, device=DEVICE)

    def __call__(self, state: Sequence[float], greedy: bool = False, eps: float = 0.0) -> int:
        self.qsa.eval()
        if (not greedy) and (np.random.rand() < self.epsilon):
            return int(np.random.choice(self.mdp.A))
        with torch.no_grad():
            q_values = self.qsa(self.state_tensor(state)).squeeze(0)
            return int(self.mdp.A[int(torch.argmax(q_values).item())])

def train_deep_sarsa_epsilon_greedy(
    mdp: Any,
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    episodes: int = 35,
    gamma: float = 0.95,
    alpha: float = 0.6,
    nn_epochs: int = 1,
    nn_lr: float = 2.5e-5,
    batch_size: int = 128,
    epsilon_init: float = 1.0,
    epsilon_decay: float = 0.95,
    epsilon_min: float = 0.05,
    verbose: bool = False,
) -> Tuple[EpsilonGreedyPolicy, QsaUCB, List[float], List[float]]:
    """Huấn luyện Deep SARSA epsilon-greedy đối chứng trên state thô."""
    qsa = QsaUCB(input_size=STATE_DIM, num_classes=len(mdp.A)).to(DEVICE)
    optimizer = optim.Adam(qsa.parameters(), lr=nn_lr)
    loss_func = nn.HuberLoss()
    policy = EpsilonGreedyPolicy(qsa, mdp, epsilon_init, epsilon_decay, epsilon_min)
    
    state_init = initial_state_from_first_row(train_series, mdp)
    series = train_series.iloc[1:].reset_index(drop=True)
    learning_curve, q_losses = [], []
    
    iterator = tqdm(range(int(episodes)), desc="Training Deep SARSA-EpsilonGreedy-Raw", disable=not verbose)
    for epi in iterator:
        policy.update_epsilon(epi)
        
        states, rewards, actions = mdp.simulate(series, state_init, policy, greedy=False)
        
        dataset = RawStatesDataset(states, rewards, actions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        losses = _sarsa_update_epoch(qsa, optimizer, dataloader, mdp, gamma, alpha, loss_func)
        q_losses.extend(losses)
        
        profit = mdp.interact_test(policy, train_series=train_series, test_series=test_series, verbose=False)
        learning_curve.append(float(profit))
        
    return policy, qsa, learning_curve, q_losses

''')

# ---------------- Cell 9: runner/aggregate no scaler ----------------
s = src(9)
s = s.replace('vae, vae_optimizer, scaler, vae_buffer, bootstrap_losses = initialize_joint_vae_components(', 'vae, vae_optimizer, vae_buffer, bootstrap_losses = initialize_joint_vae_components(')
s = s.replace('        scaler=scaler,\n', '')
s = s.replace('        "scaler": scaler,', '        "scaler": None,')
s = s.replace('        "scaler": policy.scaler,', '        "scaler": None,')
s = s.replace('        "vae_scaler": best_run.get("scaler", None),', '        "vae_scaler": None,')
set_src(9, s)

# ---------------- Cell 10: config/dry run no scaler ----------------
s = src(10)
s = s.replace('    "scaler_bootstrap_trajectories": 5,\n', '')
old = '''    dummy_states = np.random.randn(64, STATE_DIM).astype(np.float32)
    scaler = NumpyStandardScaler().fit(dummy_states)
    qsa = QsaUCB(input_size=STATE_DIM, num_classes=len(mdp.A)).to(DEVICE)
    vae = EpistemicVAE(state_dim=STATE_DIM, action_dim=len(mdp.A), latent_dim=16).to(DEVICE)
    policy_ucb = UCBVAEPolicy(qsa, mdp, vae, scaler)
    policy_eps = EpsilonGreedyPolicy(qsa, mdp, scaler)
'''
new = '''    qsa = QsaUCB(input_size=STATE_DIM, num_classes=len(mdp.A)).to(DEVICE)
    vae = EpistemicVAE(state_dim=STATE_DIM, action_dim=len(mdp.A), latent_dim=16).to(DEVICE)
    policy_ucb = UCBVAEPolicy(qsa, mdp, vae)
    policy_eps = EpsilonGreedyPolicy(qsa, mdp)
'''
s = s.replace(old, new)
set_src(10, s)

path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Updated', path)