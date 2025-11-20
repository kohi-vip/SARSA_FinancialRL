
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from environments.mdp import *
class Qsa(nn.Module):
    def __init__(self, input_size=7, num_classes=len(A)):
        super().__init__()
        self.fc_liner = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.fc_liner(x)

class StatesDataset(Dataset):
    def __init__(self, states, rewards, actions):
        self.states = torch.Tensor(states[:-1]).float()
        self.states_next = torch.Tensor(states[1:]).float()
        self.rewards = torch.Tensor(rewards).float()
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'states': self.states[idx],
            'states_next': self.states_next[idx],
            'rewards': self.rewards[idx],
            'actions': self.actions[idx]
        }

def deep_sarsa(qsa, 
               series, 
               state_init, 
               pi, 
               optimizer,
               loss_func,
               epochs=10,
               episode=100, 
               gamma=0.9,
               lr=0.7,
               eps=0.5,
               min_eps=0.05,
               decay=0.9,
               greedy=False,
               verbose=True
              ):
    """
    Deep SARSA: Thuật toán TD điều khiển theo chính sách (On-policy)
    Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
    trong đó a' là hành động thực tế được chính sách chọn tại trạng thái s'
    """
    losses = list()
    learning_curve = list()
    
    # Loop for each episode
    for epi in tqdm(range(episode), desc="Training Deep SARSA"):
        # Giảm epsilon để khám phá
        eps *= decay

        # Sinh một trajectory bằng chính sách hiện tại
        states, rewards, actions = simulate(series, state_init, pi, greedy, eps=max(min_eps, eps))

        # Tạo dataset và data loader
        dataset = StatesDataset(states, rewards, actions)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

        # Huấn luyện mạng nơ-ron
        for epo in range(epochs):
            for data_pack in dataloader:
                # Q hiện tại Q(s,a)
                input_tensor = data_pack['states']
                out = qsa(input_tensor)

                # Chuyển actions sang list nếu là tensor
                actions_list = data_pack['actions'].tolist() if torch.is_tensor(data_pack['actions']) else list(data_pack['actions'])
                current_q = out[[i for i in range(len(data_pack['rewards']))], 
                               [a+k for a in actions_list]]

                # Mục tiêu: r + γQ(s',a') với a' là hành động tiếp theo thực tế
                with torch.no_grad():
                    next_q_values = qsa(data_pack['states_next'])

                    # SARSA: dùng giá trị Q của hành động tiếp theo thực tế
                    # Với trạng thái cuối, dùng hành động cuối vì không có hành động tiếp theo
                    actions_list = data_pack['actions'].tolist() if torch.is_tensor(data_pack['actions']) else list(data_pack['actions'])
                    next_actions = actions_list[1:] + [actions_list[-1]]
                    next_action_indices = [a+k for a in next_actions]

                    next_q = next_q_values[[i for i in range(len(next_action_indices))], 
                                          next_action_indices]

                    # Mục tiêu cập nhật SARSA
                    target_q = data_pack['rewards'] + gamma * next_q

                # Cập nhật mềm: trộn Q cũ với mục tiêu mới
                target_tensor = (1 - lr) * current_q + lr * target_q

                # Tính loss và cập nhật trọng số
                loss = loss_func(current_q, target_tensor)
                qsa.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().item())

        # Đánh giá trên tập test sau mỗi episode
        learning_curve.append(interact_test(pi, series_name='test', verbose=False))
    
    # Vẽ kết quả huấn luyện
    if verbose:
        print(f"Final loss: {losses[-1]:.6f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Vẽ đồ thị loss
        ax1.plot(losses, linewidth=1, alpha=0.7)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')
        ax1.set_title('Deep SARSA Training Loss')
        ax1.grid(alpha=0.3)

        # Vẽ đồ thị đường học (learning curve)
        ax2.plot(learning_curve, linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Profit ($)')
        ax2.set_title('Deep SARSA Learning Curve (Test Set)')
        ax2.grid(alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('deep_sarsa_training.pdf')
        plt.show()

    return learning_curve