import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from collections import deque
from itertools import product
from tqdm import tqdm

from debt_repayer_environment import DebtRepayer

class ReplayBuffer(object):
    """A class for store episodes"""
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)

    def put(self, transition):
        self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)

    def get_sample(self, n):
        batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        sample = (
            torch.tensor(s_lst, dtype=torch.float),
            a_lst,
            torch.tensor(r_lst, dtype=torch.float),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst, dtype=torch.float),
        )
        return sample

class Qnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.qnet = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 60),
        )

    def forward(self, x):
        return self.qnet(x)

    def sample_action(self, obs, eps):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs)
        out = self.forward(obs)
        if random.random() < eps:
            return random.randint(0, 59)
        else:
            return out.argmax().item()


state_space = np.arange(0, 5000, 1)
debt_repay = np.arange(0, 60, 1)
coin_withdrawal = np.arange(0, 100, 1)
action_space = list(product(debt_repay, coin_withdrawal))
reward_space = state_space * 3.75e-2 / 12

lr = 1e-4
gamma = 1
buffer_limit = 5000
batch_size = 32


def train(q, q_target, buffer, opt, gamma, batch_size):
    s, a, r, s_prime, done_mask = buffer.get_sample(batch_size)

    q_out = q(s)
    action_idx = torch.tensor([action_space.index(i) for i in a], dtype=torch.int64).unsqueeze(1)
    q_a = q_out.gather(1, action_idx)
    max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
    target = -r + gamma * max_q_prime * done_mask
    loss = F.smooth_l1_loss(q_a, target)

    opt.zero_grad()
    loss.backward()
    opt.step()

def main(epoch: int):
    # set envs
    env = DebtRepayer(start_debt=5000, round=4)
    buffer = ReplayBuffer(buffer_limit)

    # set model
    q = Qnet()
    q_target = Qnet()
    opt = optim.AdamW(q.parameters(), lr=lr)
    q_target.load_state_dict(q.state_dict())
    
    # set training loop
    G = 0
    eps = 1
    pbar = tqdm(range(epoch), desc="Deep Q net trainin : ", miniters=10)
    for epi in pbar:
        eps = max(eps * 0.9997, 0.001)
        env.reset()
        s = [env.t, env.start_debt, env.coin_balance]

        # put samples until episode ends
        flag = False
        while not flag: 
            a = q.sample_action(s, eps)
            sample = env.forward_one_month(action_space[a])
            buffer.put((sample))
            s = sample[3]
            G += sample[2]
            if s[1] == 0: break
            if s[0] >= 120: break

        # training when buffer got enough samples ten backward per one loop
        if len(buffer) > 2000:
            for _ in range(10):
                train(q, q_target, buffer, opt, gamma, batch_size)

        # q_target copy trained q every 20 loop
        if epi % 20 == 0 and epi != 0:
            q_target.load_state_dict(q.state_dict())

            q_1a = action_space[q(torch.Tensor((0, 5000, 0))).argmax().item()]
            pbar.set_postfix_str("n_episode : {}, Q1 : {}, interest_amount : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(epi, q_1a, G, len(buffer), eps))

            G = 0
    
    print("Training done!!")
    return q


if __name__ == "__main__":
    main(30000)