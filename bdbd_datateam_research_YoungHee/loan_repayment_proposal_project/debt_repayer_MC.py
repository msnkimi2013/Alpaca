import numpy as np
from itertools import product
from tqdm import tqdm
from collections import deque
from threading import Thread
from multiprocessing import Pool, cpu_count

from debt_repayer_environment import DebtRepayer

class MCMDP_agent(object):
    """Monte Carlow approach of Markov Decision Process for Debt Repayer problem"""
    def __init__(self, start_debt: int, interest_rate: float, env_args, gamma=1, ) -> None:
        super().__init__()
        # set environment
        #self.env = DebtRepayer(env_args, start_debt, interest_rate)
        self.start_debt = start_debt
        self.interest_rate = interest_rate
        self.env_args = env_args

        # set variables
        self.gamma = gamma  # 현재가치 할인율

        # set base spaces
        self.state_space = np.arange(0, start_debt+1, 1)
        debt_repay = np.arange(10, 70, 10) # 1만원 단위로 갚을 수 있음
        coin_withdrawal = np.arange(0, 100, 10) # 1% 단위로 코인 인출 가능
        self.xx, self.yy = np.meshgrid(debt_repay, coin_withdrawal)
        self.action_space = list(product(debt_repay, coin_withdrawal))
        self.q_table = np.zeros((start_debt+1, 10, 6))
        self.q_visit = np.zeros_like(self.q_table)
        self.q1_history = deque(maxlen=len(self.state_space))
        self.eps = 1
        
        # set ques
        self.eps_que = deque([1]*100, maxlen=100)
        self.q_table_que = deque([np.zeros_like(self.q_table)]*100, maxlen=100)

    def train_parallel(self, epoch):
        cores = cpu_count()
        pool = Pool(cores)
        pool.map(self.train_single_step, (True, range(epoch)))   
        pool.close()
        pool.join()
        print("Done!")
        self.q_table = self.q_table_que.pop()

    def train_thread(self, epoch):
        thread_list = deque(maxlen=100)
        pbar = tqdm(range(epoch), desc='MC for Debt Repyer')
        for i in pbar:
            thread = Thread(target=self.train_single_step2, args=(False, i), name=i)
            thread.start()
            thread_list.append(thread)

            """
            if len(thread_list) // 100:
                for _ in range(100): 
                    t = thread_list.popleft()
                    t.join()
            """

    def train_single_step(self, *args):
        eps = self.eps_que.popleft()
        _policy, _q_table = self.policy_improve(eps, use_que=True)
        _q_table, reward = self.policy_evaluate(_policy, _q_table)
        # put in que again
        self.eps_que.append(max(eps * 0.9997, 0.0001))
        self.q_table_que.append(_q_table)
        self.q1_history.append(np.unravel_index(_q_table[-1].argmax(), _q_table.shape))

    def train_single_step2 (self, *args):
        _policy, _q_table = self.policy_improve(self.eps, use_que=False)
        _q_table, reward = self.policy_evaluate(_policy, _q_table)
        # put in que again
        self.eps = max(self.eps * 0.9997, 0.0001)
        self.q_table = _q_table
        self.q1_history.append(np.unravel_index(_q_table[-1].argmax(), _q_table.shape))

    def train(self, epoch):
        eps = 1
        pbar = tqdm(range(epoch), desc='MC for Debt Repyer', miniters=10)
        for i in pbar:
            eps = max(eps * 0.9997, 0.0001)
            _policy, _q_table = self.policy_improve(eps)
            _q_table = self.policy_evaluate(_policy, _q_table)                                                   
            self.q1_history.append(_q_table[-1].argmax())
            self.q_table_que.append(_q_table)

    def policy_evaluate(self, policy, q_table):
        _history = DebtRepayer(self.env_args, self.start_debt, self.interest_rate).get_one_episode(policy)
        cum_reward = 0
        for s, a, r, _, _ in _history[::-1]:
            q_s_a = q_table[int(s[0]),a[0],a[1]] + self.gamma * (cum_reward - q_table[int(s[0]),a[0],a[1]])
            q_table[int(s[0]),a[0],a[1]] = q_s_a
            #self.q_visit[int(s[0]),a[0],a[1]] += 1
            cum_reward -= r
        return q_table, cum_reward

    def policy_improve(self, eps, use_que=False):
        """vectorized policy generator considering epsilon exploration"""
        l = len(self.state_space)
        if use_que:
            q_table = self.q_table_que.popleft()
        else:
            q_table = self.q_table

        greedy_policy = np.array([np.unravel_index(q.argmax(), q.shape) for q in q_table])
        random_policy = np.stack([np.random.randint(0, 10, l), 
                                  np.random.randint(0, 6, l)], axis=1)
                                  
        random_mask = np.random.rand(l)
        random_mask = np.expand_dims(random_mask, axis=1).repeat(2, axis=1)
        _policy = np.where(random_mask > eps, greedy_policy, random_policy)
        return _policy[::-1], q_table

    @property
    def policy(self):
        # 0 부터 state 시작하니까 env 에서 거꾸로 참조되서 뒤집어 줘야함.
        _policy = np.array([np.unravel_index(q.argmax(), q.shape) for q in self.q_table])
        return _policy[::-1]


if __name__ == '__main__':

    args = {'up_a': 1.5,
            'up_b': 5.5,
            'down_a': 1.5,
            'down_b': 5.5,
            'up_roll': 0.0,
            'down_roll': 0.0}
    # 싱글코어
    agent = MCMDP_agent(5000, 0.0375, args)
    agent.train(500)
    del agent

    # 멀티쓰레드 방식 - 훨 느려짐... 특히나 iter 커지면서 생성하는 쓰레드 많아질수록
    agent1 = MCMDP_agent(5000, 0.0375, args)
    agent1.train_thread(500)
    del agent1

    # 멀티프로세스 방식 - 메모리를 공유를 안하니 애초에 쓸모가 없음. 결과 다 끝나고 합치는 형태밖에 안되서
    agent2 = MCMDP_agent(5000, 0.0375, args)
    agent2.train_parallel(500)
    print(agent2.policy)