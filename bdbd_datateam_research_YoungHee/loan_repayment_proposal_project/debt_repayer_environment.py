import numpy as np


class DebtRepayer(object):
    def __init__(self, args, start_debt, duration=120, interest_rate=3.75, round=0) -> None:
        self.start_debt = start_debt
        self.current_debt = start_debt
        self.duration = duration
        self.t = 0
        self.coin_balance = 0
        self.interest_rate = interest_rate / 100 # 은행 대출 이자 - 고정금리 가정
        self.round = round

        # coin dist args
        self.up_a = args['up_a']
        self.up_b = args['up_b']
        self.down_a = args['down_a']
        self.down_b = args['down_b']
        self.up_roll = args['up_roll']
        self.down_roll = args['down_roll']

        # action_space grid
        self.repay_grid, self.withdraw_grid = np.meshgrid(np.arange(10, 70, 10), np.arange(0, 110, 10))

        # history
        self.history = []

    def reset(self):
        self.current_debt = self.start_debt
        self.coin_balance = 0
        self.history = []
        self.t = 0

    def get_one_episode(self, policy):
        for action in policy:
            sample = self.forward_one_month((action[0], action[1]))
            if sample[3][1] == 0: break
            if sample[3][0] > 120: break

        return self.history
            
    def forward_one_month(self, action: tuple):
        past_state = (self.t, self.current_debt, self.coin_balance)

        repay_amount = self.repay_grid[action]
        withdraw_pct = self.withdraw_grid[action] / 100
        interest_amount = round(self.current_debt * self.interest_rate / 12, self.round)

        repay_amount = min(60 - interest_amount, repay_amount) # 은행 이자 제외하고 남은 금액에서만 값을 수 있음.
        coin_input = 60 - repay_amount - interest_amount
        
        withdrawed_coin = self.get_coin_balance(coin_input, withdraw_pct)
        total_repay_amount = repay_amount + withdrawed_coin

        self.current_debt -= total_repay_amount
        if self.current_debt <= 0:
            self.current_debt = 0
            done = 0
        else:
            done = 1
        self.t += 1
        prime_state = (self.t, self.current_debt, self.coin_balance)

        sample = (past_state, action, interest_amount, prime_state, done)
        self.history.append(sample)
        
        return sample

    # 다음달 코인 잔액 및 인출액 생성기 - 
    def get_coin_balance(self, current_input, withdraw_pct):
        #상승 or 하락으로 정해서 beta 분포로 변동 설정함
        if np.random.rand() >= 0.5:
            change_rate = np.random.beta(a=self.up_a, b=self.up_b) + self.up_roll
        else:
            change_rate = -(np.random.beta(a=self.down_a, b=self.down_b) + self.down_roll)
        # 코인 잔액 업데이트하고 인출액 반환 - FIXME 순서확인
        withdrawal_amount = round(self.coin_balance * withdraw_pct, self.round)
        temp_balance = self.coin_balance - withdrawal_amount   # 당월 인출
        temp_balance += current_input # 당월 입금
        self.coin_balance = temp_balance + round(temp_balance * change_rate, self.round)
        if self.coin_balance < 0: self.coin_balance = 0

        return withdrawal_amount


if __name__ == '__main__':

    env = DebtRepayer(5000, )
    history = env.get_one_episode([(1, 0)]*120)
    print(history)