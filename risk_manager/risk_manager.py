import logging


class RiskManager:
    def __init__(self):
        self.highest_capital = 0  # 记录最高资金点
        self.max_drawdown = 0  #最大回撤限制

#是否设置为静态方法
    def check_drawdown(self, current_capital, max_drawdown):
        self.max_drawdown = max_drawdown
        self.highest_capital = max(self.highest_capital, current_capital)
        drawdown = 1 - (current_capital / self.highest_capital)

        if drawdown > self.max_drawdown:
            logging.warning(f"⚠️ 触发最大回撤止损！当前回撤: {drawdown:.2%}，最大允许: {self.max_drawdown:.2%}")
            return True
        return False

    def kelly_criterion(self, win_rate, risk_reward_ratio):
        """ 计算凯利仓位 """
        kelly_fraction = win_rate - (1 - win_rate) / risk_reward_ratio
        return max(0, min(kelly_fraction, 1))  # 确保仓位在 [0, 1] 之间

