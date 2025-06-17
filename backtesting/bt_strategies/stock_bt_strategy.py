import backtrader as bt
import logging
import pandas as pd

class BacktraderEngine:
    def __init__(self, cash=100000, commission=0.001):
        self.cerebro = bt.Cerebro()
        self.cash = cash
        self.commission = commission

    def add_strategy(self, strategy, **kwargs):
        self.cerebro.addstrategy(strategy, **kwargs)

    def add_data(self, data):
        self.cerebro.adddata(data)

    def setup(self):
        self.cerebro.broker.set_cash(self.cash)
        self.cerebro.broker.setcommission(commission=self.commission)

    def run(self):
        return self.cerebro.run()

    def report(self, results):
        strat = results[0]
        logging.info(f"📊 年化收益率: {strat.analyzers.returns.get_analysis()['rnorm100']:.2f}%")
        logging.info(f"📉 最大回撤: {strat.analyzers.drawdown.get_analysis()['maxdrawdown']:.2f}%")
        logging.info(f"📈 夏普比率: {strat.analyzers.sharpe.get_analysis()['sharperatio']:.2f}")

        trade_df = pd.DataFrame(strat.trade_log)
        trade_df.to_csv("trade_log.csv", index=False)
