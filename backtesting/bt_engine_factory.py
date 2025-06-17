import backtrader as bt
import logging
from matplotlib import pyplot as plt
import pandas as pd

class BacktraderEngineFactory:
    """ Backtrader 回测引擎工厂 """

    def __init__(self, strategy, data, cash=3700, commission=0.0003, leverage=1, analyzers=True):
        self.cerebro = bt.Cerebro()
        self.strategy = strategy
        self.data = bt.feeds.PandasData(dataname=data)
        self.cash = cash
        self.commission = commission
        self.analyzers = analyzers  # 是否开启分析器
        self.leverage = leverage

    def setup(self):
        """ ✅ 设置回测环境 """

        # **添加策略**
        self.cerebro.addstrategy(self.strategy)

        # **添加数据**
        self.cerebro.adddata(self.data)

        # **初始资金**
        self.cerebro.broker.set_cash(self.cash)

        # **交易成本**
        self.cerebro.broker.setcommission(commission=self.commission, leverage=self.leverage)


        # **是否添加分析器**
        if self.analyzers:
            self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        logging.info("✅ Backtrader Engine 配置完成")

    def run(self):
        """ 🚀 运行回测 """
        self.setup()
        results = self.cerebro.run()
        self.report(results)
        return results
    

    def report(self, results):
        """ 📊 输出回测结果 """
        strat = results[0]

        # 获取分析器的结果
        returns_analysis = strat.analyzers.returns.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()

        print(returns_analysis)
        # 年化收益率
        if 'rnorm100' in returns_analysis and returns_analysis['rnorm100'] is not None:
            print(f"📊 年化收益率: {returns_analysis['rnorm100']:.2f}%")
        else:
            print("⚠️ 年化收益率数据不可用")

        # 最大回撤
        print(drawdown_analysis)
        if 'max' in drawdown_analysis and 'drawdown' in drawdown_analysis['max']:
            print(f"📉 最大回撤: {drawdown_analysis['max']['drawdown']:.2f}%")
        else:
            print("⚠️ 最大回撤数据不可用")

        print(sharpe_analysis)
        # 夏普比率
        if 'sharperatio' in sharpe_analysis and sharpe_analysis['sharperatio'] is not None:
            print(f"📈 夏普比率: {sharpe_analysis['sharperatio']:.2f}")
        else:
            print("⚠️ 夏普比率数据不可用")

        trade_df = pd.DataFrame(strat.order_log)
        trade_df.to_csv("./log/order_log.csv", index=False)
        #可视化结果
        plt.figure(figsize=(12, 6))
        plt.plot(strat.value_history, label="Cash Balance")
        plt.xlabel("Days")
        plt.ylabel("Cash(￥)")
        plt.title("Account Cash Balance Over Time")
        plt.legend()
        plt.savefig("./log/cash_balance.png")
        plt.show()

    def plot(self):
        """ 📈 绘制回测结果 """
        self.cerebro.plot(style='candlestick')
