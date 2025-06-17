from collections import deque
from datetime import time, datetime, timedelta
import math
import backtrader as bt
import logging
import numpy as np
from torch import sigmoid
from risk_manager.risk_manager import RiskManager
from models.model_container import model_container
import pandas as pd
from infrastructure.global_config import config_container

class IndexBtXGB(bt.Strategy):
    """ 指数交易策略,模型XGBoost """
    params = (
        ("asset_type", "index"),
        ("asset", "000001.SH"),
        ("stop_loss", 0.05),
        ("max_drawdown", 0.15),
        ("short_window", 10),  #计算短期预测准确率/相关性的窗口
        ("order_log", True),  
        ("warmup_period", 40)  #冷启动，期间不交易，只做预测和记录数据
    )
    
    def __init__(self):
        self.model_registry = model_container.model_registry()
        self.risk_manager = RiskManager()
        self.training_manager = model_container.training_manager()
        self.triggered_drawdown = False
        self.triggered_meltdown = False
        self.order_log = []
        self.success_list = []  #记录预测方向成功与否
        self.active_sell_order = None  # 跟踪当前活跃的卖出订单
        self.value_history = []  #账户价值变化
        self.return_record = pd.DataFrame(columns=["date","pred_return","actual_return"])
        self.bar_count = 0 #记录运行bar数
        self.risk_exposure = 1  #风险暴露限制
        self.cold_day = 0
        self.max_drawdown = 0.15
        self.pin_value  = 0
        self.main_order = None  # 主买单
        self.child_orders = []  # 记录子单（止盈、止损）     
        self.entry_value = None
        self.entry_price = None
        self.value_history = []
        self.peak_value = self.broker.get_value()
        self.pnl_list = deque(maxlen=100)


    def next(self):
        """ 每日执行策略逻辑 """
        self.bar_count += 1
        current_date = self.data.datetime.date(0)  # 当前日期
        current_value = self.broker.get_value() 
        current_position = self.position.size
        available_cash = self.broker.get_cash()
        holding_value = current_value - available_cash
        print(f"日期：{current_date} 可用现金 {available_cash} 持仓头寸：{current_position} 持仓市值：{holding_value} 账户总值：{current_value}")     

        # **准备XGB模型**   
        self.training_manager.should_train(self.params.asset_type, self.params.asset, "XGB", current_date)
        print("开始获取xgb模型")
        model = self.model_registry.get_model(self.params.asset_type, "XGB")
        if not model:
            print(f"⚠️ {self.params.asset} 缺少xgb模型，跳过交易")
            return
        #预测趋势（收益率）
        pred_return = model.predict(self.params.asset_type, self.params.asset, current_date)
        print(f"预测收益率：{pred_return:.2%}")

        #买卖逻辑（简单版）
        if not self.position:
            #计算凯莉仓位
            b = self._calc_avg_win_loss_ratio()
            p = self._calc_win_rate()
            kelly = max(0, p - (1 - p) / b if b else 0.5)
            kelly = min(kelly, 1.0)
            invest_cash = available_cash * kelly

            if pred_return > 0.001:
                # size = invest_cash / self.data.close[0]
                self.buy()
                self.entry_value = self.broker.get_value()
                self.entry_price = self.data.close[0]
                self.peak_value = self.entry_value
                print(f"✅ {current_date}开多")
                
            elif pred_return < -0.001:
                # size = invest_cash / self.data.close[0]
                self.sell()
                self.entry_value = self.broker.get_value()
                self.entry_price = self.data.close[0]
                self.peak_value = self.entry_value
                print(f"⚠️ {current_date}开空")

        else:
            value = self.broker.get_value()
            drawdown = (self.peak_value - value) / self.peak_value
            if value > self.peak_value:
                self.peak_value = value

            trend_reversed = (
                (self.position.size > 0 and pred_return < -0.001) or
                (self.position.size < 0 and pred_return > 0.001)
            )

            if drawdown > self.params.stop_loss or trend_reversed:
                print(f"⛔ 退出：回撤 {drawdown:.2%}, 趋势反转: {trend_reversed}")
                self.close()

        total_drawdown = (self.peak_value - self.broker.get_value()) / self.peak_value
        if total_drawdown > self.params.max_drawdown:
            print(f"🚨 触发账户最大回撤限制：{total_drawdown:.2%}")
            self.close()

        self.value_history.append(self.broker.get_value())  # 记录账户价值变化，用于画图
            

    def notify_order(self, order):
        """ 处理订单状态更新 """
        order_info = {
            'datetime': self.data.datetime.date(0),
            'ref': order.ref,
            'type': 'BUY' if order.isbuy() else 'SELL',
            'status': order.getstatusname(),
            'size': order.created.size if order.created else 0,
            'price': order.created.price,
            'executed_size': order.executed.size if order.executed else 0,
            'executed_price': order.executed.price if order.executed else None,
        }
        self.order_log.append(order_info)
    
    def notify_trade(self, trade):
        if trade.isclosed:
            print(f"💹 交易完成 PnL: {trade.pnl:.2f}")
            self.pnl_list.append(trade.pnl)
    
    def _calc_win_rate(self):
        if not self.pnl_list:
            return 0.5
        return np.mean([p > 0 for p in self.pnl_list])

    def _calc_avg_win_loss_ratio(self):
        wins = [p for p in self.pnl_list if p > 0]
        losses = [-p for p in self.pnl_list if p < 0]
        if not losses:
            return 2.0
        return np.mean(wins) / np.mean(losses) if wins else 0.5
        
    
