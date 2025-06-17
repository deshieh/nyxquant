from datetime import time, datetime, timedelta
import backtrader as bt
import logging
import numpy as np
from torch import sigmoid
from risk_manager.risk_manager import RiskManager
from models.model_container import model_container
import pandas as pd
from infrastructure.global_config import config_container
#收益率都是10天的对数收益率

class IndexBtTransformer(bt.Strategy):
    """ 指数交易策略,模型Transformer """

    params = (
        ("asset_type", "index"),
        ("asset", "000001.SH"),
        ("daily_rf", 0.0001173),  
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

    def next(self):
        """ 每日执行策略逻辑 """
        self.bar_count += 1
        current_date = self.data.datetime.date(0)  # 当前日期
        pred_date = current_date + timedelta(days=10)  #'T+10'日
        current_value = self.broker.get_value() 
        current_position = self.position.size
        available_cash = self.broker.get_cash()
        holding_value = current_value - available_cash
        print(f"日期：{current_date} 可用现金 {available_cash} 持仓头寸：{current_position} 持仓市值：{holding_value} 账户总值：{current_value}")



        # **准备Transformer模型**
        self.training_manager.should_train(self.params.asset_type, self.params.asset, "Transformer", current_date)
        print("开始获取Transformer模型")
        transformer = self.model_registry.get_model(self.params.asset_type, "Transformer")
        if not transformer:
            print(f"⚠️ {self.params.asset} 缺少transformer模型，跳过交易")
            return

        # **在 `T` 预测T+10的持有期收益率、波动率、预测一致性，储存预测收益率**
        pred_return, pred_volatility, pred_consistency = transformer.predict(self.params.asset_type, self.params.asset, current_date)
        new_pred = pd.DataFrame({"date": [pred_date], "pred_return": [pred_return], "actual_return": [np.nan]})
        self.return_record = pd.concat([self.return_record, new_pred], ignore_index=True)

        print(f"预测10天后收益率{pred_return} 波动{pred_volatility} ")
        # **计算T-1相对于T-11的真实收益率**
        if len(self) > 10:
            actual_return = np.log(self.data.close[0]) - np.log(self.data.close[-10])
            #找到对应的预测收益率，储存真实收益率
            idx = self.return_record[self.return_record["date"] == current_date].index
            if len(idx) > 0:
                self.return_record.loc[idx, "actual_return"] = actual_return

        """冷启动判定"""
        if self.bar_count < self.params.warmup_period:
            print(f"⏳ 冷启动期 {self.bar_count}/{self.params.warmup_period} 天，暂不交易")
            return
        """冷启动结束，开始交易"""

            
        # **处理预测/真实收益率记录**
        matched_return_record = self.return_record.dropna()
        if len(matched_return_record) >= self.params.short_window:
            matched_return_record["accuracy"] = (np.sign(matched_return_record["pred_return"]) == np.sign(matched_return_record["actual_return"])).astype(int)

            #计算长短期预测准确率
            short_term_acc = matched_return_record["accuracy"].iloc[-self.params.short_window:].mean()
            long_term_acc = matched_return_record["accuracy"].mean()
            print(f"截至 [{current_date}] 短期准确率: {short_term_acc:.2%}, 长期准确率: {long_term_acc:.2%}")
            #计算长短期预测相关性
            short_term_corr = matched_return_record["pred_return"].iloc[-self.params.short_window:].corr(matched_return_record["actual_return"].iloc[-self.params.short_window:])
            long_term_corr = matched_return_record["pred_return"].corr(matched_return_record["actual_return"])
            print(f"截至 [{current_date}] 短期预测相关性: {short_term_corr:.2%}, 长期预测相关性: {long_term_corr:.2%},预测一致性：{pred_consistency}")


        """ARIMA-GARCH模型1天预测"""
        # **准备ARIMA-GARCH模型**
        self.training_manager.should_train(self.params.asset_type, self.params.asset, "ARIMA-GARCH", current_date)
        arima_garch = self.model_registry.get_model(self.params.asset_type, "ARIMA-GARCH")

        # **在 `T` 预测T+1的价格期望值和单日波动率
        next_mean, next_vol = arima_garch.predict(self.params.asset_type, self.params.asset, current_date)

        # **预测交易价格区间**
        long_price = next_mean - 0.95 * next_vol

        """模型熔断机制"""
        if self.cold_day > 0:
            self.cold_day -= 1

        if self.triggered_meltdown and short_term_acc >0.4:
            print(f"✅ 模型熔断恢复，短期准确率回升至 {short_term_acc:.2%}")
            self.triggered_meltdown = False

        if short_term_acc < 0.4 and not self.triggered_meltdown and self.cold_day == 0 :
            self.pin_value = self.broker.get_value()
            print(f"🚨 触发模型熔断！重新训练模型")
            self.triggered_meltdown = True
            self.training_manager.train_model(self.params.asset_type, self.params.asset, "Transformer", current_date)
            self.cold_day = 10
            return
        
        
        """检查最大回撤，暂时不用
        if self.position:   
            if self.risk_manager.check_drawdown(current_value, self.max_drawdown):
                self.close()
                print(f"⛔ ⛔⛔触发最大回撤，清仓！⛔⛔⛔")
        """

        """ 买多逻辑 """

        if self.main_order and self.main_order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            print(f"{self.data.datetime.date(0)}: 未成交的主买单被取消")
            self.cancel(self.main_order)

        if (pred_return >= 0.03
            and available_cash > 0
            and pred_consistency > 0.5):
            #计算调整凯利仓位
            win_rate = long_term_acc
            odds = 3
            kelly = self.risk_manager.kelly_criterion(win_rate, odds)
            adjusted_kelly = kelly * pred_consistency
            adjusted_kelly = max(0, adjusted_kelly)
            target_investment = available_cash * adjusted_kelly * self.risk_exposure
            print(f"Kelly公式计算 winrate:{win_rate} odds:{odds} 凯利仓位:{kelly},调整凯利仓位:{adjusted_kelly},目标追加金额{target_investment}")
            if target_investment > 0:
                print("⭐⭐⭐触发买入信号，开始计算投入资金⭐⭐⭐")
                # 提交买单（有限期至当日收盘)
                target_size = target_investment / long_price
                orders = self.buy_bracket(
                    size = target_size,
                    price = long_price,
                    limitprice = long_price * (1 + pred_return),
                    trailpercent=0.3
                )
                self.main_order = orders[0]
                self.child_orders.append(orders[1:])  # 记录止盈止损子单
                print(f"[{current_date}] 提交买入单，买入头寸{target_size} 价格={long_price}")

        elif (pred_return <  -0.03 and self.position.size > 0):
            print("🏃🏃🏃‍下降趋势警告！！！准备清仓！！！")
            if self.child_orders:
                for order in self.child_orders:
                    if order.status in [bt.Order.Submitted, bt.Order.Accepted]:
                        print(f"{self.data.datetime.date(0)}: 准备清仓，未成交的所有止盈止损被取消")
                        self.cancel(order)      
            print("关闭所有仓位！！！")
            self.close()
              
        """

        #闲置资金投入无风险市场    
        #interest = self.broker.getcash() * self.params.daily_rf
        #self.broker.add_cash(interest)
        #print(f"💴💴💴获得利息{interest}💴💴💴")            
        """
        
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
        if order.status == order.Completed:
            if order == self.main_order:
                print(f"{self.data.datetime.date(0)}: 主买单成交，价格：{order.executed.price}")
        elif order.status in [order.Canceled, order.Expired]:
            if order == self.main_order:
                print(f"{self.data.datetime.date(0)}: 主买单取消或过期")
