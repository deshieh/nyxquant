import pandas as pd
import tushare as ts
from infrastructure.global_config import config_container

class TushareSource():
    """Tushare 数据源"""

    def __init__(self):
        self.token = config_container.config().TUSHARE_TOKEN
        self.pro = ts.pro_api(self.token)

    def get_data(self, asset_type, asset, start_date, end_date, data_type="ohlc", bar_type="1d", structure="series"):
        """获取数据"""
        if asset_type == "index":
            df = self.pro.idx_factor_pro(ts_code=asset, start_date=start_date, end_date=end_date)
            #处理数据符合bt框架
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            df.sort_index(inplace=True)
            df.rename(columns={"vol": "volume"}, inplace=True)
            return df
        
        elif asset_type == "stock":
            df = self.pro.daily(ts_code=asset, start_date=start_date, end_date=end_date)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            df.rename(columns={"vol": "volume"}, inplace=True)            
            return df
        else:
            raise ValueError(f"Tushare不支持的数据源: {asset_type}")

