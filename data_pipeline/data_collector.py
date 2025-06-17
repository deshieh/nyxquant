class DataCollector:
    """
    负责从不同的数据源拉取数据，并返回标准化的 Pandas DataFrame
    """

    def __init__(self):
        from data_pipeline.data_container import data_container
        self.tushare = data_container.tushare_source()
        self.wind = data_container.wind_source()
        self.binance = data_container.biance_source()
        self.yfinance = data_container.yfinance_source()

    def collect_data(self, asset_type, asset, start_date, end_date, 
                  data_type="ohlc", bar_type="1d", structure="series", features=None):
        """
        加载数据，根据资产类别选择不同的数据源

        :param asset_type: "index", "stock", "crypto", "forex", "bond"
        :param asset: 具体资产，如 "000300.SH", "AAPL", "BTC/USDT"
        :param start_date: 数据开始日期
        :param end_date: 数据结束日期
        :param data_type: "ohlc", "returns", "factors", "fundamental", "macro", "sentiment"
        :param bar_type: "1d", "5m", "tick"
        :param structure: "series"（时间序列）或 "cross_section"（截面）
        :param features: 需要的特征，如 ["SMA", "MACD", "P/E"]
        :return: Pandas DataFrame
        """
        if asset_type in ["index", "stock"]:
            return self.tushare.get_data(asset_type, asset, start_date, end_date)

        elif asset_type == "crypto":
            return self.binance.get_data(asset, start_date, end_date, bar_type)

        elif asset_type == "forex":
            return self.yfinance.get_data(asset, start_date, end_date, data_type, bar_type)

        elif asset_type == "bond":
            return self.wind.get_data(asset, start_date, end_date, data_type, bar_type)

        else:
            raise ValueError(f"未知的资产类型: {asset_type}")

