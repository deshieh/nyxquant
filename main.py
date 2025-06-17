import pandas as pd
from backtesting.bt_engine_factory import BacktraderEngineFactory
from backtesting.bt_strategies.index_bt_xgb import IndexBtXGB
from data_pipeline.data_container import data_container
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter('ignore', category=pd.errors.PerformanceWarning)
data = data_container.data_manager().get_data(asset_type="index", asset="000001.SH", start_date="20220101", end_date="20250430")
engine = BacktraderEngineFactory(IndexBtXGB, data)
engine.run()
engine.plot()