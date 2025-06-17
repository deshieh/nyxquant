from dependency_injector import containers, providers
from data_pipeline.data_manager import DataManager
from data_pipeline.data_collector import DataCollector
from data_pipeline.data_processing import arima_garch_prep, transformer_prep, xgb_prep
from data_pipeline.data_sources.biance_source import BinanceSource
from data_pipeline.data_sources.tushare_source import TushareSource
from data_pipeline.data_sources.wind_source import WindSource
from data_pipeline.data_sources.yfinance_source import YfinanceSource


class DataContainer(containers.DeclarativeContainer):

    #数据源容器
    tushare_source = providers.Singleton(TushareSource)
    biance_source = providers.Singleton(BinanceSource)
    wind_source = providers.Singleton(WindSource)
    yfinance_source = providers.Singleton(YfinanceSource)


    #数据收集容器
    data_collector = providers.Singleton(DataCollector)

    #数据预处理容器
    transformer_prep = providers.Factory(transformer_prep.TransformerPrep)
    arima_garch_prep = providers.Factory(arima_garch_prep.ArimaGarchPrep)
    xgb_prep = providers.Factory(xgb_prep.XGBPrep)

    #数据管理容器
    data_manager = providers.Singleton(DataManager)

data_container = DataContainer()