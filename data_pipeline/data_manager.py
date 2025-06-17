import logging
import os
import shutil
import pandas as pd
import redis
import pickle
class DataManager:
    """ 统一管理数据加载、缓存、清洗、特征工程 """

    def __init__(self, cache_enabled=True):
        from data_pipeline.data_container import data_container
        self.data_collector = data_container.data_collector()
        self.cache_enabled = cache_enabled
        self.cache = redis.Redis(host='localhost', port=6379, decode_responses=False) if cache_enabled else None

    def get_data(self, asset_type, asset, start_date, end_date, 
                 data_type="ohlc", bar_type="1d", structure="series",
                 features=None, update_freq="daily"):
        """
        获取数据，优先从本地/缓存获取，如果没有，则调用 DataLoader 下载并存储。
        """
        key = f"{asset_type}_{asset}_{data_type}_{bar_type}_{structure}_{update_freq}"
        start = pd.to_datetime(start_date, format="%Y%m%d")
        end = pd.to_datetime(end_date, format="%Y%m%d")

        # 1️⃣ 先检查本地存储
        file_path = f"./data/{key}.parquet"
        try:
            df = pd.read_parquet(file_path)
            if df.index.min() <= start and df.index.max() >= end:
                print(f"Local cache hit for {key}")
                return df.loc[start:end]
            else:
                print(f"Parquet data not enough for {key}")
        except FileNotFoundError:
            pass

        # 2️⃣ 再检查 Redis 缓存
        try:
            if self.cache_enabled and self.cache.exists(key):
                df = pickle.loads(self.cache.get(key))
                if df.index.min() <= start and df.index.max() >= end:
                    print(f"Redis cache hit for {key}")
                    return df.loc[start:end]
                else:
                    print(f"Redis data not enough for {key}")
        except Exception as e:
            print(f"Redis error for {key}: {e}")

        # 3️⃣ 没有缓存，则调用 DataLoader 获取最大原始数据
        default_start_date = "20000101"
        default_end_date = pd.Timestamp.today().strftime("%Y%m%d")
        logging.info(f"Fetching new data from DataCollector for {key}")
        df = self.data_collector.collect_data(asset_type, asset, start_date=default_start_date, end_date=default_end_date)
        if  df.empty:
            print(f"Data is empty for {key}")
            return None
        else:
            # 4️⃣ 存储数据
            if self.cache_enabled:
                try:
                    df.to_parquet(file_path)  #存到parquet
                    print(f"Data stored in Parquet for {key}")
                    self.cache.set(key, pickle.dumps(df))  # 存到 Redis
                    print(f"Data stored in Redis for {key}")
                except Exception as e:
                    print(f"Data store error for {key}: {e}")

            if df.index.min() <= start and df.index.max() >= end:
                    print(f"DataCollector hit for {key}")
                    return df.loc[start:end]
            else:
                print(f"Parquet data not enough for {key}")
                return None

            
    @classmethod
    def clear_all_parquet_cache(cls):
        """
        清除所有本地 Parquet 文件缓存。
        """
        directory = "./data/"
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)  # 删除整个目录及其内容
                logging.info("All Parquet caches have been cleared.")
            else:
                logging.info("No Parquet directory found.")
            # 重新创建空目录以保持系统一致性
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logging.error(f"Error clearing Parquet cache directory: {e}")

    @classmethod
    def clear_all_redis_cache(cls, redis_instance):
        """
        清除所有 Redis 缓存。
        """
        try:
            keys = redis_instance.keys("*")  # 获取所有 Redis 键
            if keys:
                redis_instance.delete(*keys)  # 删除所有键
                logging.info("All Redis caches have been cleared.")
            else:
                logging.info("No Redis keys found.")
        except Exception as e:
            logging.error(f"Error clearing all Redis cache: {e}")


    

