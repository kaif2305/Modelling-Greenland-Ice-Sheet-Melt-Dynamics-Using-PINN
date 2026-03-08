import pandas as pd
from mlProject import logger
from mlProject.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_and_resample_to_daily(self):
        """
        Fetches hourly data for all stations defined in config,
        resamples to daily averages, and saves as local CSVs.
        """
        for station in self.config.stations:
            # Construct the hourly URL dynamically
            target_url = f"{self.config.base_url}hour/{station}_hour.csv"
            
            # Construct where the daily file will be saved
            local_file_path = Path(self.config.local_data_dir) / f"{station}_daily.csv"
            
            logger.info(f"Fetching hourly data for {station}...")
            
            try:
                # 1. Load the hourly data
                df_hourly = pd.read_csv(
                    target_url, 
                    comment='#', 
                    index_col=0, 
                    na_values=['', 'nan'], 
                    parse_dates=True
                )
                
                # 2. Resample to daily averages
                logger.info(f"Resampling {station} to daily averages...")
                df_daily = df_hourly.resample('1D').mean()
                
                # 3. Save the daily data locally
                df_daily.to_csv(local_file_path)
                logger.info(f"Successfully saved {station} daily data to {local_file_path}")
                
            except Exception as e:
                logger.error(f"Failed processing data for {station}: {e}")
                raise e