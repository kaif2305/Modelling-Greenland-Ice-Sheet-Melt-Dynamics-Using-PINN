import pandas as pd
import numpy as np
from pathlib import Path
from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def clean_and_combine(self):
        """
        Main orchestration method:
        1. Cleans individual stations (GPS, Albedo, Time Engineering).
        2. Merges all stations into one long-format Master Dataset.
        3. Applies a robust multi-stage backstop to eliminate all input NaNs.
        """
        combined_list = []
        
        # Use the station list provided by the ConfigurationManager (from config.yaml)
        for station in self.config.stations:
            file_path = self.config.data_path / f"{station}_daily.csv"
            
            if not file_path.exists():
                logger.warning(f"File not found for {station} at {file_path}, skipping...")
                continue
            
            logger.info(f"Step 1: Transforming station: {station}")
            
            # Load data - 'time' becomes the index
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # --- A. Physical & Space Interpolation ---
            # Fill Albedo with the winter constant defined in config.yaml (0.85)
            df['albedo'] = df['albedo'].fillna(self.config.winter_albedo)

            # Interpolate GPS coordinates (Must be continuous for PINN spatial derivatives)
            gps_cols = ['gps_lat', 'gps_lon', 'gps_alt']
            df[gps_cols] = df[gps_cols].interpolate(method='linear', limit_direction='both')

            # --- B. Time Engineering for PINN ---
            # Continuous Time: Days since start of project (Linear 't' input for PDE)
            df['time_cont'] = (df.index - pd.Timestamp("2008-01-01")).days
            
            # Cyclical Time: Sine/Cosine encoding to capture annual seasonality
            df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
            df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

            # --- C. Metadata & Schema Alignment ---
            df['station_name'] = station
            
            # Reset index so 'time' is available for the merge
            df = df.reset_index()

            # Filter for the exact columns required by the PINN Schema
            final_cols = [
                'time', 'time_cont', 'day_sin', 'day_cos', 
                'gps_lat', 'gps_lon', 'gps_alt', 
                't_u', 'rh_u', 'wspd_u', 'albedo', 
                'station_name', 't_surf'
            ]
            
            df = df[[c for c in final_cols if c in df.columns]]
            combined_list.append(df)

        # --- STEP 2: MERGE STATIONS ---
        master_df = pd.concat(combined_list, axis=0, ignore_index=True)
        logger.info(f"Step 2: Merged {len(self.config.stations)} stations into master dataset.")

        # --- STEP 3: THE FINAL BACKSTOP (Robust Input Cleaning) ---
        input_features = ['t_u', 'rh_u', 'wspd_u']
        
        logger.info("Step 3: Applying robust backstop cleaning for PINN input stability...")

        for col in input_features:
            if col in master_df.columns:
                # 1. Internal Interpolation: Catch gaps inside the station records
                master_df[col] = master_df.groupby('station_name')[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
                
                # 2. Edge Case Filling: Catch gaps at the very start or end of the records
                master_df[col] = master_df.groupby('station_name')[col].transform(
                    lambda x: x.ffill().bfill()
                )

        # --- STEP 4: GLOBAL SAFETY NET ---
        # Final resort: If a variable is missing for an entire station, fill with global mean.
        for col in input_features:
            if master_df[col].isnull().any():
                global_mean = master_df[col].mean()
                master_df[col] = master_df[col].fillna(global_mean)
                logger.warning(f"Extreme case: Filled remaining NaNs in {col} with global average: {global_mean}")

        # --- STEP 5: FINAL VERIFICATION ---
        input_nans = master_df[input_features].isnull().sum().sum()
        target_nans = master_df['t_surf'].isnull().sum()
        
        if input_nans == 0:
            logger.info(f"SUCCESS: Input features are 100% clean. Rows: {len(master_df)}")
            logger.info(f"Physics Collocation Points (Target NaNs): {target_nans}")
        else:
            logger.error(f"CRITICAL: {input_nans} NaNs still remain in inputs!")

        # Save the master file for the Training stage
        master_df.to_csv(self.config.transformed_path, index=False)
        logger.info(f"Master Space-Time dataset saved to: {self.config.transformed_path}")