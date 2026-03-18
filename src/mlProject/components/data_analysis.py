import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mlProject import logger
from mlProject.entity.config_entity import DataAnalysisConfig

class DataAnalysis:
    def __init__(self, config: DataAnalysisConfig):
        self.config = config
        # Set style for professional research graphs
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("talk")

    def analyze_missing_values(self):
        """
        Dynamically scans the input directory for all station CSVs,
        calculates missing values strictly for the schema variables,
        and saves a unified JSON report.
        """
        input_dir = Path(self.config.input_data_dir)
        report = {}

        # Scan the directory for any CSV files (no hardcoded station names!)
        csv_files = list(input_dir.glob("*_daily.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV data found in {input_dir}")
            return

        for file_path in csv_files:
            station_name = file_path.stem.replace("_daily", "")
            logger.info(f"Analyzing missing data for station: {station_name}...")
            
            try:
                # Load the dataset
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Filter for variables that exist in BOTH the schema and the CSV
                available_vars = [var for var in self.config.target_variables if var in df.columns]
                
                # Identify if any schema columns are completely missing from the station
                missing_vars = [var for var in self.config.target_variables if var not in df.columns]
                if missing_vars:
                    logger.warning(f"Station {station_name} is completely missing schema columns: {missing_vars}")

                # Calculate missing values
                null_counts = df[available_vars].isnull().sum().to_dict()
                total_rows = len(df)
                
                # Add to the master report
                report[station_name] = {
                    "total_rows": total_rows,
                    "null_counts": null_counts,
                    "missing_schema_columns": missing_vars
                }
                
            except Exception as e:
                logger.error(f"Failed analyzing data for {station_name}: {e}")
                raise e

        # Save the final report as a JSON file in the reports directory
        report_path = Path(self.config.reports_dir) / "missing_values_report.json"
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Successfully saved missing values report to {report_path}")

    def generate_research_graphs(self):
        """
        Generates visualizations of the raw data problems (NaNs, GPS gaps, non-linearities)
        to be used in the research paper, complete with professional annotations.
        """
        logger.info("Generating research graphs for data analysis...")
        
        # We will use KAN_L as the representative station for the graphs
        file_path = Path(self.config.input_data_dir) / "KAN_L_daily.csv"
        
        if not file_path.exists():
            logger.warning(f"Could not find {file_path} for graphing. Skipping graphs.")
            return

        df_raw = pd.read_csv(file_path, parse_dates=['time'])
        
        # ==========================================
        # GRAPH 1: Missing Sensors (The Math Crash Risk)
        # ==========================================
        plt.figure(figsize=(12, 5))
        window = df_raw[(df_raw['time'] > '2010-01-01') & (df_raw['time'] < '2012-01-01')]
        
        plt.plot(window['time'], window['t_u'], color='red', marker='.', linestyle='', alpha=0.7)

        window_nans = window[window['t_u'].isna()]
        if not window_nans.empty:
            start_gap = window_nans['time'].iloc[0]
            end_gap = window_nans['time'].iloc[-1]
            plt.axvspan(start_gap, end_gap, color='gray', alpha=0.2)

        plt.title("Raw Air Temperature ($T_u$) - Discontinuous Inputs", fontsize=14, pad=15)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Air Temp (°C)", fontsize=12)
        plt.tight_layout()
        plt.savefig(Path(self.config.reports_dir) / "analysis_missing_sensors.png", dpi=300)
        plt.close()

        # ==========================================
        # GRAPH 2: Broken GPS (The Teleportation Bug)
        # ==========================================
        if 'gps_alt' in df_raw.columns:
            plt.figure(figsize=(12, 5))
            plt.plot(df_raw['time'], df_raw['gps_alt'], color='orange', linewidth=2)
            
            # Dynamically find the biggest jump/drop in altitude to point an arrow at it
            altitude_diffs = df_raw['gps_alt'].diff().abs()
            if not altitude_diffs.isna().all():
                max_jump_idx = altitude_diffs.idxmax()
                jump_time = df_raw.loc[max_jump_idx, 'time']
                jump_val = df_raw.loc[max_jump_idx, 'gps_alt']
                
            plt.title("Raw GPS Altitude - Spatial Discontinuity", fontsize=14, pad=15)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Altitude (meters)", fontsize=12)
            plt.tight_layout()
            plt.savefig(Path(self.config.reports_dir) / "analysis_broken_gps.png", dpi=300)
            plt.close()

        # ==========================================
        # GRAPH 3: The 0°C Ceiling (Non-Linearity)
        # ==========================================
        if 't_u' in df_raw.columns and 't_surf' in df_raw.columns:
            plt.figure(figsize=(9, 7))
            sns.scatterplot(data=df_raw, x='t_u', y='t_surf', alpha=0.3, color='purple', edgecolor=None)
            
            # The 0-degree plateau line
            plt.axhline(0, color='red', linestyle='--', linewidth=2.5, label='0°C Melting Point')
            
            

            plt.title("Thermodynamic Non-Linearity: Air vs. Surface Temp", fontsize=14, pad=15)
            plt.xlabel("Air Temperature ($T_u$) [°C]", fontsize=12)
            plt.ylabel("Ice Surface Temperature ($T_{surf}$) [°C]", fontsize=12)
            plt.legend(loc='lower right', fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(Path(self.config.reports_dir) / "analysis_melting_plateau.png", dpi=300)
            plt.close()
            
        logger.info(f"Successfully generated research-grade annotated graphs in {self.config.reports_dir}")