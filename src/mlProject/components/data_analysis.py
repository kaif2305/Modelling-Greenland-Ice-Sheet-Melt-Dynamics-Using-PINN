import pandas as pd
import json
from pathlib import Path
from mlProject import logger
from mlProject.entity.config_entity import DataAnalysisConfig

class DataAnalysis:
    def __init__(self, config: DataAnalysisConfig):
        self.config = config

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
                
                # 1. Filter for variables that exist strictly as columns
                existing_cols = [var for var in self.config.target_variables if var in df.columns]
                
                # 2. Identify variables that are missing from BOTH columns AND the index
                missing_vars = [
                    var for var in self.config.target_variables 
                    if var not in df.columns and var != df.index.name and var != "time"
                ]
                
                if missing_vars:
                    logger.warning(f"Station {station_name} is completely missing schema columns: {missing_vars}")

                # 3. Calculate missing values for the actual columns
                null_counts = df[existing_cols].isnull().sum().to_dict()
                
                # 4. Safely calculate missing values for the index (time) if required by schema
                if df.index.name in self.config.target_variables or "time" in self.config.target_variables:
                    null_counts['time'] = int(df.index.isnull().sum())
                    
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