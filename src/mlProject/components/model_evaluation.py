import pandas as pd
import numpy as np
import torch
import joblib
import mlflow
import mlflow.pytorch
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from mlProject import logger
from mlProject.utils.common import save_json
from mlProject.components.model_training import AdvancedPINN

class ModelEvaluation:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_sequences(self, data, targets, masks, stations, dates, seq_length):
        xs, ys, ms, st, dt = [], [], [], [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:(i + seq_length)])
            ys.append(targets[i + seq_length - 1]) 
            ms.append(masks[i + seq_length - 1])
            st.append(stations[i + seq_length - 1])
            dt.append(dates[i + seq_length - 1])
        return np.array(xs), np.array(ys), np.array(ms), np.array(st), np.array(dt)

    def evaluate_metrics(self, actual, predicted):
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        pvr = (np.sum(predicted > 0.05) / len(predicted)) * 100 
        return rmse, mae, pvr

    def generate_thesis_plots(self, df_results, scores):
        logger.info("Generating thesis-ready graphs...")
        sns.set_theme(style="whitegrid")
        stations_list = df_results['Station'].unique()
        
        # Plot 1: Spatial Performance (RMSE by Station)
        plt.figure(figsize=(10, 5))
        rmses = [scores.get(f"RMSE_{s}", 0) for s in stations_list]
        sns.barplot(x=list(stations_list), y=rmses, palette="Blues_d")
        plt.title("Model Accuracy across Ice Sheet Gradients (Spatial Generalization)")
        plt.ylabel("Root Mean Square Error (°C)")
        plt.xlabel("PROMICE Station")
        plt.xticks(rotation=45)
        plt.savefig(Path(self.config.root_dir) / "spatial_rmse_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Actual vs Predicted Scatter
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df_results, x='Actual', y='Predicted', hue='Season', alpha=0.3, palette={"Summer": "red", "Winter": "blue"})
        plt.plot([-40, 5], [-40, 5], color='black', linestyle='--') 
        plt.axhline(0, color='red', linestyle=':', label="0°C Physical Boundary")
        plt.title("Ensemble Mean Prediction: Actual vs. Predicted Temperature")
        plt.xlabel("Actual Surface Temperature (°C)")
        plt.ylabel("Predicted Surface Temperature (°C)")
        plt.legend()
        plt.savefig(Path(self.config.root_dir) / "actual_vs_predicted.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 3: The "Money Shot" - Time Series with Uncertainty Band
        plt.figure(figsize=(14, 5))
        sample_station = "KAN_L" if "KAN_L" in stations_list else stations_list[0]
        df_sample = df_results[(df_results['Station'] == sample_station)].sort_values('Date').tail(365) # Show last 1 year
        
        plt.plot(df_sample['Date'], df_sample['Actual'], label="Actual Temp (Sensor)", color="black", alpha=0.7)
        plt.plot(df_sample['Date'], df_sample['Predicted'], label="PINN Ensemble Mean", color="blue")
        
        # Draw the Shaded Confidence Interval
        plt.fill_between(df_sample['Date'],
                         df_sample['Predicted'] - (1.96 * df_sample['Uncertainty']),
                         df_sample['Predicted'] + (1.96 * df_sample['Uncertainty']),
                         color="blue", alpha=0.2, label="95% Confidence Interval")
        
        plt.axhline(0, color='red', linestyle=':', label="0°C Melt Threshold")
        plt.title(f"Thermodynamic Prediction with Uncertainty Bounds ({sample_station})")
        plt.ylabel("Surface Temperature (°C)")
        plt.legend(loc='lower right')
        plt.savefig(Path(self.config.root_dir) / "ensemble_uncertainty_timeseries.png", dpi=300, bbox_inches='tight')
        plt.close()

    def log_into_mlflow(self):
        logger.info("Starting Ensemble Evaluation and MLflow logging...")
        
        # 1. Load Data & Scalers
        df = pd.read_csv(self.config.test_data_path)
        df['time'] = pd.to_datetime(df['time'])
        df[self.config.collocation_flag] = df[self.config.target_feature].isna()
        df[self.config.target_feature] = df[self.config.target_feature].fillna(0.0)

        X_raw = df[self.config.input_features].values
        y_raw = df[self.config.target_feature].values.reshape(-1, 1)
        mask_raw = (~df[self.config.collocation_flag]).values.reshape(-1, 1)
        station_raw = df['station_name'].values
        dates_raw = df['time'].values

        scaler_X = joblib.load(self.config.scaler_X_path)
        scaler_y = joblib.load(self.config.scaler_y_path)
        X_scaled = scaler_X.transform(X_raw)
        y_scaled = scaler_y.transform(y_raw)

        seq_X, seq_y, seq_mask, seq_st, seq_dt = self.create_sequences(
            X_scaled, y_scaled, mask_raw, station_raw, dates_raw, self.config.seq_length
        )

        X_tensor = torch.tensor(seq_X, dtype=torch.float32).to(self.device)
        valid_mask = seq_mask.flatten()
        actual_t_surf = scaler_y.inverse_transform(seq_y).flatten()

        # 2. Find and Evaluate ALL Ensemble Models
        model_files = list(Path(self.config.model_dir).glob("pinn_model_seed_*.pth"))
        if not model_files:
            logger.error(f"No models found in {self.config.model_dir}! Ensure training completed.")
            return

        logger.info(f"Found {len(model_files)} models in the ensemble. Generating predictions...")
        
        all_predictions = []
        learned_Chs = []
        learned_Csws = []

        base_model = AdvancedPINN(
            input_dim=self.config.input_dim, 
            hidden_dim=self.config.hidden_layers[0], 
            output_dim=self.config.output_dim
        ).to(self.device)

        for m_file in model_files:
            base_model.load_state_dict(torch.load(m_file, map_location=self.device))
            base_model.eval()
            learned_Chs.append(base_model.C_h.item())
            learned_Csws.append(base_model.C_sw.item())

            with torch.no_grad():
                preds = base_model(X_tensor)
                pred_scaled = preds[:, 0].cpu().numpy().reshape(-1, 1)
                pred_unscaled = scaler_y.inverse_transform(pred_scaled).flatten()
                all_predictions.append(pred_unscaled)

        # 3. Calculate Ensemble Math (Mean & Standard Deviation)
        all_predictions = np.array(all_predictions) # Shape: (5_models, total_days)
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0) # This is the Uncertainty!

        # Build Results DataFrame
        results_df = pd.DataFrame({
            'Actual': actual_t_surf[valid_mask],
            'Predicted': mean_predictions[valid_mask],
            'Uncertainty': std_predictions[valid_mask],
            'Station': seq_st[valid_mask],
            'Date': pd.to_datetime(seq_dt[valid_mask])
        })
        
        results_df['Month'] = results_df['Date'].dt.month
        results_df['Season'] = np.where(results_df['Month'].isin([5, 6, 7, 8, 9]), 'Summer', 'Winter')

        # 4. Calculate Master Metrics based on the MEAN prediction
        overall_rmse, overall_mae, overall_pvr = self.evaluate_metrics(results_df['Actual'], results_df['Predicted'])
        scores = {"Overall_RMSE": overall_rmse, "Overall_MAE": overall_mae, "Overall_PVR": overall_pvr}

        stations_list = results_df['Station'].unique()
        for station in stations_list:
            st_data = results_df[results_df['Station'] == station]
            if not st_data.empty:
                r, m, p = self.evaluate_metrics(st_data['Actual'], st_data['Predicted'])
                scores[f"RMSE_{station}"] = r
                scores[f"PVR_{station}"] = p

        for season in ['Summer', 'Winter']:
            sz_data = results_df[results_df['Season'] == season]
            if not sz_data.empty:
                r, m, p = self.evaluate_metrics(sz_data['Actual'], sz_data['Predicted'])
                scores[f"RMSE_{season}"] = r
                scores[f"PVR_{season}"] = p

        logger.info(f"Ensemble Evaluation Complete: Mean RMSE={overall_rmse:.4f}, Mean PVR={overall_pvr:.2f}%")

        # 5. Generate Graphs
        self.generate_thesis_plots(results_df, scores)

        # 6. Log to MLflow
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_metrics(scores)
            
            # Log the parameter stability (Average learned physics)
            mlflow.log_metric("Mean_Learned_Ch", np.mean(learned_Chs))
            mlflow.log_metric("Std_Learned_Ch", np.std(learned_Chs))
            mlflow.log_metric("Mean_Learned_Csw", np.mean(learned_Csws))
            mlflow.log_metric("Std_Learned_Csw", np.std(learned_Csws))

            save_json(path=Path(self.config.metric_file_name), data=scores)
            mlflow.log_artifact(str(Path(self.config.root_dir) / "spatial_rmse_plot.png"))
            mlflow.log_artifact(str(Path(self.config.root_dir) / "actual_vs_predicted.png"))
            mlflow.log_artifact(str(Path(self.config.root_dir) / "ensemble_uncertainty_timeseries.png"))

            # Safely save and upload the raw model weights to avoid PyTorch version crashes
            model_save_path = str(Path(self.config.root_dir) / "Greenland_Ensemble_PINN.pth")
            torch.save(base_model.state_dict(), model_save_path)
            
            if tracking_url_type_store != "file":
                mlflow.log_artifact(model_save_path, artifact_path="model")
            else:
                mlflow.log_artifact(model_save_path, artifact_path="model")