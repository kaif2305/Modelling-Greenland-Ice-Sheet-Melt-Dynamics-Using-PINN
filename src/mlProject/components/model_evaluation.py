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

    # ==========================================
    # FIXED: Train vs Validation Loss Curve
    # ==========================================
    def generate_loss_curves(self):
        logger.info("Generating Train vs Validation Loss Curve...")
        sns.set_theme(style="whitegrid")
        
        # FIX: Point to the model_trainer folder where the CSVs are actually saved!
        trainer_dir = Path(self.config.root_dir).parent / "model_trainer"
        loss_files = list(trainer_dir.glob("loss_history_seed_*.csv"))
        
        if not loss_files:
            logger.warning("No loss history CSVs found! Skipping loss plot.")
            return None

        df_loss = pd.read_csv(loss_files[0])
        
        if 'test_loss' in df_loss.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df_loss['epoch'], df_loss['train_loss'], label='Train Loss', color='blue', linewidth=2)
            plt.plot(df_loss['epoch'], df_loss['test_loss'], label='Test (Validation) Loss', color='orange', linewidth=2)
            
            plt.axvline(x=1000, color='red', linestyle='--', label="Physics Curriculum Starts")
            
            plt.title("Model Convergence: Training vs Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (MSE + Physics Penalty)")
            plt.yscale("log") 
            plt.legend()
            
            loss_plot_path = str(Path(self.config.root_dir) / "train_vs_val_loss_curve.png")
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return loss_plot_path
        else:
            return None

    # ==========================================
    # Geographical Error Map
    # ==========================================
    def generate_spatial_map(self, scores):
        logger.info("Attempting to generate Geographical Bubble Map...")
        try:
            import geopandas as gpd
        except ImportError as e:
            logger.error(f"Skipping map generation because geopandas is missing on the HPC: {e}")
            return None 

        station_coords = {
            "KAN_L": (67.095, -50.067), "KAN_U": (67.000, -47.017),
            "QAS_L": (61.030, -46.849), "QAS_U": (61.175, -46.233),
            "TAS_L": (65.640, -38.899), "SCO_L": (72.223, -27.233),
            "THU_L2": (76.399, -68.266)
        }

        data = []
        stations = [s.replace("RMSE_", "") for s in scores.keys() if s.startswith("RMSE_") and "Overall" not in s and "Season" not in s]
        
        for st in stations:
            if st in station_coords:
                lat, lon = station_coords[st]
                data.append({
                    "Station": st, "Latitude": lat, "Longitude": lon,
                    "RMSE": scores.get(f"RMSE_{st}", 0),
                    "PVR": scores.get(f"PVR_{st}", 0)
                })

        df = pd.DataFrame(data)
        
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            greenland = world[world.name == 'Greenland']

            fig, ax = plt.subplots(figsize=(10, 12))
            greenland.plot(ax=ax, color='aliceblue', edgecolor='black', linewidth=1)

            bubble_sizes = df['RMSE'] * 5000  

            scatter = ax.scatter(
                df['Longitude'], df['Latitude'], 
                s=bubble_sizes, c=df['PVR'], cmap='coolwarm', 
                alpha=0.8, edgecolors='black', linewidth=1.5
            )

            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.05)
            cbar.set_label('Physical Violation Rate (PVR %)', fontsize=12)

            for i, row in df.iterrows():
                ax.annotate(row['Station'], (row['Longitude'], row['Latitude']), 
                            xytext=(8, 8), textcoords='offset points', 
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

            plt.title('Spatial Error Map: RMSE (Size) & PVR (Color)', fontsize=16, pad=20)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            map_path = str(Path(self.config.root_dir) / "geographical_error_map.png")
            plt.savefig(map_path, dpi=300, bbox_inches='tight')
            plt.close()
            return map_path
        except Exception as e:
            logger.error(f"Failed to draw the map data: {e}")
            return None

    def generate_thesis_plots(self, df_results, scores):
        logger.info("Generating thesis-ready graphs...")
        sns.set_theme(style="whitegrid")
        stations_list = df_results['Station'].unique()

        plt.figure(figsize=(10, 5))
        rmses = [scores.get(f"RMSE_{s}", 0) for s in stations_list]
        sns.barplot(x=list(stations_list), y=rmses, palette="Blues_d")
        plt.title("Model Accuracy across Ice Sheet Gradients (Spatial Generalization)")
        plt.ylabel("Root Mean Square Error (°C)")
        plt.xlabel("PROMICE Station")
        plt.xticks(rotation=45)
        plt.savefig(Path(self.config.root_dir) / "spatial_rmse_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

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

        plt.figure(figsize=(14, 5))
        sample_station = "KAN_L" if "KAN_L" in stations_list else stations_list[0]
        df_sample = df_results[(df_results['Station'] == sample_station)].sort_values('Date').tail(365) 

        plt.plot(df_sample['Date'], df_sample['Actual'], label="Actual Temp (Sensor)", color="black", alpha=0.7)
        plt.plot(df_sample['Date'], df_sample['Predicted'], label="PINN Ensemble Mean", color="blue")

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

        model_files = list(Path(self.config.model_dir).glob("pinn_model_seed_*.pth"))
        if not model_files:
            logger.error(f"No models found in {self.config.model_dir}! Ensure training completed.")
            return

        logger.info(f"Found {len(model_files)} models in the ensemble. Generating predictions (Batching to prevent memory crash)...")

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
                # FIX: BATCH INFERENCE TO PREVENT SILENT OUT-OF-MEMORY CRASHES
                pred_scaled_list = []
                batch_size = 1024 
                for i in range(0, len(X_tensor), batch_size):
                    batch = X_tensor[i:i+batch_size]
                    batch_preds = base_model(batch)
                    pred_scaled_list.append(batch_preds[:, 0].cpu().numpy().reshape(-1, 1))
                
                pred_scaled = np.vstack(pred_scaled_list)
                pred_unscaled = scaler_y.inverse_transform(pred_scaled).flatten()
                all_predictions.append(pred_unscaled)

        all_predictions = np.array(all_predictions) 
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0) 

        results_df = pd.DataFrame({
            'Actual': actual_t_surf[valid_mask],
            'Predicted': mean_predictions[valid_mask],
            'Uncertainty': std_predictions[valid_mask],
            'Station': seq_st[valid_mask],
            'Date': pd.to_datetime(seq_dt[valid_mask])
        })

        results_df['Month'] = results_df['Date'].dt.month
        results_df['Season'] = np.where(results_df['Month'].isin([5, 6, 7, 8, 9]), 'Summer', 'Winter')

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
        
        self.generate_thesis_plots(results_df, scores)
        map_artifact_path = self.generate_spatial_map(scores)
        loss_curve_path = self.generate_loss_curves()

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_metrics(scores)

            mlflow.log_metric("Mean_Learned_Ch", np.mean(learned_Chs))
            mlflow.log_metric("Std_Learned_Ch", np.std(learned_Chs))
            mlflow.log_metric("Mean_Learned_Csw", np.mean(learned_Csws))
            mlflow.log_metric("Std_Learned_Csw", np.std(learned_Csws))

            save_json(path=Path(self.config.metric_file_name), data=scores)
            
            mlflow.log_artifact(str(Path(self.config.root_dir) / "spatial_rmse_plot.png"))
            mlflow.log_artifact(str(Path(self.config.root_dir) / "actual_vs_predicted.png"))
            mlflow.log_artifact(str(Path(self.config.root_dir) / "ensemble_uncertainty_timeseries.png"))
            
            if map_artifact_path:
                mlflow.log_artifact(map_artifact_path)
            if loss_curve_path:
                mlflow.log_artifact(loss_curve_path)

            model_save_path = str(Path(self.config.root_dir) / "Greenland_Ensemble_PINN.pth")
            torch.save(base_model.state_dict(), model_save_path)
            
            if tracking_url_type_store != "file":
                mlflow.log_artifact(model_save_path, artifact_path="model")
            else:
                mlflow.log_artifact(model_save_path, artifact_path="model")