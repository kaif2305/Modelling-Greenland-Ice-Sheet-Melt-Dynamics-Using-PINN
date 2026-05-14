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
        sns.set_theme(style="whitegrid", context="paper")

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
        # PVR calculation: percentage of predictions physically inconsistent 
        pvr = (np.sum(predicted > 0.05) / len(predicted)) * 100
        return rmse, mae, pvr

    # ==========================================
    # VISUALIZATION METHODS
    # ==========================================

    def generate_loss_curves(self):
        logger.info("Generating individual seed loss curves...")
        trainer_dir = Path(self.config.root_dir).parent / "model_trainer"
        loss_files = list(trainer_dir.glob("loss_history_seed_*.csv"))
        
        paths = []
        for i, file in enumerate(loss_files):
            df_loss = pd.read_csv(file)
            plt.figure(figsize=(10, 6))
            plt.plot(df_loss['epoch'], df_loss['train_loss'], label='Train Loss', color='#1f77b4')
            plt.plot(df_loss['epoch'], df_loss['test_loss'], label='Validation Loss', color='#ff7f0e')
            plt.axvline(x=1000, color='red', linestyle='--', label="Physics Curriculum")
            
            plt.title(f"Loss Convergence - Seed {i+1}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (MSE + Physics Penalty)")
            plt.yscale("log")
            plt.legend()
            
            # Changed to .pdf
            path = Path(self.config.root_dir) / f"loss_curve_seed_{i+1}.pdf"
            plt.savefig(path, format='pdf', bbox_inches='tight')
            plt.close()
            paths.append(path)
        return paths

    def generate_station_map(self):
        logger.info("Generating geographical station map...")
        try:
            import geopandas as gpd
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            greenland = world[world.name == 'Greenland']
            
            station_coords = {
                "KAN_L": (-50.067, 67.095), "KAN_U": (-47.017, 67.000),
                "QAS_L": (-46.849, 61.030), "QAS_U": (-46.233, 61.175),
                "TAS_L": (-38.899, 65.640), "SCO_L": (-27.233, 72.223),
                "THU_L": (-68.266, 76.399), "UPE_U": (-52.703, 72.887),
                "KPC_L": (-19.601, 79.911), "EGP": (-35.500, 75.600)
            }

            fig, ax = plt.subplots(figsize=(8, 12))
            greenland.plot(ax=ax, color='#f1f5f9', edgecolor='#475569')
            
            for name, (lon, lat) in station_coords.items():
                ax.scatter(lon, lat, color='red', s=80, edgecolors='white', zorder=5)
                ax.annotate(name, (lon, lat), xytext=(5, 5), textcoords='offset points', fontsize=8, weight='bold')

            plt.title("Geographical Distribution of PROMICE Stations")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            
            # Changed to .pdf
            path = Path(self.config.root_dir) / "station_location_map.pdf"
            plt.savefig(path, format='pdf', bbox_inches='tight')
            plt.close()
            return path
        except:
            return None

    def generate_thesis_plots(self, df_results, scores):
        logger.info("Generating results visualization suite...")
        
        # A. 10 Individual Station Uncertainty Plots
        for station in df_results['Station'].unique():
            df_st = df_results[df_results['Station'] == station].sort_values('Date').tail(365)
            plt.figure(figsize=(14, 5))
            plt.plot(df_st['Date'], df_st['Actual'], label="Observed", color="black", alpha=0.5)
            plt.plot(df_st['Date'], df_st['Predicted'], label="Ensemble Mean", color="#2563eb")
            plt.fill_between(df_st['Date'], df_st['Predicted']-1.96*df_st['Uncertainty'], 
                             df_st['Predicted']+1.96*df_st['Uncertainty'], color="#2563eb", alpha=0.2, label="95% Confidence Interval")
            plt.axhline(0, color='red', linestyle=':', label="0°C Phase Boundary")
            
            plt.title(f"Melt Season Prediction & Uncertainty: {station}")
            plt.xlabel("Date")
            plt.ylabel("Surface Temperature (°C)")
            plt.legend(loc="best")
            
            # Changed to .pdf
            plt.savefig(Path(self.config.root_dir) / f"uncertainty_{station}.pdf", format='pdf', bbox_inches='tight')
            plt.close()

        # B. Greenland Mean Plot
        df_mean = df_results.groupby('Date').agg({'Actual':'mean', 'Predicted':'mean', 'Uncertainty':'mean'}).reset_index()
        plt.figure(figsize=(14, 6))
        plt.plot(df_mean['Date'], df_mean['Actual'], color="black", label="Domain Actual")
        plt.plot(df_mean['Date'], df_mean['Predicted'], color="#0891b2", label="Domain PINN Mean")
        plt.fill_between(df_mean['Date'], df_mean['Predicted']-df_mean['Uncertainty'], 
                         df_mean['Predicted']+df_mean['Uncertainty'], color="#0891b2", alpha=0.2, label="Uncertainty Bound")
        
        plt.title("Generalized Greenland Ice Sheet Pulse")
        plt.xlabel("Date")
        plt.ylabel("Surface Temperature (°C)")
        plt.legend(loc="best")
        
        # Changed to .pdf
        plt.savefig(Path(self.config.root_dir) / "greenland_mean_timeseries.pdf", format='pdf', bbox_inches='tight')
        plt.close()

        # C. Unit-less Error Bar Chart
        plt.figure(figsize=(10, 5))
        st_names = df_results['Station'].unique()
        rmses = [scores.get(f"RMSE_{s}", 0) for s in st_names]
        sns.barplot(x=list(st_names), y=rmses, palette="viridis")
        
        plt.title("Spatial RMSE Analysis")
        plt.xlabel("Station ID")
        plt.ylabel("Root Mean Square Error") # No unit
        
        # Changed to .pdf
        plt.savefig(Path(self.config.root_dir) / "spatial_rmse_bar.pdf", format='pdf', bbox_inches='tight')
        plt.close()

        # D. Actual vs Predicted Scatter
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df_results, x='Actual', y='Predicted', hue='Season', alpha=0.2)
        plt.plot([-45, 5], [-45, 5], 'k--', label="Ideal Prediction (y=x)")
        
        plt.title("Model Performance: Actual vs. Predicted Temperature")
        plt.xlabel("Actual Surface Temperature (°C)")
        plt.ylabel("Predicted Surface Temperature (°C)")
        plt.legend(title="Season", loc="best")
        
        # Changed to .pdf
        plt.savefig(Path(self.config.root_dir) / "scatter_performance.pdf", format='pdf', bbox_inches='tight')
        plt.close()
        

    # ==========================================
    # MAIN EVALUATION ORCHESTRATOR
    # ==========================================

    def log_into_mlflow(self):
        logger.info("Starting Ensemble Evaluation Orchestrator...")

        # Data Loading
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
        actual_unscaled = scaler_y.inverse_transform(seq_y).flatten()

        # Ensemble Inference
        model_files = list(Path(self.config.model_dir).glob("pinn_model_seed_*.pth"))
        all_preds = []
        learned_params = {'Ch': [], 'Csw': []}

        base_model = AdvancedPINN(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_layers[0],
            output_dim=self.config.output_dim
        ).to(self.device)

        for m_file in model_files:
            base_model.load_state_dict(torch.load(m_file, map_location=self.device, weights_only=True))
            base_model.eval()
            learned_params['Ch'].append(base_model.C_h.item())
            learned_params['Csw'].append(base_model.C_sw.item())

            with torch.no_grad():
                preds_list = []
                for i in range(0, len(X_tensor), 1024):
                    batch_preds = base_model(X_tensor[i:i+1024])
                    preds_list.append(batch_preds[:, 0].cpu().numpy().reshape(-1, 1))
                all_preds.append(scaler_y.inverse_transform(np.vstack(preds_list)).flatten())

        all_preds = np.array(all_preds)
        mean_preds = np.mean(all_preds, axis=0)
        std_preds = np.std(all_preds, axis=0)

        results_df = pd.DataFrame({
            'Actual': actual_unscaled[valid_mask],
            'Predicted': mean_preds[valid_mask],
            'Uncertainty': std_preds[valid_mask],
            'Station': seq_st[valid_mask],
            'Date': pd.to_datetime(seq_dt[valid_mask])
        })
        results_df['Month'] = results_df['Date'].dt.month
        results_df['Season'] = np.where(results_df['Month'].isin([5,6,7,8,9]), 'Summer', 'Winter')

        # Scoring
        o_rmse, o_mae, o_pvr = self.evaluate_metrics(results_df['Actual'], results_df['Predicted'])
        scores = {"Overall_RMSE": o_rmse, "Overall_MAE": o_mae, "Overall_PVR": o_pvr}
        
        for st in results_df['Station'].unique():
            st_df = results_df[results_df['Station'] == st]
            r, m, p = self.evaluate_metrics(st_df['Actual'], st_df['Predicted'])
            scores[f"RMSE_{st}"] = r
            scores[f"PVR_{st}"] = p

        # Plotting
        self.generate_thesis_plots(results_df, scores)
        map_p = self.generate_station_map()
        loss_ps = self.generate_loss_curves()

        # MLflow Logging
        mlflow.set_registry_uri(self.config.mlflow_uri)
        with mlflow.start_run():
            mlflow.log_metrics(scores)
            mlflow.log_metric("Mean_Ch", np.mean(learned_params['Ch']))
            mlflow.log_metric("Mean_Csw", np.mean(learned_params['Csw']))
            
            save_json(path=Path(self.config.metric_file_name), data=scores)
            
            # Changed the glob search to find *.pdf so MLflow grabs the newly formatted plots
            for img in Path(self.config.root_dir).glob("*.pdf"):
                mlflow.log_artifact(str(img))
            
            model_path = Path(self.config.root_dir) / "Greenland_Ensemble_PINN.pth"
            torch.save(base_model.state_dict(), model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")

        logger.info("Evaluation and Logging Successful.")