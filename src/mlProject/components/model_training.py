import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

from mlProject import logger
# from mlProject.entity.config_entity import ModelTrainerConfig

# ==========================================
# 1. Advanced LSTM-PINN Architecture
# ==========================================
class AdvancedPINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AdvancedPINN, self).__init__()
        
        # LSTM for Thermal Memory (Temporal Inertia)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Fully connected output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim) # Outputs: [t_surf, melt_rate]
        )
        
        # INVERSE MODELING (Learnable Physics Constants)
        self.C_h = nn.Parameter(torch.tensor([0.01]))  # Sensible Heat Transfer Coefficient
        self.C_sw = nn.Parameter(torch.tensor([1.0]))  # Shortwave Radiation Multiplier
        self.C_ice = nn.Parameter(torch.tensor([0.5])) # Ice Thermal Mass capability
        
        # DYNAMIC LOSS WEIGHTING (Solves Gradient Pathology)
        self.log_var_data = nn.Parameter(torch.zeros(1))
        self.log_var_phys = nn.Parameter(torch.zeros(1))
        self.log_var_bound = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # We only care about the prediction for the final day in the sequence
        last_day_features = lstm_out[:, -1, :] 
        return self.fc(last_day_features)

# ==========================================
# 2. Model Trainer Component
# ==========================================
class ModelTrainer:
    def __init__(self, config): # Type hint removed temporarily for standalone execution
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Advanced LSTM-PINN Trainer initialized on: {self.device}")

    def create_sequences(self, data, targets, masks, seq_length):
        """
        Converts flat time-series into overlapping windows for the LSTM.
        e.g., 7 days of weather to predict the 7th day's surface temperature.
        """
        xs, ys, ms = [], [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:(i + seq_length)])
            ys.append(targets[i + seq_length - 1]) 
            ms.append(masks[i + seq_length - 1])
        return np.array(xs), np.array(ys), np.array(ms)

    def prepare_data(self):
        """
        Loads the transformed dataset, scales it, creates sequences, 
        and prepares PyTorch DataLoaders.
        """
        logger.info("Step 1: Loading and Sequencing master dataset...")
        df = pd.read_csv(self.config.clean_data_path)
        
        # Define collocation points mask
        df[self.config.collocation_flag] = df[self.config.target_feature].isna()
        df[self.config.target_feature] = df[self.config.target_feature].fillna(0.0)

        # Extract arrays
        X_raw = df[self.config.input_features].values
        y_raw = df[self.config.target_feature].values.reshape(-1, 1)
        mask_raw = (~df[self.config.collocation_flag]).values.reshape(-1, 1)
        
        # Scale to [-1, 1] for stable gradients
        self.scaler_X = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = self.scaler_X.fit_transform(X_raw)
        y_scaled = self.scaler_y.fit_transform(y_raw)
        
        # Create LSTM Sequences
        seq_X, seq_y, seq_mask = self.create_sequences(
            X_scaled, y_scaled, mask_raw, self.config.seq_length
        )
        
        # Convert to PyTorch Tensors
        self.X_tensor = torch.tensor(seq_X, dtype=torch.float32).to(self.device)
        self.y_tensor = torch.tensor(seq_y, dtype=torch.float32).to(self.device)
        self.mask_tensor = torch.tensor(seq_mask, dtype=torch.bool).to(self.device)
        
        dataset = TensorDataset(self.X_tensor, self.y_tensor, self.mask_tensor)
        self.dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        logger.info(f"Sequencing complete. Total LSTM windows: {len(seq_X)}")

    def train(self):
        """
        Main orchestration method:
        Runs the Physics-Informed training loop using Explicit Surface Energy Balance.
        """
        self.prepare_data()
        
        model = AdvancedPINN(
            input_dim=self.config.input_dim, 
            hidden_dim=self.config.hidden_layers[0], 
            output_dim=self.config.output_dim
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # Map feature column indices dynamically from config
        IDX_SIN = self.config.input_features.index('day_sin')
        IDX_COS = self.config.input_features.index('day_cos')
        IDX_TU = self.config.input_features.index('t_u')
        IDX_WSPD = self.config.input_features.index('wspd_u')
        IDX_ALB = self.config.input_features.index('albedo')

        logger.info("Step 2: Starting Advanced Surface Energy Balance PINN Training...")
        loss_history = []
        
        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0
            
            for batch_X, batch_y, batch_mask in self.dataloader:
                optimizer.zero_grad()
                batch_X.requires_grad_(True)
                
                # Forward Pass
                preds = model(batch_X)
                pred_t_surf = preds[:, 0].unsqueeze(1)
                pred_melt = preds[:, 1].unsqueeze(1)
                
                current_weather = batch_X[:, -1, :] # Last day in sequence
                
                # --- A. DATA LOSS ---
                valid_preds = torch.masked_select(pred_t_surf, batch_mask)
                valid_targets = torch.masked_select(batch_y, batch_mask)
                loss_data = nn.MSELoss()(valid_preds, valid_targets)
                
                # --- B. PHYSICS LOSS (SEB) ---
                solar_proxy = torch.relu(current_weather[:, IDX_SIN] + current_weather[:, IDX_COS])
                net_sw = solar_proxy * (1.0 - current_weather[:, IDX_ALB]) * model.C_sw
                sensible_heat = model.C_h * current_weather[:, IDX_WSPD] * (current_weather[:, IDX_TU] - pred_t_surf.squeeze())
                total_seb = net_sw + sensible_heat
                
                dT_dt = torch.autograd.grad(
                    outputs=pred_t_surf, inputs=batch_X, 
                    grad_outputs=torch.ones_like(pred_t_surf), create_graph=True
                )[0][:, -1, IDX_TU] 
                
                physics_residual = dT_dt - (total_seb * model.C_ice)
                loss_physics = torch.mean(physics_residual ** 2)

                # --- C. BOUNDARY LOSS ---
                pred_unscaled = pred_t_surf * (self.scaler_y.data_max_[0] - self.scaler_y.data_min_[0]) / 2.0 + (self.scaler_y.data_max_[0] + self.scaler_y.data_min_[0]) / 2.0
                loss_boundary = torch.mean(torch.relu(pred_unscaled - 0.0))
                
                frozen_mask = (pred_unscaled < -0.1).squeeze()
                loss_melt_logic = torch.mean((pred_melt.squeeze()[frozen_mask]) ** 2) if frozen_mask.any() else torch.tensor(0.0).to(self.device)

                # --- D. DYNAMIC WEIGHTING & BACKPROP ---
                loss = (torch.exp(-model.log_var_data) * loss_data + model.log_var_data) + \
                       (torch.exp(-model.log_var_phys) * loss_physics + model.log_var_phys) + \
                       (torch.exp(-model.log_var_bound) * (loss_boundary + loss_melt_logic) + model.log_var_bound)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            if (epoch + 1) % 50 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1:04d}/{self.config.epochs} | Total Loss: {epoch_loss:.4f} | "
                            f"Learned C_h: {model.C_h.item():.4f} | Learned C_sw: {model.C_sw.item():.4f}")
                
            loss_history.append({"epoch": epoch + 1, "loss": epoch_loss})

        # --- STEP 3: SAVE MODEL & LOGS ---
        logger.info("Step 3: Training complete. Saving artifacts...")
        torch.save(model.state_dict(), self.config.trained_model_path)
        pd.DataFrame(loss_history).to_csv(self.config.loss_log_path, index=False)
        
        joblib.dump(self.scaler_X, Path(self.config.root_dir) / "scaler_X.joblib")
        joblib.dump(self.scaler_y, Path(self.config.root_dir) / "scaler_y.joblib")
        
        logger.info(f"Model and scalers successfully saved to: {self.config.root_dir}")