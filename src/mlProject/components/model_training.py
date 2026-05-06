import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

from mlProject import logger

# ==========================================
# 1. Advanced LSTM-PINN Architecture
# ==========================================
class AdvancedPINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AdvancedPINN, self).__init__()

        # Added dropout=0.2 to LSTM to prevent memorization of weather patterns
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)

        # Fully connected output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Dropout(0.2),  # Added Dropout to the dense layer
            nn.Linear(64, output_dim) # Outputs: [t_surf, melt_rate]
        )

        # INVERSE MODELING (Learnable Physics Constants)
        self.C_h = nn.Parameter(torch.tensor([0.01]))  # Sensible Heat Transfer Coefficient
        self.C_sw = nn.Parameter(torch.tensor([1.0]))  # Shortwave Radiation Multiplier
        self.C_ice = nn.Parameter(torch.tensor([0.5])) # Ice Thermal Mass capability

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
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Advanced LSTM-PINN Trainer initialized on: {self.device}")

    def create_sequences(self, data, targets, masks, seq_length):
        xs, ys, ms = [], [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:(i + seq_length)])
            ys.append(targets[i + seq_length - 1])
            ms.append(masks[i + seq_length - 1])
        return np.array(xs), np.array(ys), np.array(ms)

    def prepare_data(self):
        logger.info("Step 1: Loading and splitting master dataset into Train/Test...")
        
        # Load the single master dataset
        df = pd.read_csv(self.config.clean_data_path)

        # SPLIT THE DATA (80% Train, 20% Test)
        # We split chronologically because it is time-series weather data
        split_idx = int(len(df) * 0.8)
        df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
        df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

        logger.info(f"Training on {len(df_train)} rows, Testing on {len(df_test)} rows.")

        # --- PROCESS TRAIN DATA ---
        df_train[self.config.collocation_flag] = df_train[self.config.target_feature].isna()
        df_train[self.config.target_feature] = df_train[self.config.target_feature].fillna(0.0)
        X_train_raw = df_train[self.config.input_features].values
        y_train_raw = df_train[self.config.target_feature].values.reshape(-1, 1)
        mask_train_raw = (~df_train[self.config.collocation_flag]).values.reshape(-1, 1)

        # Fit Scalers ONLY on training data (Zero Data Leakage!)
        self.scaler_X = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = self.scaler_X.fit_transform(X_train_raw)
        y_train_scaled = self.scaler_y.fit_transform(y_train_raw)

        seq_X_train, seq_y_train, seq_mask_train = self.create_sequences(
            X_train_scaled, y_train_scaled, mask_train_raw, self.config.seq_length
        )

        self.train_dataloader = DataLoader(
            TensorDataset(torch.tensor(seq_X_train, dtype=torch.float32), 
                          torch.tensor(seq_y_train, dtype=torch.float32), 
                          torch.tensor(seq_mask_train, dtype=torch.bool)),
            batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        # --- PROCESS TEST DATA ---
        df_test[self.config.collocation_flag] = df_test[self.config.target_feature].isna()
        df_test[self.config.target_feature] = df_test[self.config.target_feature].fillna(0.0)
        X_test_raw = df_test[self.config.input_features].values
        y_test_raw = df_test[self.config.target_feature].values.reshape(-1, 1)
        mask_test_raw = (~df_test[self.config.collocation_flag]).values.reshape(-1, 1)

        # Transform Test Data (DO NOT FIT SCALERS ON TEST DATA)
        X_test_scaled = self.scaler_X.transform(X_test_raw)
        y_test_scaled = self.scaler_y.transform(y_test_raw)

        seq_X_test, seq_y_test, seq_mask_test = self.create_sequences(
            X_test_scaled, y_test_scaled, mask_test_raw, self.config.seq_length
        )

        self.test_dataloader = DataLoader(
            TensorDataset(torch.tensor(seq_X_test, dtype=torch.float32), 
                          torch.tensor(seq_y_test, dtype=torch.float32), 
                          torch.tensor(seq_mask_test, dtype=torch.bool)),
            batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

    def train(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.prepare_data()

        model = AdvancedPINN(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_layers[0],
            output_dim=self.config.output_dim
        ).to(self.device)

        # Added weight_decay (L2 regularization) to penalize over-reliance on single features
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)

        IDX_SIN = self.config.input_features.index('day_sin')
        IDX_COS = self.config.input_features.index('day_cos')
        IDX_TU = self.config.input_features.index('t_u')
        IDX_WSPD = self.config.input_features.index('wspd_u')
        IDX_ALB = self.config.input_features.index('albedo')

        loss_history = []
        for epoch in range(self.config.epochs):
            model.train() # Make sure model is in training mode for Dropout
            epoch_loss = 0

            for batch_X, batch_y, batch_mask in self.train_dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_mask = batch_mask.to(self.device)

                optimizer.zero_grad()
                batch_X.requires_grad_(True)
                
                with torch.backends.cudnn.flags(enabled=False):
                    preds = model(batch_X)

                pred_t_surf = preds[:, 0].unsqueeze(1)
                pred_melt = preds[:, 1].unsqueeze(1)
                current_weather = batch_X[:, -1, :]

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
                
                # FIXED: Reduced diffusion multiplier to 0.5 to prevent extreme winter damping
                loss_physics = 0.5 * torch.mean(physics_residual ** 2)
                
                # --- C. BOUNDARY LOSS (Unscaling to Celsius) ---
                y_max = self.scaler_y.data_max_[0]
                y_min = self.scaler_y.data_min_[0]
                pred_unscaled = pred_t_surf * (y_max - y_min) / 2.0 + (y_max + y_min) / 2.0
                
                # FIXED: Added a 4.0 degree "soft margin" since the sensor reads 2m air temp, 
                # which can naturally exceed 0°C during summer without melting the physical ice below.
                air_temp_margin = 4.0
                loss_boundary = torch.mean(torch.relu(pred_unscaled - air_temp_margin))

                frozen_mask = (pred_unscaled < -0.1).squeeze()
                loss_melt_logic = torch.mean((pred_melt.squeeze()[frozen_mask]) ** 2) if frozen_mask.any() else torch.tensor(0.0).to(self.device)

                # --- D. CURRICULUM LEARNING LOGIC ---
                if epoch < 1000:
                    phys_weight = 0.0
                else:
                    phys_weight = min(0.1, (epoch - 1000) * 0.0001)

                loss = loss_data + (phys_weight * loss_physics) + (phys_weight * (loss_boundary + loss_melt_logic))

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # --- EVALUATE ON TEST SET AT END OF EPOCH ---
            model.eval() # Turn off Dropout to get accurate test predictions
            test_loss = 0.0
            with torch.no_grad(): # Don't calculate gradients for testing
                for test_X, test_y, test_mask in self.test_dataloader:
                    test_X = test_X.to(self.device)
                    test_y = test_y.to(self.device)
                    test_mask = test_mask.to(self.device)
                    
                    test_preds = model(test_X)
                    test_pred_t_surf = test_preds[:, 0].unsqueeze(1)
                    
                    valid_test_preds = torch.masked_select(test_pred_t_surf, test_mask)
                    valid_test_targets = torch.masked_select(test_y, test_mask)
                    
                    if len(valid_test_preds) > 0:
                        batch_test_loss = nn.MSELoss()(valid_test_preds, valid_test_targets)
                        test_loss += batch_test_loss.item()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                logger.info(f"Seed {seed} | Epoch {epoch+1:04d}/{self.config.epochs} | Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Phys Weight: {phys_weight:.4f}")

            # Save BOTH Train and Test loss
            loss_history.append({
                "epoch": epoch + 1, 
                "train_loss": epoch_loss,
                "test_loss": test_loss
            })
            
        # --- STEP 3: SAVE MODEL ---
        save_name = f"pinn_model_seed_{seed}.pth"
        torch.save(model.state_dict(), Path(self.config.root_dir) / save_name)

        pd.DataFrame(loss_history).to_csv(Path(self.config.root_dir) / f"loss_history_seed_{seed}.csv", index=False)
        joblib.dump(self.scaler_X, Path(self.config.root_dir) / "scaler_X.joblib")
        joblib.dump(self.scaler_y, Path(self.config.root_dir) / "scaler_y.joblib")

        logger.info(f"Ensemble model saved to: {Path(self.config.root_dir) / save_name}")