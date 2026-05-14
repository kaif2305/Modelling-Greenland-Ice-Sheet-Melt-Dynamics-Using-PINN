# PINNs on Ice: Modelling Surface Melt Dynamics in the Greenland Ice Sheet using Physics-Informed Neural Networks

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)
![Status](https://img.shields.io/badge/Status-Complete-success)

**Master's Degree Thesis in AI for Health | Spring Term 2026** **Authors:** Kaif Ahmad and Raghunandan Rajkumar  
**Supervisor:** Victoria Oberländer  
**Institution:** Department of Computers and Systems Sciences, Stockholm University  

---

## 🌍 Abstract & Project Overview
The Greenland Ice Sheet (GrIS) is currently undergoing accelerating mass loss, predominantly driven by surface meltwater production, representing a critical contributor to global sea-level rise. High-resolution numerical methods can resolve complex glacier dynamics, but are highly limited in scale and computationally expensive. Purely data-driven machine learning models are not as limited by scale, but fail to adhere to the laws of physics, resulting in poor generalization out of distribution and unphysical predictions. 

This research addresses this limitation by investigating the integration of thermodynamic partial differential equation (PDE) constraints within a Physics-Informed Neural Network (PINN) framework wrapped around a Long Short-Term Memory (LSTM) backbone. This hybrid **LSTM-PINN** architecture achieves both computational efficiency and physical validity, providing a highly scalable tool for glaciological forecasting.

---

## 🏔️ Background & The Scientific Challenge
Surface melt in an ice sheet is fundamentally dictated by the **Surface Energy Balance (SEB)**, which accounts for fluxes in radiation, sensible heat, and latent heat exchange:

`Q_M = SW_in(1 - α) + ΔLW + Q_S + Q_L`

A key thermodynamic boundary occurs at $0^{\circ}C$. At this point, surplus energy ceases to raise the ice temperature and is instead consumed as latent heat of fusion for a change in phase (melting). 

**The Limitation of Standard AI:** Traditional deep learning architectures function as unconstrained "black boxes" that routinely fail to recognize this physical ceiling during peak summer atmospheric forcing, predicting unphysical ice temperatures far above the melting point. While standard PINNs have been used to enforce mass and momentum conservation, the explicit coupling of SEB thermodynamics within a composite loss landscape remains largely unexplored.

---

## 📊 Data Architecture & Spatial Strategy
To ensure the model learns a generalized physical representation of Greenland rather than memorizing a localized climate, we utilized 10 years of historical data (2014–2024) sourced from 10 Automated Weather Stations (AWS) operated by the **Programme for Monitoring of the Greenland Ice Sheet (PROMICE)**.

* **Stations Evaluated:** KAN_L, KAN_U, QAS_L, QAS_U, TAS_L, SCO_L, THU_L, UPE_U, KPC_L, EGP.
* **Geographic Stratification:** This distribution captures the massive latitudinal shift from the active southern margin to the high Arctic, as well as the elevational lapse rate from the ablation zone up to the stable, frozen interior.
* **Feature Set:** Inputs include Air Temperature, Wind Speed, Albedo, Shortwave Radiation, and Sensible Heat Flux. Target variables are Surface Temperature ($T_{surf}$) and Melt Rate.
* **Temporal Context:** Data was transformed into **14-day rolling windows** to capture the thermal inertia of the ice sheet. We used a strict chronological split (80% Train / 20% Test) to prevent data leakage.

---

## 🧠 Model Architecture: The LSTM-PINN
The core model is implemented in `PyTorch` and consists of three main components:

1. **The Temporal Backbone:** A stacked 2-layer LSTM with 20% Dropout processes the 14-day weather sequences, accounting for cumulative energy history.
2. **The Prediction Head:** A 64-neuron Dense layer with a `Tanh` activation function.
3. **The Physics Layer (Inverse Modeling):** Solves for physical constants (e.g., Sensible Heat Transfer $C_h$, Solar Multiplier $C_{sw}$) as learnable parameters during backpropagation.

### The Composite Loss Equation
The network is optimized using a custom multi-objective loss function that penalizes both data misfits and thermodynamic violations:

`Loss_Total = MSE(Data) + λ_phys(SEB_Residual) + λ_bound(Soft_Margin_Penalty)`

The inclusion of a **Soft-Margin boundary constraint** dynamically enforces the $0^{\circ}C$ phase-change threshold while allowing realistic 2m-air temperatures up to $4.0^{\circ}C$, eliminating the unphysical temperature overshoots characteristic of unconstrained models.

---

## 📈 Evaluation Strategy & Key Results
Evaluation was designed to prove the model learned actual science, not just statistical correlations. We utilized a **5-seed ensemble** to quantify predictive uncertainty (95% Confidence Intervals) alongside a custom **Physical Violation Rate (PVR)** metric.

* **Predictive Accuracy:** The LSTM-PINN achieved an overall Root Mean Square Error (RMSE) of **$1.11^{\circ}C$** while maintaining high spatial generalization ($R^{2} > 0.95$).
* **Elimination of the "Flatline":** The architecture successfully learned the $0^{\circ}C$ phase-change threshold, fully eliminating unphysical surface temperature spikes.
* **The Spatial Dichotomy:** Active southern ablation zones exhibited exceptional statistical precision but slightly elevated physical stress at the melting boundary. Conversely, stable northern/interior regimes demonstrated near-perfect physical adherence.
* **Telemetry Blackout Robustness:** During instrumental data gaps, the model successfully transitioned into a "purely physical emulation mode," widening its predictive confidence envelope while maintaining continuous SEB closure.

---
