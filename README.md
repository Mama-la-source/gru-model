# PrognosAI – Remaining Useful Life Prediction

PrognosAI is an end-to-end **predictive maintenance** project that predicts the **Remaining Useful Life (RUL)** of aircraft engines using **GRU-based deep learning** on multivariate time-series sensor data.

---

## 🔍 Overview

The goal of this project is to estimate how many operational cycles an engine has left before failure. It uses historical sensor data to learn degradation patterns and converts predictions into **actionable alert levels** for maintenance decisions.

---

## 📊 Dataset

* **NASA CMAPSS** benchmark dataset
* Multivariate time-series data
* 21 sensor readings per cycle
* Multiple operating conditions (FD001–FD004)

---

## 🧠 Model

* **Architecture:** GRU-based neural network
* **Input:** Sliding windows of 30 cycles × 21 sensors
* **Output:** Predicted Remaining Useful Life (RUL)
* **Why GRU:** Captures temporal dependencies with lower complexity than LSTM

**Evaluation Metrics:** RMSE, MAE

---

## 🛠️ Project Structure

```
PrognosAI/
├── data/                    # NASA CMAPSS (download separately)
├── models/                  # Trained GRU weights
│   ├── fd001_gru_model.weights.h5
│   ├── fd002_gru_model.weights.h5
│   ├── fd003_gru_model.weights.h5
│   └── fd004_gru_model.weights.h5
├── outputs/                 # Evaluation results + plots
│   ├── fd00[1-4]_sequences.npz
│   ├── fd00[1-4]_evaluation.npz  
│   ├── fd00[1-4]_alerts.npz
│   └── fd00[1-4]_evaluation_plots.png
└── dashboard/             
    └── prognosai_master_dashboard.py  ← MAIN APP


---

## 📈 Dashboard Features

* Dataset-level performance summary (RMSE, MAE)
* Engine-wise RUL prediction and alert status
* Single engine sensor analysis
* Actual vs predicted RUL comparison
* CSV report download

**Alert Levels:** VERY SAFE, SAFE, WARNING, CRITICAL

---

## ▶️ Run the Project

```bash
pip install streamlit numpy pandas matplotlib tensorflow
streamlit run prognosai_master_dashboard.py 
```

---

## 👤 Author

**Mamatha Gaje**
