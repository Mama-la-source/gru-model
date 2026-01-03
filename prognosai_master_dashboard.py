import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title="PrognosAI Master Dashboard", layout="wide")

# Load ALL 4 datasets
@st.cache_data
def load_all_datasets():
    datasets = {}
    for fd in ['fd001', 'fd002', 'fd003', 'fd004']:
        seq_data = np.load(f'{fd}_sequences.npz')
        eval_data = np.load(f'{fd}_evaluation.npz')
        datasets[fd] = {
            'X': seq_data['X'], 
            'y_test': eval_data['y_test'],
            'y_test_pred': eval_data['y_test_pred'],
            'rmse': np.sqrt(np.mean((eval_data['y_test_pred'] - eval_data['y_test'])**2)),
            'mae': np.mean(np.abs(eval_data['y_test_pred'] - eval_data['y_test'])),
            'n_samples': len(eval_data['y_test'])
        }
    return datasets

datasets = load_all_datasets()

# Load models
@st.cache_resource
def load_all_models():
    models = {}
    for fd in ['fd001', 'fd002', 'fd003', 'fd004']:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(30, 21)),
            tf.keras.layers.GRU(50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        model.load_weights(f'{fd}_gru_model.weights.h5')
        models[fd] = model
    return models

models = load_all_models()

def classify_alert(rul):
    if rul < 10: return 'CRITICAL'
    elif rul < 30: return 'WARNING'
    elif rul < 125: return 'SAFE'
    return 'VERY_SAFE'

# === MASTER DASHBOARD ===
st.title("🚀 PrognosAI - NASA CMAPSS Master Dashboard")
st.markdown("**ALL 4 Datasets | GRU Models | Live Predictions | Cross-Comparison**")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "FD001", "FD002", "FD003", "FD004"])

# Tab 1: Overview
with tab1:
    st.header("📈 Cross-Dataset Performance Comparison")
    
    # RMSE Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # RMSE comparison
    rmse_data = {fd: datasets[fd]['rmse'] for fd in datasets}
    engines = {fd: datasets[fd]['n_samples'] for fd in datasets}
    
    ax1.bar(rmse_data.keys(), rmse_data.values(), color=['green', 'blue', 'orange', 'red'])
    ax1.set_title('Test RMSE by Dataset')
    ax1.set_ylabel('RMSE')
    for i, v in enumerate(rmse_data.values()):
        ax1.text(i, v+0.5, f'{v:.1f}', ha='center')
    
    # Dataset sizes
    ax2.bar(engines.keys(), engines.values(), color=['lightgreen', 'lightblue', 'yellow', 'pink'])
    ax2.set_title('Test Samples by Dataset')
    ax2.set_ylabel('Samples')
    for i, v in enumerate(engines.values()):
        ax2.text(i, v+100, f'{v:,}', ha='center')
    
    st.pyplot(fig)
    
    # Alert summary table
    alert_summary = {}
    for fd in datasets:
        alerts = [classify_alert(r) for r in datasets[fd]['y_test_pred']]
        alert_summary[fd] = pd.Series(alerts).value_counts(normalize=True).round(3)
    
    st.subheader("Alert Distribution Across Datasets")
    df_alerts = pd.DataFrame(alert_summary).T.fillna(0)*100
    st.dataframe(df_alerts.style.format('{:.1f}%').background_gradient())

# Individual dataset tabs
for i, fd in enumerate(['fd001', 'fd002', 'fd003', 'fd004']):
    with eval(f'tab{i+2}'):
        st.header(f"🔧 {fd.upper()} Live Dashboard")
        
        # Sidebar for this dataset
        with st.sidebar:
            st.header(f"{fd.upper()} Engine Selector")
            engine_id = st.slider("Engine ID", 0, len(datasets[fd]['X'])-1, 0)
        
        # Live prediction
        col1, col2 = st.columns([2,1])
        with col1:
            test_seq = datasets[fd]['X'][engine_id]
            pred_rul = models[fd].predict(test_seq[None,...], verbose=0)[0,0]
            st.metric("Live RUL Prediction", f"{pred_rul:.0f} cycles")
            
            alert = classify_alert(pred_rul)
            colors = {"CRITICAL": "🔴", "WARNING": "🟡", "SAFE": "🟢", "VERY_SAFE": "🔵"}
            st.metric("Alert Status", f"{colors[alert]} {alert}")
        
        with col2:
            st.metric("Test RMSE", f"{datasets[fd]['rmse']:.1f}")
            st.metric("Test Samples", f"{datasets[fd]['n_samples']:,}")
        
        # Plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Predicted vs Actual
        ax1.scatter(datasets[fd]['y_test'], datasets[fd]['y_test_pred'], alpha=0.6, s=10)
        ax1.plot([0,200],[0,200],'r--', lw=2)
        ax1.set_xlabel('Actual RUL'); ax1.set_ylabel('Predicted RUL')
        ax1.set_title(f'{fd.upper()} Test Set (RMSE: {datasets[fd]["rmse"]:.1f})')
        
        # Alert pie
        alerts = [classify_alert(r) for r in datasets[fd]['y_test_pred']]
        counts = pd.Series(alerts).value_counts()
        ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
        ax2.set_title(f'{fd.upper()} Alert Distribution')
        
        st.pyplot(fig)

# Footer
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.success("✅ FD001")
with col2:
    st.info("✅ FD002") 
with col3:
    st.warning("✅ FD003")
with col4:
    st.error("✅ FD004")

st.markdown("""
**🏆 PrognosAI Complete!** All 5 Milestones × 4 Datasets = 20 Deliverables ✅
**NASA CMAPSS Dataset | GRU Time-Series Models | Production Dashboard**
""")
