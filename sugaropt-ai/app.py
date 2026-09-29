"""
SugarOpt AI — Sugarcane Recovery Prediction & Optimization System
Streamlit Production-Style Application
"""

import os
import sys

# Ensure safe console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Automatically delegate to Streamlit runtime if executed directly via 'python app.py'
if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as runtime_exists
        running_in_streamlit = runtime_exists()
    except Exception:
        running_in_streamlit = False

    if not running_in_streamlit:
        from streamlit.web import cli as stcli
        print("\n" + "=" * 60)
        print("[SugarOpt AI] Detected 'python app.py'")
        print(">> Launching via Streamlit runtime...")
        print("   (Standard command: streamlit run app.py)")
        print("=" * 60 + "\n")
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
        sys.exit(stcli.main())

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_processing import (
    load_data, inspect_data_quality, clean_and_preprocess,
    prepare_train_test, compute_correlation_matrix
)
from src.model_training import train_and_evaluate_all, save_model_artifact
from src.prediction import predict_recovery
from src.explainability import explain_prediction
from src.optimization import run_parameter_optimization
from src.utils import format_percentage, SCIENTIFIC_DISCLAIMER

# Page config
st.set_page_config(
    page_title="SugarOpt AI — Sugarcane Recovery Optimization",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if 'dataset_source' not in st.session_state:
    st.session_state.dataset_source = None
if 'is_demo' not in st.session_state:
    st.session_state.is_demo = False
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = 'Recovery_Rate'
if 'prepared_data' not in st.session_state:
    st.session_state.prepared_data = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'selected_model_name' not in st.session_state:
    st.session_state.selected_model_name = None
if 'current_inputs' not in st.session_state:
    st.session_state.current_inputs = {
        'Sucrose': 14.2,
        'Fiber': 13.0,
        'Moisture': 70.0,
        'Rainfall': 850.0,
        'Purity': 84.5,
        'Trash_Content': 4.5
    }
if 'latest_prediction' not in st.session_state:
    st.session_state.latest_prediction = None
if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None

# Helper to automatically load demo dataset
def load_demo_dataset():
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_data.csv")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sugarcane_data.csv")
    
    df, err = load_data(sample_path)
    if df is not None:
        st.session_state.raw_df = df
        st.session_state.dataset_source = "Demo/Synthetic Dataset — For Application Demonstration"
        st.session_state.is_demo = True
        st.session_state.target_col = 'Recovery_Rate' if 'Recovery_Rate' in df.columns else df.columns[-1]
        
        # Preprocess
        cleaned, logs = clean_and_preprocess(df, st.session_state.target_col)
        st.session_state.cleaned_df = cleaned
        prep = prepare_train_test(cleaned, st.session_state.target_col)
        st.session_state.prepared_data = prep
        
        # Train models automatically for zero-friction demo
        res = train_and_evaluate_all(prep)
        st.session_state.training_results = res
        st.session_state.selected_model_name = res['best_model_name']
        
        # Sync current inputs with dataset means
        defaults = {}
        for f in prep['feature_names']:
            defaults[f] = float(prep['feature_bounds'][f]['mean'])
        # If standard core inputs exist, give clean demo values
        if 'Sucrose' in defaults: defaults['Sucrose'] = 14.20
        if 'Fiber' in defaults: defaults['Fiber'] = 13.00
        if 'Moisture' in defaults: defaults['Moisture'] = 70.00
        if 'Rainfall' in defaults: defaults['Rainfall'] = 850.0
        st.session_state.current_inputs = defaults
        
        # Compute baseline prediction
        best_model = res['trained_models'][res['best_model_name']]
        best_rmse = res['comparison_df'][res['comparison_df']['Model'] == res['best_model_name']]['RMSE'].values[0]
        pred_res = predict_recovery(best_model, defaults, prep['feature_names'], prep['feature_bounds'], res['best_model_name'], rmse_metric=best_rmse)
        st.session_state.latest_prediction = pred_res
        return True, "Demo dataset loaded and models trained successfully!"
    return False, f"Failed to load demo data: {err}"

# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌾 SugarOpt AI")
    st.caption("Sugarcane Recovery Optimization Engine")
    st.markdown("---")
    
    page = st.radio(
        "Application Navigation",
        [
            "1. Dashboard",
            "2. Dataset & Quality",
            "3. Model Performance",
            "4. Prediction",
            "5. Explainability",
            "6. What-If Simulator",
            "7. Optimization Engine",
            "8. About & Limitations"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### ⚡ Quick Actions")
    if st.button("🚀 Load Demo Dataset", use_container_width=True, type="primary"):
        ok, msg = load_demo_dataset()
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)
            
    if st.session_state.dataset_source:
        st.markdown(f"**Loaded Data:**\n`{st.session_state.dataset_source}`")
        if st.session_state.is_demo:
            st.info("ℹ️ Mode: Synthetic Demo")
        if st.session_state.selected_model_name:
            st.markdown(f"**Active Model:** `{st.session_state.selected_model_name}`")
    else:
        st.warning("No dataset loaded yet. Click 'Load Demo Dataset' or visit the Dataset page.")

    st.markdown("---")
    st.caption("Prototype decision support. Non-causal model output.")


# -------------------------------------------------------------
# PAGE 1: DASHBOARD
# -------------------------------------------------------------
if page == "1. Dashboard":
    st.markdown("""
        <div class="hero-header">
            <div class="hero-title">SUGAROPT AI</div>
            <div class="hero-subtitle">AI-Powered Sugarcane Recovery Prediction & Optimization System</div>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.is_demo:
        st.markdown("""
            <div class="warning-notice">
                <strong>Demo/Synthetic Dataset Active:</strong> Loaded for application demonstration purposes. 
                Predictions and optimizations represent statistical behavior of simulated mill and field data.
            </div>
        """, unsafe_allow_html=True)

    # Empty State Guidance
    if st.session_state.training_results is None:
        st.info("👋 Welcome! To begin the interactive decision-support tour, load the sample sugarcane dataset.")
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            if st.button("🚀 Load Demo Dataset & Train Models", key="dash_demo_btn", type="primary"):
                ok, msg = load_demo_dataset()
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        st.stop()

    # KPI Summary Cards
    t_res = st.session_state.training_results
    if t_res is not None:
        comp_df = t_res['comparison_df']
        best_m = st.session_state.selected_model_name
        best_stats = comp_df[comp_df['Model'] == best_m].iloc[0] if not comp_df.empty else None
    else:
        comp_df = pd.DataFrame()
        best_m = "None"
        best_stats = None

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        current_pred_val = st.session_state.latest_prediction['predicted_recovery'] if st.session_state.latest_prediction else "--"
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Current Predicted Recovery</div>
                <div class="kpi-value">{current_pred_val}%</div>
                <div class="kpi-sub">Commercial Sugar Yield</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Best Selected Model</div>
                <div class="kpi-value" style="font-size: 1.35rem; margin-top: 0.35rem;">{best_m}</div>
                <div class="kpi-sub">Validation-Selected</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi3:
        r2_val = f"{best_stats['R²']:.3f}" if best_stats is not None else "--"
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Model Test R²</div>
                <div class="kpi-value">{r2_val}</div>
                <div class="kpi-sub">Explained Variance Ratio</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi4:
        n_samples = len(st.session_state.cleaned_df) if st.session_state.cleaned_df is not None else 0
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Training Records</div>
                <div class="kpi-value">{n_samples}</div>
                <div class="kpi-sub">{len(st.session_state.prepared_data['feature_names'])} Input Features</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Dashboard Visual Grid
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown("#### 🎯 Recovery Rate Distribution & Prediction Placement")
        # Plot target distribution
        clean_df = st.session_state.cleaned_df
        t_col = st.session_state.target_col
        fig_dist = px.histogram(
            clean_df, x=t_col, nbins=30,
            title=f"Observed Historical {t_col} Distribution",
            color_discrete_sequence=['#10b981'],
            opacity=0.75
        )
        if st.session_state.latest_prediction:
            pred_v = st.session_state.latest_prediction['predicted_recovery']
            fig_dist.add_vline(
                x=pred_v, line_width=3, line_dash="dash", line_color="#f59e0b",
                annotation_text=f"Current Prediction: {pred_v}%",
                annotation_position="top right"
            )
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        st.markdown("#### 📊 Candidate Models Validation Benchmark")
        # Comparison Chart
        fig_comp = px.bar(
            comp_df, x='Model', y=['R²', 'RMSE', 'MAE'],
            barmode='group',
            title="Validation Metrics by Regressor",
            color_discrete_map={'R²': '#10b981', 'RMSE': '#ef4444', 'MAE': '#f59e0b'}
        )
        fig_comp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Workflow Navigator Cards
    st.markdown("---")
    st.markdown("#### 🗺️ Decision Support Workflow Navigation")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("##### 1. Prediction")
        st.write("Input cane & milling parameters to estimate recovery with confidence bounds.")
        if st.button("Go to Prediction ➡️", key="btn_to_pred"):
            st.info("Select '4. Prediction' from the left sidebar navigation.")
    with c2:
        st.markdown("##### 2. Explainability")
        st.write("Understand SHAP contributions and empirical feature impacts.")
        if st.button("Go to Explainability ➡️", key="btn_to_exp"):
            st.info("Select '5. Explainability' from the left sidebar navigation.")
    with c3:
        st.markdown("##### 3. What-If Simulator")
        st.write("Adjust process sliders to simulate estimated recovery deltas.")
        if st.button("Go to What-If ➡️", key="btn_to_whatif"):
            st.info("Select '6. What-If Simulator' from the left sidebar navigation.")
    with c4:
        st.markdown("##### 4. Optimization Engine")
        st.write("Solve bounded mathematical optimization to seek yield improvements.")
        if st.button("Go to Optimization ➡️", key="btn_to_opt"):
            st.info("Select '7. Optimization Engine' from the left sidebar navigation.")


# -------------------------------------------------------------
# PAGE 2: DATASET & DATA QUALITY
# -------------------------------------------------------------
elif page == "2. Dataset & Quality":
    st.title("📂 Dataset Management & Data Quality Assurance")
    st.markdown("Inspect input data, detect anomalies, check statistical boundaries, or upload factory-specific CSV files.")

    tab_upload, tab_quality, tab_corr = st.tabs(["📁 Dataset Source & Upload", "🔍 Data Quality Audit", "📈 Feature Correlations"])

    with tab_upload:
        st.markdown("#### Load or Ingest Dataset")
        col_u1, col_u2 = st.columns([1.5, 1])
        with col_u1:
            uploaded_file = st.file_uploader("Upload Sugarcane Processing CSV", type=["csv"])
            if uploaded_file is not None:
                df, err = load_data(uploaded_file)
                if df is not None:
                    st.session_state.raw_df = df
                    st.session_state.dataset_source = f"Uploaded File: {uploaded_file.name}"
                    st.session_state.is_demo = False
                    st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df)} rows, {len(df.columns)} columns).")
                else:
                    st.error(f"Upload error: {err}")
        with col_u2:
            st.markdown("##### Or Use Pre-packaged Demo Dataset")
            st.write("Pre-loaded with realistic agronomic features (Sucrose, Fiber, Moisture, Rainfall, Purity, Trash).")
            if st.button("Load Pre-packaged Demo Data", type="primary", use_container_width=True):
                ok, msg = load_demo_dataset()
                if ok:
                    st.success(msg)
                    st.rerun()

        if st.session_state.raw_df is not None:
            df = st.session_state.raw_df
            st.markdown("---")
            st.markdown("#### Target Column Selection & Data Preparation")
            num_cols = list(df.select_dtypes(include=[np.number]).columns)
            if not num_cols:
                st.error("No numeric columns found in the uploaded dataset. Please verify the CSV format.")
            else:
                default_target_idx = num_cols.index(st.session_state.target_col) if st.session_state.target_col in num_cols else len(num_cols)-1
                target_choice = st.selectbox("Select Target Column (Predicted Variable)", num_cols, index=default_target_idx)
                st.session_state.target_col = target_choice
                
                c_clean1, c_clean2 = st.columns([1, 1])
                with c_clean1:
                    remove_dups = st.checkbox("Remove exact duplicate rows", value=True)
                with c_clean2:
                    impute_strat = st.selectbox("Imputation Strategy for Missing Values", ["median", "mean"], index=0)
                
                if st.button("Apply Preprocessing & Prepare Train/Test Split", type="primary"):
                    try:
                        clean_df, logs = clean_and_preprocess(df, target_choice, remove_duplicates=remove_dups, impute_strategy=impute_strat)
                        st.session_state.cleaned_df = clean_df
                        prep = prepare_train_test(clean_df, target_choice)
                        st.session_state.prepared_data = prep
                        
                        # Retrain models
                        res = train_and_evaluate_all(prep)
                        st.session_state.training_results = res
                        st.session_state.selected_model_name = res['best_model_name']
                        
                        # Update current inputs
                        new_inputs = {}
                        for f in prep['feature_names']:
                            new_inputs[f] = float(prep['feature_bounds'][f]['mean'])
                        st.session_state.current_inputs = new_inputs
                        
                        st.success("Preprocessing and model retraining completed successfully!")
                        with st.expander("Inspection and Cleaning Logs", expanded=True):
                            for log_msg in logs:
                                st.write(f"- {log_msg}")
                    except Exception as e:
                        st.error(f"Error during preprocessing: {str(e)}")

            st.markdown("#### Raw Data Preview")
            st.dataframe(df.head(20), use_container_width=True)

    with tab_quality:
        if st.session_state.raw_df is None:
            st.info("Please load or upload a dataset first.")
        else:
            df = st.session_state.raw_df
            t_col = st.session_state.target_col
            report = inspect_data_quality(df, t_col)

            st.markdown("#### 📋 Data Quality Audit Report")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Total Records", report['num_rows'])
            q2.metric("Total Columns", report['num_cols'])
            q3.metric("Duplicate Rows", report['duplicate_rows'])
            q4.metric("Total Missing Values", report['total_missing'])

            st.markdown("##### Outlier Inspection (1.5 × IQR Rule)")
            outlier_rows = []
            for col, stats in report['outlier_summary'].items():
                outlier_rows.append({
                    'Feature': col,
                    'Outlier Count': stats['count'],
                    'Outlier %': f"{stats['pct']}%",
                    'Lower Bound (IQR)': stats['lower_bound'],
                    'Upper Bound (IQR)': stats['upper_bound']
                })
            st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)
            st.caption("Note: Potential outliers are retained during initial cleaning to prevent loss of operational edge-case signals.")

            st.markdown("##### Numerical Feature Statistics & Observed Ranges")
            stats_rows = []
            for col, s in report['feature_stats'].items():
                stats_rows.append({
                    'Feature': col,
                    'Min': s['min'],
                    'Median': s['median'],
                    'Mean': s['mean'],
                    'Max': s['max'],
                    'Std Dev': s['std']
                })
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

    with tab_corr:
        if st.session_state.cleaned_df is not None:
            st.markdown("#### 🔗 Pearson Correlation Heatmap")
            corr_mat = compute_correlation_matrix(st.session_state.cleaned_df)
            fig_corr = px.imshow(
                corr_mat, text_auto=True, aspect="auto",
                color_continuous_scale="Viridis",
                title="Linear Correlation Coefficients (Features vs Recovery Rate)"
            )
            fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8')
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Load and preprocess data to view correlation matrices.")


# -------------------------------------------------------------
# PAGE 3: MODEL PERFORMANCE & BENCHMARKING
# -------------------------------------------------------------
elif page == "3. Model Performance":
    st.title("⚖️ Model Training, Evaluation & Benchmarking")
    st.markdown("Fair, transparent comparison across multiple machine learning regressors on a held-out test split.")

    if st.session_state.training_results is None:
        st.warning("No models trained yet. Please load a dataset first.")
        if st.button("🚀 Load Demo Dataset & Train Models", type="primary"):
            load_demo_dataset()
            st.rerun()
        st.stop()

    t_res = st.session_state.training_results
    comp_df = t_res['comparison_df']
    best_m = t_res['best_model_name']

    st.markdown("#### 🏆 Model Comparison Leaderboard")
    st.markdown("Models are ranked objectively by validation **RMSE** (Root Mean Squared Error) and **R²**.")
    
    # Styled dataframe
    st.dataframe(
        comp_df.style.highlight_min(subset=['RMSE', 'MAE', 'MSE'], color='#064e3b')
                     .highlight_max(subset=['R²'], color='#064e3b'),
        use_container_width=True
    )

    st.markdown(f"""
        <div class="scientific-notice">
            <strong>Active Model Selection:</strong> <code>{st.session_state.selected_model_name}</code> was selected 
            based on superior generalization error on the held-out test set.<br>
            <em>Scientific Honesty Note:</em> Complex algorithms (e.g. XGBoost or Random Forest) are only chosen if they genuinely 
            outperform Linear Regression on validation data. Overfitting is tracked via Train R² vs Test R².
        </div>
    """, unsafe_allow_html=True)

    # Allow user to manually override active model
    c_m1, c_m2 = st.columns([1.5, 2])
    with c_m1:
        chosen = st.selectbox(
            "Select Active Inference Model:",
            comp_df['Model'].tolist(),
            index=comp_df['Model'].tolist().index(st.session_state.selected_model_name)
        )
        if chosen != st.session_state.selected_model_name:
            st.session_state.selected_model_name = chosen
            st.success(f"Active model switched to {chosen}.")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📈 Actual vs. Predicted Validation Scatter")
    
    pred_data = t_res['predictions_dict'].get(st.session_state.selected_model_name)
    if pred_data is not None:
        y_test = pred_data['y_test']
        y_pred = pred_data['y_pred_test']
        
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_test, y=y_pred,
            mode='markers',
            marker=dict(size=8, color='#10b981', opacity=0.7),
            name='Test Predictions'
        ))
        
        # Perfect prediction line
        min_v = min(min(y_test), min(y_pred))
        max_v = max(max(y_test), max(y_pred))
        fig_scatter.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode='lines',
            line=dict(color='#f59e0b', dash='dash'),
            name='Ideal 1:1 Line'
        ))
        
        fig_scatter.update_layout(
            title=f"Actual vs Predicted Sugarcane Recovery ({st.session_state.selected_model_name})",
            xaxis_title="Observed Test Recovery (%)",
            yaxis_title="Model Predicted Recovery (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# -------------------------------------------------------------
# PAGE 4: PREDICTION MODULE
# -------------------------------------------------------------
elif page == "4. Prediction":
    st.title("🎯 Sugarcane Recovery Rate Prediction")
    st.markdown("Generate point recovery rate estimates with statistical uncertainty intervals and input boundary verification.")

    if st.session_state.training_results is None:
        st.warning("Please load a dataset and train models before running predictions.")
        if st.button("🚀 Load Demo Dataset", type="primary"):
            load_demo_dataset()
            st.rerun()
        st.stop()

    prep = st.session_state.prepared_data
    feature_names = prep['feature_names']
    feature_bounds = prep['feature_bounds']
    active_model_name = st.session_state.selected_model_name
    model = st.session_state.training_results['trained_models'][active_model_name]
    best_rmse = st.session_state.training_results['comparison_df'][st.session_state.training_results['comparison_df']['Model'] == active_model_name]['RMSE'].values[0]

    st.markdown("#### Input Parameter Calibration")
    st.caption("Adjust values representing field harvest metrics or factory laboratory readings.")

    # Render inputs dynamically in a clean grid
    cols = st.columns(min(len(feature_names), 3))
    new_inputs = {}
    
    for idx, feat in enumerate(feature_names):
        col = cols[idx % len(cols)]
        b_info = feature_bounds.get(feat, {'min': 0.0, 'max': 100.0, 'mean': 10.0})
        curr_val = float(st.session_state.current_inputs.get(feat, b_info['mean']))
        
        # Step size tailored to magnitude
        step = 0.1 if b_info['max'] < 100 else 10.0
        
        with col:
            val = st.number_input(
                f"{feat} (Observed: {b_info['min']:.1f} - {b_info['max']:.1f})",
                value=curr_val,
                step=step,
                format="%.2f" if step < 1 else "%.1f",
                key=f"pred_input_{feat}"
            )
            new_inputs[feat] = val

    st.session_state.current_inputs = new_inputs

    # Execute Prediction
    if st.button("🔮 Calculate Predicted Sugar Recovery", type="primary", use_container_width=True):
        pred_res = predict_recovery(model, new_inputs, feature_names, feature_bounds, active_model_name, rmse_metric=best_rmse)
        st.session_state.latest_prediction = pred_res

    # Display Prediction Results
    if st.session_state.latest_prediction is not None:
        res = st.session_state.latest_prediction
        pred_val = res['predicted_recovery']
        unc = res['uncertainty']

        st.markdown("---")
        p_col1, p_col2 = st.columns([1, 1.4])
        
        with p_col1:
            st.markdown(f"""
                <div class="kpi-card" style="text-align: left; padding: 1.5rem;">
                    <div class="kpi-label">Predicted Sugar Recovery Rate</div>
                    <div class="kpi-value" style="font-size: 3.2rem; color: #10b981;">{pred_val:.2f}%</div>
                    <div style="font-size: 0.95rem; color: #94a3b8; margin-top: 0.5rem;">
                        <strong>Estimated Range:</strong> [{unc['lower_bound']:.2f}% — {unc['upper_bound']:.2f}%]<br>
                        <strong>Uncertainty Method:</strong> {unc['method']}<br>
                        <strong>Model Employed:</strong> {res['model_name']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with p_col2:
            st.markdown("##### Input Distribution & Domain Boundary Status")
            if res['has_warnings']:
                for w in res['warnings']:
                    st.markdown(f"""
                        <div class="warning-notice">
                            ⚠️ <strong>Out-of-Distribution Warning:</strong><br>{w}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="scientific-notice">
                        ✅ <strong>Inputs Within Training Range:</strong> All specified feature values reside comfortably 
                        within the statistical bounds of the historical training dataset.
                    </div>
                """, unsafe_allow_html=True)

            # Table of inputs vs training range
            status_table = []
            for feat, b in res['bounds_check'].items():
                status_table.append({
                    'Parameter': feat,
                    'Input Value': b['value'],
                    'Training Range': f"[{b['min']:.2f}, {b['max']:.2f}]",
                    'Status': '✅ In Bounds' if b['in_bounds'] else '⚠️ Out of Range'
                })
            st.dataframe(pd.DataFrame(status_table), use_container_width=True)


# -------------------------------------------------------------
# PAGE 5: EXPLAINABILITY (SHAP & PLAIN-ENGLISH)
# -------------------------------------------------------------
elif page == "5. Explainability":
    st.title("🔬 Model Explainability & Feature Contribution")
    st.markdown("Understand **why** the model predicted this specific recovery rate using local SHAP attribution and global feature hierarchies.")

    if st.session_state.training_results is None or st.session_state.latest_prediction is None:
        st.warning("Please generate a prediction first in the 'Prediction' page or load the demo dataset.")
        if st.button("🚀 Load Demo Dataset & Auto-Predict", type="primary"):
            load_demo_dataset()
            st.rerun()
        st.stop()

    prep = st.session_state.prepared_data
    feature_names = prep['feature_names']
    active_model_name = st.session_state.selected_model_name
    model = st.session_state.training_results['trained_models'][active_model_name]
    curr_inputs = st.session_state.current_inputs
    input_df = pd.DataFrame([{f: float(curr_inputs[f]) for f in feature_names}])

    with st.spinner("Computing SHAP feature attributions..."):
        shap_res = explain_prediction(model, input_df, prep['X_train'], feature_names, active_model_name)

    st.markdown("#### 🗣️ Plain-English Interpretation")
    st.markdown(f"""
        <div class="scientific-notice" style="font-size: 0.98rem; line-height: 1.6;">
            {shap_res['narrative']}
        </div>
    """, unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([1.2, 1])

    with col_e1:
        st.markdown("#### 🎯 Local Feature Contribution (SHAP Impact)")
        st.caption("How each parameter pushed the prediction higher (emerald) or lower (coral) relative to the baseline.")
        
        contrib_df = shap_res['local_contrib'].copy()
        contrib_df['Color'] = contrib_df['Contribution'].apply(lambda x: '#10b981' if x >= 0 else '#f43f5e')
        
        fig_local = go.Figure(go.Bar(
            x=contrib_df['Contribution'],
            y=contrib_df['Feature'],
            orientation='h',
            marker=dict(color=contrib_df['Color']),
            text=[f"{v:+.3f}%" for v in contrib_df['Contribution']],
            textposition='auto'
        ))
        fig_local.update_layout(
            title="Local SHAP Attribution for Current Prediction",
            xaxis_title="Contribution to Recovery Rate (% pts)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_local, use_container_width=True)

    with col_e2:
        st.markdown("#### 🌐 Global Feature Importance")
        st.caption("Overall relative importance across the entire training dataset.")
        
        g_df = shap_res['global_importance']
        fig_global = px.bar(
            g_df, x='Importance', y='Feature', orientation='h',
            title=f"Global Feature Hierarchy ({active_model_name})",
            color_discrete_sequence=['#3b82f6']
        )
        fig_global.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_global, use_container_width=True)


# -------------------------------------------------------------
# PAGE 6: WHAT-IF SIMULATOR
# -------------------------------------------------------------
elif page == "6. What-If Simulator":
    st.title("🎛️ What-If Scenario Simulator")
    st.markdown("""
        <div class="warning-notice">
            <strong>MODEL SIMULATION NOTICE:</strong> This tool simulates the trained machine learning model's response 
            to hypothetical input shifts. It does <em>not</em> guarantee that changing physical mill or farm conditions will 
            produce exact identical yield increases in an operating sugar factory.
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.training_results is None:
        st.warning("Please load demo data or train models first.")
        if st.button("🚀 Load Demo Dataset", type="primary"):
            load_demo_dataset()
            st.rerun()
        st.stop()

    prep = st.session_state.prepared_data
    feature_names = prep['feature_names']
    feature_bounds = prep['feature_bounds']
    active_model_name = st.session_state.selected_model_name
    model = st.session_state.training_results['trained_models'][active_model_name]
    best_rmse = st.session_state.training_results['comparison_df'][st.session_state.training_results['comparison_df']['Model'] == active_model_name]['RMSE'].values[0]

    # Compute baseline
    current_inputs = st.session_state.current_inputs
    base_pred_res = predict_recovery(model, current_inputs, feature_names, feature_bounds, active_model_name, rmse_metric=best_rmse)
    base_val = base_pred_res['predicted_recovery']

    st.markdown("#### Scenario Parameter Adjustment")
    st.caption("Tweak one or multiple parameters to simulate the potential impact on predicted sugar recovery.")

    sim_cols = st.columns(min(len(feature_names), 3))
    scenario_inputs = {}

    for idx, feat in enumerate(feature_names):
        col = sim_cols[idx % len(sim_cols)]
        b_info = feature_bounds.get(feat, {'min': 0.0, 'max': 100.0, 'mean': 10.0})
        curr_v = float(current_inputs.get(feat, b_info['mean']))
        
        # Calculate reasonable slider bounds (from 80% min to 120% max of training range)
        f_min = float(round(b_info['min'] * 0.9, 1))
        f_max = float(round(b_info['max'] * 1.1, 1))
        step = 0.1 if f_max < 100 else 10.0
        
        with col:
            s_val = st.slider(
                f"{feat} (Current: {curr_v:.2f})",
                min_value=f_min,
                max_value=f_max,
                value=curr_v,
                step=step,
                key=f"whatif_slider_{feat}"
            )
            scenario_inputs[feat] = s_val

    # Evaluate scenario
    scenario_pred_res = predict_recovery(model, scenario_inputs, feature_names, feature_bounds, active_model_name, rmse_metric=best_rmse)
    scenario_val = scenario_pred_res['predicted_recovery']
    delta_val = round(scenario_val - base_val, 2)

    st.markdown("---")
    st.markdown("#### 📊 Simulation Comparison Results")

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Baseline Predicted Recovery</div>
                <div class="kpi-value">{base_val:.2f}%</div>
                <div class="kpi-sub">Original Input Settings</div>
            </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Scenario Predicted Recovery</div>
                <div class="kpi-value" style="color: {'#10b981' if delta_val >= 0 else '#f43f5e'};">{scenario_val:.2f}%</div>
                <div class="kpi-sub">Hypothetical Simulation</div>
            </div>
        """, unsafe_allow_html=True)

    with res_col3:
        sign = "+" if delta_val > 0 else ""
        delta_color = "#10b981" if delta_val > 0 else ("#f43f5e" if delta_val < 0 else "#94a3b8")
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Simulated Shift (Delta)</div>
                <div class="kpi-value" style="color: {delta_color};">{sign}{delta_val:.2f} % pts</div>
                <div class="kpi-sub">Estimated Differential</div>
            </div>
        """, unsafe_allow_html=True)

    # Before / After Plotly Visualization
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Bar(
        x=['Baseline Conditions', 'Simulated Scenario'],
        y=[base_val, scenario_val],
        marker=dict(color=['#3b82f6', '#10b981' if delta_val >= 0 else '#f43f5e']),
        text=[f"{base_val:.2f}%", f"{scenario_val:.2f}%"],
        textposition='auto',
        width=[0.4, 0.4]
    ))
    fig_sim.update_layout(
        title="Comparison of Baseline vs. Simulated Recovery Rate",
        yaxis_title="Predicted Sugar Recovery (%)",
        yaxis=dict(range=[max(0, min(base_val, scenario_val) - 2), max(base_val, scenario_val) + 2]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94a3b8'
    )
    st.plotly_chart(fig_sim, use_container_width=True)


# -------------------------------------------------------------
# PAGE 7: OPTIMIZATION ENGINE
# -------------------------------------------------------------
elif page == "7. Optimization Engine":
    st.title("⚙️ Bounded Parameter Optimization Engine")
    st.markdown("""
        <div class="warning-notice">
            <strong>CRITICAL SCIENTIFIC DISCLAIMER:</strong> Optimization is performed mathematically against the trained 
            ML model surrogate function. It identifies input configurations that maximize the statistical response within 
            realistic boundaries. It does <em>not</em> guarantee real-world factory recovery improvements.
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.training_results is None:
        st.warning("Please load a dataset and train models first.")
        if st.button("🚀 Load Demo Dataset", type="primary"):
            load_demo_dataset()
            st.rerun()
        st.stop()

    prep = st.session_state.prepared_data
    feature_names = prep['feature_names']
    feature_bounds = prep['feature_bounds']
    active_model_name = st.session_state.selected_model_name
    model = st.session_state.training_results['trained_models'][active_model_name]
    curr_inputs = st.session_state.current_inputs

    st.markdown("#### 1. Define Optimization Boundaries & Controllability")
    st.markdown("Configure realistic allowable bounds. Non-controllable variables (e.g. historical Rainfall) can be held fixed.")

    opt_bounds = {}
    fixed_feats = []

    b_cols = st.columns(min(len(feature_names), 3))
    for idx, f in enumerate(feature_names):
        col = b_cols[idx % len(b_cols)]
        train_b = feature_bounds.get(f, {'min': 0.0, 'max': 100.0, 'mean': 10.0})
        curr_v = float(curr_inputs.get(f, train_b['mean']))
        
        with col:
            st.markdown(f"**{f}** (Current: `{curr_v:.2f}`)")
            is_fixed = st.checkbox(f"Lock as Fixed", value=(f.lower() in ['rainfall', 'weather']), key=f"fix_opt_{f}")
            if is_fixed:
                fixed_feats.append(f)
                opt_bounds[f] = (curr_v, curr_v)
            else:
                default_min = float(round(train_b['min'], 2))
                default_max = float(round(train_b['max'], 2))
                c_min, c_max = st.slider(
                    f"Safe Range for {f}",
                    min_value=float(round(train_b['min'] * 0.85, 2)),
                    max_value=float(round(train_b['max'] * 1.15, 2)),
                    value=(default_min, default_max),
                    key=f"slider_opt_{f}"
                )
                opt_bounds[f] = (c_min, c_max)

    st.markdown("#### 2. Optimizer Configuration")
    opt_method = st.selectbox(
        "Optimization Solver:",
        ["SLSQP (Sequential Least Squares Programming)", "L-BFGS-B (Bounded Quasi-Newton)", "differential_evolution (Global Search)"],
        index=0
    )
    solver_key = "SLSQP" if "SLSQP" in opt_method else ("L-BFGS-B" if "L-BFGS-B" in opt_method else "differential_evolution")

    if st.button("🚀 Find Best Realistic Scenario", type="primary", use_container_width=True):
        with st.spinner("Executing bounded optimization against surrogate ML model..."):
            opt_res = run_parameter_optimization(
                model=model,
                current_inputs=curr_inputs,
                feature_names=feature_names,
                param_bounds=opt_bounds,
                fixed_features=fixed_feats,
                method=solver_key
            )
            st.session_state.optimization_result = opt_res

    # Display Optimization Results
    if st.session_state.optimization_result is not None:
        ores = st.session_state.optimization_result
        
        st.markdown("---")
        st.markdown("#### 🎯 Optimization Outcome")

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Current Baseline Recovery</div>
                    <div class="kpi-value">{ores['baseline_recovery']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Optimized Scenario Recovery</div>
                    <div class="kpi-value" style="color: #10b981;">{ores['optimized_recovery']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Potential Estimated Improvement</div>
                    <div class="kpi-value" style="color: #10b981;">+{ores['improvement_delta']:.2f} % pts</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("##### 📝 Analytical Commentary")
        st.markdown(f"""
            <div class="scientific-notice">
                {ores['explanation']}
            </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 📋 Recommended Parameter Adjustments")
        st.dataframe(ores['comparison_df'], use_container_width=True)

        # Plot Parameter Shifts
        fig_opt = go.Figure()
        comp_data = ores['comparison_df']
        fig_opt.add_trace(go.Bar(
            name='Current', x=comp_data['Parameter'], y=comp_data['Current'],
            marker_color='#3b82f6'
        ))
        fig_opt.add_trace(go.Bar(
            name='Optimized', x=comp_data['Parameter'], y=comp_data['Optimized'],
            marker_color='#10b981'
        ))
        fig_opt.update_layout(
            barmode='group',
            title="Current vs. Optimizer-Suggested Parameter Values",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8'
        )
        st.plotly_chart(fig_opt, use_container_width=True)


# -------------------------------------------------------------
# PAGE 8: ABOUT & LIMITATIONS
# -------------------------------------------------------------
elif page == "8. About & Limitations":
    st.title("ℹ️ About SugarOpt AI & System Limitations")

    st.markdown(f"""
        ### 🌾 Mission & Purpose
        **SugarOpt AI** is a decision-support prototype engineered to assist sugarcane mill managers, 
        process engineers, and agronomic planners. It transforms raw cane parameter measurements into actionable 
        recovery predictions and simulated yield optimization pathways.

        ---

        ### 🔬 Core Scientific Principles & Distinction of Concepts
        To maintain absolute scientific integrity, this system strictly categorizes operational insights into four levels:
        
        1. **Observed Data:** Historical records of cane sampling, juice purity, fiber, and actual factory sugar recovery.
        2. **Model Prediction:** Statistical point estimates conditioned on learned mathematical patterns. Includes 95% confidence intervals to convey uncertainty.
        3. **Model Explanation:** Local SHAP attributions and global feature hierarchies that explain **why** the model made a prediction without assuming physical causation.
        4. **Model-Based Optimization:** Constrained mathematical maximization (using SciPy SLSQP/L-BFGS-B) across the surrogate ML landscape.
        
        ---

        ### ⚠️ System Limitations & Operational Boundaries
        - **Decision Support Only:** This application does *not* automate physical mill controls, valves, or shredder speeds.
        - **No Hardware or Sensor Integration:** No IoT devices, drones, or automated telemetry are connected. All inputs are provided via tabular batch or manual interface.
        - **Surrogate Optimization Caveat:** The optimization engine finds the optimum of the *trained ML model*, not the physical chemical factory. Unmodeled dynamics (e.g. microbial dextran formation, mill roller grooving wear, boiler steam pressure) also affect recovery.
        - **Out-of-Distribution Degradation:** Model reliability degrades when extrapolating beyond training domain boundaries. The system explicitly flags these warnings.

        ---

        ### 🛡️ Disclaimer
        > *{SCIENTIFIC_DISCLAIMER}*
    """)
