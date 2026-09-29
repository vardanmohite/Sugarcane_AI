"""
Comprehensive End-to-End Test Suite for SugarOpt AI.
Verifies all modules: data processing, training, prediction, explainability, optimization.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_processing import (
    load_data, inspect_data_quality, clean_and_preprocess,
    prepare_train_test, compute_correlation_matrix
)
from src.model_training import train_and_evaluate_all, save_model_artifact, load_saved_model
from src.prediction import predict_recovery
from src.explainability import explain_prediction
from src.optimization import run_parameter_optimization

def run_all_tests():
    print("==================================================")
    print("RUNNING SUGAROPT AI COMPREHENSIVE TEST SUITE")
    print("==================================================")

    # 1. Test Data Loading
    sample_csv = os.path.join(os.path.dirname(__file__), "data", "sample_data.csv")
    assert os.path.exists(sample_csv), f"Sample data file missing at {sample_csv}"
    df, err = load_data(sample_csv)
    assert err is None, f"Data load failed: {err}"
    assert len(df) > 50, f"Expected >50 rows, got {len(df)}"
    print(f"✅ 1. Data Loading Passed: {len(df)} rows, {len(df.columns)} columns.")

    # 2. Test Data Quality Inspection
    quality_rep = inspect_data_quality(df, target_col='Recovery_Rate')
    assert quality_rep['num_rows'] == len(df)
    assert 'Recovery_Rate' in quality_rep['numeric_cols']
    assert len(quality_rep['outlier_summary']) > 0
    print(f"✅ 2. Data Quality Audit Passed: Found {quality_rep['duplicate_rows']} duplicates, {quality_rep['total_missing']} missing values.")

    # 3. Test Preprocessing and Cleaning
    cleaned_df, logs = clean_and_preprocess(df, target_col='Recovery_Rate', remove_duplicates=True, impute_strategy="median")
    assert cleaned_df.isnull().sum().sum() == 0, "Cleaned dataset still contains missing values"
    assert cleaned_df.duplicated().sum() == 0, "Cleaned dataset still contains duplicates"
    assert len(logs) > 0, "Expected preprocessing logs"
    print(f"✅ 3. Data Cleaning Passed: Cleaned shape {cleaned_df.shape}. Actions: {len(logs)} logged.")

    # 4. Test Train/Test Split
    prep = prepare_train_test(cleaned_df, target_col='Recovery_Rate')
    assert len(prep['X_train']) > len(prep['X_test'])
    assert 'Sucrose' in prep['feature_names']
    print(f"✅ 4. Train/Test Split Passed: {len(prep['X_train'])} train, {len(prep['X_test'])} test samples.")

    # 5. Test Model Training & Comparison
    train_res = train_and_evaluate_all(prep)
    comp_df = train_res['comparison_df']
    assert not comp_df.empty, "Comparison table is empty"
    assert len(comp_df) >= 3, f"Expected at least 3 models, got {len(comp_df)}"
    assert 'R²' in comp_df.columns and 'RMSE' in comp_df.columns
    best_name = train_res['best_model_name']
    assert best_name in train_res['trained_models']
    print("✅ 5. Model Training & Benchmarking Passed. Comparison Leaderboard:")
    for idx, row in comp_df.iterrows():
        print(f"   - {row['Model']}: R²={row['R²']:.3f}, RMSE={row['RMSE']:.3f}, MAE={row['MAE']:.3f}")
    print(f"   Selected Best Generalization Model: '{best_name}'")

    # 6. Test Model Serialization
    best_model = train_res['trained_models'][best_name]
    best_rmse = comp_df[comp_df['Model'] == best_name]['RMSE'].values[0]
    art_path = save_model_artifact(
        best_model, prep['feature_names'], prep['feature_bounds'],
        best_name, 'Recovery_Rate', train_res['results_dict'][best_name],
        output_dir="models"
    )
    assert os.path.exists(art_path), "Model artifact was not saved"
    loaded_bundle = load_saved_model(art_path)
    assert loaded_bundle['model_name'] == best_name
    print(f"✅ 6. Model Persistence Passed: Saved and reloaded artifact at {art_path}")

    # 7. Test Prediction Engine & Bounds Check
    test_inputs = {f: float(prep['feature_bounds'][f]['mean']) for f in prep['feature_names']}
    test_inputs['Sucrose'] = 14.2
    test_inputs['Fiber'] = 13.0
    test_inputs['Moisture'] = 70.0
    test_inputs['Rainfall'] = 850.0

    pred_res = predict_recovery(best_model, test_inputs, prep['feature_names'], prep['feature_bounds'], best_name, rmse_metric=best_rmse)
    assert 5.0 <= pred_res['predicted_recovery'] <= 18.0, f"Unrealistic recovery: {pred_res['predicted_recovery']}"
    assert 'uncertainty' in pred_res
    assert pred_res['uncertainty']['lower_bound'] <= pred_res['predicted_recovery'] <= pred_res['uncertainty']['upper_bound']
    print(f"✅ 7. Point Prediction Passed: Predicted {pred_res['predicted_recovery']}% (95% CI: [{pred_res['uncertainty']['lower_bound']}% - {pred_res['uncertainty']['upper_bound']}%])")

    # 8. Test Out-of-Bounds Warning Handling
    extreme_inputs = test_inputs.copy()
    extreme_inputs['Sucrose'] = 28.5  # Absurdly high
    pred_extreme = predict_recovery(best_model, extreme_inputs, prep['feature_names'], prep['feature_bounds'], best_name, rmse_metric=best_rmse)
    assert pred_extreme['has_warnings'] is True
    assert len(pred_extreme['warnings']) > 0
    print(f"✅ 8. Out-of-Distribution Warning Passed: Successfully flagged extreme input: {pred_extreme['warnings'][0]}")

    # 9. Test SHAP Explainability Engine
    exp_res = explain_prediction(
        best_model, pred_res['input_df'], prep['X_train'], prep['feature_names'], best_name
    )
    assert not exp_res['local_contrib'].empty
    assert 'narrative' in exp_res and len(exp_res['narrative']) > 20
    print(f"✅ 9. SHAP Explainability Passed. Top Influential Factor: {exp_res['top_feature']}. Narrative preview:")
    print(f"   \"{exp_res['narrative'].splitlines()[0]}\"")

    # 10. Test SciPy Constrained Optimization
    opt_bounds = {f: (float(prep['feature_bounds'][f]['min']), float(prep['feature_bounds'][f]['max'])) for f in prep['feature_names']}
    opt_res = run_parameter_optimization(
        best_model,
        current_inputs=test_inputs,
        feature_names=prep['feature_names'],
        param_bounds=opt_bounds,
        fixed_features=['Rainfall'],
        method="SLSQP"
    )
    assert opt_res['optimized_recovery'] >= opt_res['baseline_recovery'], "Optimization failed to improve or maintain recovery"
    assert opt_res['comparison_df'].loc[opt_res['comparison_df']['Parameter'] == 'Rainfall', 'Is_Controllable'].values[0] == 'Fixed'
    print(f"✅ 10. Optimization Engine Passed: Baseline {opt_res['baseline_recovery']}% -> Optimized {opt_res['optimized_recovery']}% (Delta: +{opt_res['improvement_delta']}% pts).")

    # 11. Test 4-Feature Baseline Dataset Compatibility
    base_4_csv = os.path.join(os.path.dirname(__file__), "data", "sugarcane_data.csv")
    df_4, _ = load_data(base_4_csv)
    c4, _ = clean_and_preprocess(df_4, 'Recovery_Rate')
    p4 = prepare_train_test(c4, 'Recovery_Rate')
    t4 = train_and_evaluate_all(p4)
    assert len(t4['comparison_df']) >= 3
    print(f"✅ 11. 4-Feature Baseline Dataset Compatibility Passed: Tested with {c4.shape[1]} columns successfully.")

    print("\n==================================================")
    print("🎉 ALL TESTS PASSED SUCCESSFULLY! ZERO ERRORS.")
    print("==================================================")

if __name__ == "__main__":
    run_all_tests()
