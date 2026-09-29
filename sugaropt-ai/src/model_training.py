"""
Model training, evaluation, comparison, and persistence for SugarOpt AI.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def get_candidate_models():
    """Returns candidate regression models with solid default hyper-parameters."""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
    }
    if HAS_XGBOOST:
        models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
    return models

def train_and_evaluate_all(prepared_data: dict):
    """
    Trains all candidate models on the prepared train split and evaluates on test split.
    Computes R2, MAE, RMSE, MSE honestly.
    """
    X_train = prepared_data['X_train']
    X_test = prepared_data['X_test']
    y_train = prepared_data['y_train']
    y_test = prepared_data['y_test']

    candidate_models = get_candidate_models()
    results = {}
    trained_instances = {}
    predictions_dict = {}

    for name, model in candidate_models.items():
        try:
            # Fit model
            model.fit(X_train, y_train)
            trained_instances[name] = model

            # Predict on train & test
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            predictions_dict[name] = {
                'y_test': y_test.values,
                'y_pred_test': y_pred_test,
                'y_pred_train': y_pred_train
            }

            # Metrics
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)

            train_mse = mean_squared_error(y_train, y_pred_train)
            train_rmse = np.sqrt(train_mse)
            train_r2 = r2_score(y_train, y_pred_train)

            results[name] = {
                'Test R2': round(float(test_r2), 4),
                'Test MAE': round(float(test_mae), 4),
                'Test RMSE': round(float(test_rmse), 4),
                'Test MSE': round(float(test_mse), 4),
                'Train R2': round(float(train_r2), 4),
                'Train RMSE': round(float(train_rmse), 4),
                'Status': 'Trained Successfully'
            }
        except Exception as e:
            results[name] = {
                'Test R2': None,
                'Test MAE': None,
                'Test RMSE': None,
                'Test MSE': None,
                'Train R2': None,
                'Train RMSE': None,
                'Status': f'Failed: {str(e)}'
            }

    # Format comparison DataFrame
    comparison_rows = []
    for name, m in results.items():
        if m['Status'] == 'Trained Successfully':
            comparison_rows.append({
                'Model': name,
                'R²': m['Test R2'],
                'MAE': m['Test MAE'],
                'RMSE': m['Test RMSE'],
                'MSE': m['Test MSE'],
                'Train R²': m['Train R2']
            })

    comparison_df = pd.DataFrame(comparison_rows)
    
    # Sort by lowest Test RMSE (or highest R2)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(by=['RMSE', 'MAE'], ascending=[True, True]).reset_index(drop=True)
        best_model_name = comparison_df.iloc[0]['Model']
    else:
        best_model_name = None

    return {
        'comparison_df': comparison_df,
        'results_dict': results,
        'trained_models': trained_instances,
        'predictions_dict': predictions_dict,
        'best_model_name': best_model_name
    }

def save_model_artifact(model, feature_names, feature_bounds, model_name, target_name, metrics, output_dir="models"):
    """Serializes the trained model bundle for inference."""
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        'model': model,
        'model_name': model_name,
        'feature_names': feature_names,
        'feature_bounds': feature_bounds,
        'target_name': target_name,
        'metrics': metrics
    }
    file_path = os.path.join(output_dir, "best_sugaropt_model.joblib")
    joblib.dump(bundle, file_path)
    return file_path

def load_saved_model(file_path="models/best_sugaropt_model.joblib"):
    """Loads a serialized model bundle."""
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None
