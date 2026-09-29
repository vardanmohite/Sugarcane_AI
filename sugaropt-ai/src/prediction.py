"""
Inference and prediction service with input range validation and uncertainty estimation.
"""

import numpy as np
import pandas as pd

def predict_recovery(model, input_dict: dict, feature_names: list, feature_bounds: dict, model_name: str = "Trained Model", rmse_metric: float = 0.35):
    """
    Predict sugarcane recovery rate given an input vector.
    Validates range against training distribution and computes defensible uncertainty interval.
    """
    # 1. Validate all features are present
    missing_feats = [f for f in feature_names if f not in input_dict]
    if missing_feats:
        raise ValueError(f"Missing required input features: {missing_feats}")

    # 2. Build aligned DataFrame
    ordered_values = {f: [float(input_dict[f])] for f in feature_names}
    input_df = pd.DataFrame(ordered_values)

    # 3. Check bounds against training distribution
    bounds_check = {}
    warnings = []
    for f in feature_names:
        val = float(input_dict[f])
        f_min = feature_bounds.get(f, {}).get('min', -np.inf)
        f_max = feature_bounds.get(f, {}).get('max', np.inf)
        
        is_in_bounds = (val >= f_min) and (val <= f_max)
        bounds_check[f] = {
            'value': val,
            'min': f_min,
            'max': f_max,
            'in_bounds': is_in_bounds
        }
        if not is_in_bounds:
            if val < f_min:
                warnings.append(
                    f"Warning: '{f}' value ({val:.2f}) is below the training range [{f_min:.2f} to {f_max:.2f}]. "
                    f"Prediction reliability may be reduced."
                )
            else:
                warnings.append(
                    f"Warning: '{f}' value ({val:.2f}) is above the training range [{f_min:.2f} to {f_max:.2f}]. "
                    f"Prediction reliability may be reduced."
                )

    # 4. Point prediction
    raw_pred = float(model.predict(input_df)[0])
    # Commercial sugar recovery is bounded physically between ~5% and 18%
    predicted_recovery = round(float(np.clip(raw_pred, 5.0, 18.0)), 2)

    # 5. Defensible uncertainty interval
    # If random forest, we can compute tree-level variance:
    if hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'predict'):
        try:
            tree_preds = [tree.predict(input_df.values)[0] for tree in model.estimators_]
            std_err = float(np.std(tree_preds))
            lower_bound = round(max(5.0, predicted_recovery - 1.96 * std_err), 2)
            upper_bound = round(min(18.0, predicted_recovery + 1.96 * std_err), 2)
            method = "Ensemble Estimator Variance (95% CI)"
        except Exception:
            std_err = float(rmse_metric)
            lower_bound = round(max(5.0, predicted_recovery - 1.96 * std_err), 2)
            upper_bound = round(min(18.0, predicted_recovery + 1.96 * std_err), 2)
            method = "Validation Residual Standard Error (95% CI)"
    else:
        std_err = float(rmse_metric) if rmse_metric else 0.40
        lower_bound = round(max(5.0, predicted_recovery - 1.96 * std_err), 2)
        upper_bound = round(min(18.0, predicted_recovery + 1.96 * std_err), 2)
        method = "Validation Residual Standard Error (95% CI)"

    return {
        'predicted_recovery': predicted_recovery,
        'uncertainty': {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'margin': round(upper_bound - predicted_recovery, 2),
            'method': method
        },
        'bounds_check': bounds_check,
        'warnings': warnings,
        'has_warnings': len(warnings) > 0,
        'model_name': model_name,
        'input_df': input_df
    }
