"""
Explainability engine using SHAP and plain-English non-causal interpretation.
"""

import numpy as np
import pandas as pd
import shap

def explain_prediction(model, input_df: pd.DataFrame, background_data: pd.DataFrame, feature_names: list, model_name: str = ""):
    """
    Computes local SHAP values for the specific prediction and global feature importance.
    Provides plain-English, non-causal commentary.
    """
    shap_values = None
    expected_value = None
    explainer = None

    # Sample background data for speed and stability
    bg_sample = background_data[feature_names].sample(min(len(background_data), 100), random_state=42)

    try:
        # Determine appropriate SHAP explainer
        if any(t in type(model).__name__ for t in ['RandomForest', 'GradientBoosting', 'XGB']):
            explainer = shap.TreeExplainer(model)
            shap_obj = explainer(input_df[feature_names])
            shap_values = shap_obj.values[0]
            expected_value = float(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
        elif 'LinearRegression' in type(model).__name__:
            explainer = shap.LinearExplainer(model, bg_sample)
            shap_obj = explainer(input_df[feature_names])
            shap_values = shap_obj.values[0]
            expected_value = float(explainer.expected_value)
        else:
            explainer = shap.Explainer(model.predict, bg_sample)
            shap_obj = explainer(input_df[feature_names])
            shap_values = shap_obj.values[0]
            expected_value = float(explainer.expected_value)

        if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
            shap_values = shap_values.flatten()
            
    except Exception as e:
        # Graceful fallback: feature perturbation / coefficient approximation
        shap_values = np.zeros(len(feature_names))
        base_pred = float(model.predict(input_df[feature_names])[0])
        expected_value = float(model.predict(bg_sample.mean().to_frame().T)[0])
        
        for idx, f in enumerate(feature_names):
            perturbed_df = input_df[feature_names].copy()
            perturbed_df[f] = bg_sample[f].mean()
            alt_pred = float(model.predict(perturbed_df)[0])
            shap_values[idx] = base_pred - alt_pred

    # Build local contribution DataFrame
    local_contrib = pd.DataFrame({
        'Feature': feature_names,
        'Value': [float(input_df[f].iloc[0]) for f in feature_names],
        'Contribution': [round(float(v), 3) for v in shap_values],
        'AbsoluteContribution': [abs(round(float(v), 3)) for v in shap_values]
    }).sort_values(by='AbsoluteContribution', ascending=False).reset_index(drop=True)

    # Global feature importance
    global_importance = None
    if hasattr(model, 'feature_importances_'):
        global_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': [round(float(v), 4) for v in model.feature_importances_]
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    elif hasattr(model, 'coef_'):
        coef_abs = np.abs(model.coef_)
        total_coef = np.sum(coef_abs) if np.sum(coef_abs) > 0 else 1.0
        global_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': [round(float(v / total_coef), 4) for v in coef_abs]
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    else:
        global_importance = local_contrib[['Feature', 'AbsoluteContribution']].rename(columns={'AbsoluteContribution': 'Importance'})

    # Generate Plain-English, Non-Causal Explanation
    top_feature = local_contrib.iloc[0]
    top_name = top_feature['Feature']
    top_val = top_feature['Value']
    top_impact = top_feature['Contribution']

    positive_factors = local_contrib[local_contrib['Contribution'] > 0.02]['Feature'].tolist()
    negative_factors = local_contrib[local_contrib['Contribution'] < -0.02]['Feature'].tolist()

    lines = []
    if top_impact > 0:
        lines.append(
            f"**{top_name}** ({top_val:.2f}) had the strongest positive influence on this prediction, "
            f"associating with a **+{top_impact:.2f}%** shift relative to the dataset baseline."
        )
    else:
        lines.append(
            f"**{top_name}** ({top_val:.2f}) exerted the strongest downward influence on this prediction, "
            f"associating with a **{top_impact:.2f}%** shift relative to the dataset baseline."
        )

    if positive_factors:
        pos_str = ", ".join(f"**{p}**" for p in positive_factors[:3])
        lines.append(f"In this scenario, higher values in {pos_str} are statistically associated with elevated predicted recovery within the model's learned data.")

    if negative_factors:
        neg_str = ", ".join(f"**{n}**" for n in negative_factors[:3])
        lines.append(f"Conversely, {neg_str} are contributing downward pressure on the predicted recovery index.")

    lines.append(
        "*(Note: These insights reflect empirical associations captured by the model rather than direct mechanical causation. "
        "Real factory crystallization and milling efficiency also depend on operating sanitation and equipment parameters.)*"
    )

    narrative = "\n\n".join(lines)

    return {
        'local_contrib': local_contrib,
        'global_importance': global_importance,
        'expected_value': round(expected_value, 2) if expected_value is not None else 10.0,
        'narrative': narrative,
        'top_feature': top_name
    }
