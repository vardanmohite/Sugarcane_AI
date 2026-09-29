"""
Constrained parameter optimization engine using SciPy.
Maximizes predicted recovery rate within realistic agricultural/process bounds.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

def run_parameter_optimization(
    model,
    current_inputs: dict,
    feature_names: list,
    param_bounds: dict,
    fixed_features: list = None,
    method: str = "SLSQP"
):
    """
    Finds optimal parameter values within specified bounds to maximize predicted recovery.
    
    Args:
        model: Trained scikit-learn or XGBoost regression model.
        current_inputs: Dict of current baseline values.
        feature_names: List of model feature names.
        param_bounds: Dict of {feature: (min_val, max_val)}.
        fixed_features: Features that cannot be controlled/altered (e.g. Rainfall).
        method: SciPy optimization algorithm ('SLSQP', 'L-BFGS-B', or 'differential_evolution').
    """
    if fixed_features is None:
        fixed_features = []

    # Baseline evaluation
    base_df = pd.DataFrame([{f: float(current_inputs[f]) for f in feature_names}])
    baseline_recovery = round(float(model.predict(base_df)[0]), 2)

    # Initial guess x0
    x0 = [float(current_inputs[f]) for f in feature_names]
    
    # Construct bounds
    bounds_list = []
    for f in feature_names:
        if f in fixed_features:
            # Lock fixed feature tightly to current value
            val = float(current_inputs[f])
            bounds_list.append((val, val))
        else:
            b_min, b_max = param_bounds.get(f, (x0[feature_names.index(f)] * 0.9, x0[feature_names.index(f)] * 1.1))
            bounds_list.append((float(b_min), float(b_max)))

    # Objective function: minimize negative predicted recovery
    def objective(x):
        sample_df = pd.DataFrame([dict(zip(feature_names, x))])
        # Predict
        pred = model.predict(sample_df)[0]
        # Return negative for maximization
        return -float(pred)

    try:
        if method == "differential_evolution":
            res = differential_evolution(objective, bounds=bounds_list, seed=42, maxiter=50, popsize=10)
            optimal_x = res.x
        else:
            # SLSQP or L-BFGS-B
            opt_method = "SLSQP" if method == "SLSQP" else "L-BFGS-B"
            res = minimize(objective, x0=x0, bounds=bounds_list, method=opt_method)
            optimal_x = res.x
            
        opt_df = pd.DataFrame([dict(zip(feature_names, optimal_x))])
        optimized_recovery = round(float(model.predict(opt_df)[0]), 2)
        
    except Exception as e:
        # Fallback to random search within bounds
        best_val = baseline_recovery
        optimal_x = x0.copy()
        np.random.seed(42)
        for _ in range(200):
            cand_x = []
            for i, f in enumerate(feature_names):
                b_min, b_max = bounds_list[i]
                cand_x.append(np.random.uniform(b_min, b_max))
            cand_df = pd.DataFrame([dict(zip(feature_names, cand_x))])
            cand_pred = float(model.predict(cand_df)[0])
            if cand_pred > best_val:
                best_val = cand_pred
                optimal_x = cand_x
        optimized_recovery = round(float(best_val), 2)

    improvement_delta = round(optimized_recovery - baseline_recovery, 2)

    # Format parameter comparison table
    comparison_records = []
    for idx, f in enumerate(feature_names):
        curr_v = float(current_inputs[f])
        opt_v = float(optimal_x[idx])
        delta_v = opt_v - curr_v
        pct_change = (delta_v / curr_v * 100) if curr_v != 0 else 0.0
        
        comparison_records.append({
            'Parameter': f,
            'Current': round(curr_v, 2),
            'Optimized': round(opt_v, 2),
            'Delta': round(delta_v, 2),
            'Pct_Change': round(pct_change, 1),
            'Is_Controllable': "Fixed" if f in fixed_features else "Controllable",
            'AbsDelta': abs(delta_v)
        })

    comp_df = pd.DataFrame(comparison_records).sort_values(by='AbsDelta', ascending=False).reset_index(drop=True)

    # Explain which variables changed most
    controllable_changes = comp_df[comp_df['Is_Controllable'] == 'Controllable']
    top_change = controllable_changes.iloc[0] if not controllable_changes.empty else comp_df.iloc[0]

    explanation = (
        f"Under the model's learned relationships, this scenario produces the highest predicted recovery "
        f"({optimized_recovery:.2f}%) within the selected parameter bounds.\n\n"
        f"The parameter with the most substantial suggested adjustment is **{top_change['Parameter']}**, "
        f"shifting from **{top_change['Current']:.2f}** to **{top_change['Optimized']:.2f}** "
        f"({top_change['Delta']:+.2f}, {top_change['Pct_Change']:+.1f}%)."
    )

    return {
        'baseline_recovery': baseline_recovery,
        'optimized_recovery': optimized_recovery,
        'improvement_delta': improvement_delta,
        'comparison_df': comp_df.drop(columns=['AbsDelta']),
        'optimized_inputs': dict(zip(feature_names, [round(float(v), 2) for v in optimal_x])),
        'explanation': explanation,
        'disclaimer': (
            "Optimization is performed against the trained ML model. "
            "It does not guarantee real-world improvement. Mill parameters must always be verified by chemical control staff."
        )
    }
