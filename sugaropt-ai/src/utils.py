"""
Utility functions, constants, formatting, and domain styling for SugarOpt AI.
"""

# Agricultural & Industrial Domain Defaults
DEFAULT_FEATURE_RANGES = {
    'Sucrose': {'min': 8.0, 'max': 18.0, 'unit': '%', 'desc': 'Sucrose (Pol % juice or cane)'},
    'Fiber': {'min': 9.0, 'max': 17.5, 'unit': '%', 'desc': 'Dry fiber content in cane'},
    'Moisture': {'min': 62.0, 'max': 78.0, 'unit': '%', 'desc': 'Moisture content in cane'},
    'Rainfall': {'min': 400.0, 'max': 2000.0, 'unit': 'mm', 'desc': 'Cumulative pre-harvest rainfall'},
    'Purity': {'min': 70.0, 'max': 95.0, 'unit': '%', 'desc': 'Apparent juice purity (Pol/Brix)'},
    'Trash_Content': {'min': 0.5, 'max': 15.0, 'unit': '%', 'desc': 'Extraneous plant material & soil'},
}

# Scientific Honesty Disclaimer
SCIENTIFIC_DISCLAIMER = (
    "SugarOpt AI is an AI-powered decision-support prototype. "
    "Predictions and optimizations represent statistical patterns learned from historical data "
    "and must be validated by mill process engineers and agronomists before operational application."
)

def format_percentage(val, decimals=2, include_sign=False):
    """Format float into clean percentage string."""
    if val is None:
        return "N/A"
    if include_sign:
        return f"{val:+.{decimals}f}%"
    return f"{val:.{decimals}f}%"

def check_feature_bounds(value, col_name, min_val, max_val):
    """
    Check if a parameter value is outside the training data distribution.
    Returns (is_valid, warning_message).
    """
    if value < min_val:
        return False, f"'{col_name}' ({value:.2f}) is below training min ({min_val:.2f}). Prediction reliability may be reduced."
    elif value > max_val:
        return False, f"'{col_name}' ({value:.2f}) is above training max ({max_val:.2f}). Prediction reliability may be reduced."
    return True, ""
