"""
Data loading, validation, quality inspection, and preprocessing for SugarOpt AI.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path_or_buffer):
    """Load dataset from path or uploaded file buffer."""
    try:
        if isinstance(file_path_or_buffer, str):
            df = pd.read_csv(file_path_or_buffer)
        else:
            df = pd.read_csv(file_path_or_buffer)
        return df, None
    except Exception as e:
        return None, f"Failed to load dataset: {str(e)}"

def inspect_data_quality(df: pd.DataFrame, target_col: str = None):
    """
    Perform thorough data quality inspection:
    - Shape, types, missing values, duplicates
    - Outlier detection via IQR
    - Feature ranges & target statistics
    """
    report = {
        'num_rows': len(df),
        'num_cols': len(df.columns),
        'columns': list(df.columns),
        'numeric_cols': list(df.select_dtypes(include=[np.number]).columns),
        'non_numeric_cols': list(df.select_dtypes(exclude=[np.number]).columns),
        'missing_summary': df.isnull().sum().to_dict(),
        'total_missing': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'outlier_summary': {},
        'feature_stats': {},
        'target_stats': None
    }

    numeric_cols = report['numeric_cols']
    
    # Calculate IQR outliers and summary stats for each numeric column
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            report['outlier_summary'][col] = {
                'count': len(outliers),
                'lower_bound': round(float(lower_bound), 3),
                'upper_bound': round(float(upper_bound), 3),
                'pct': round(len(outliers) / len(series) * 100, 2)
            }

            report['feature_stats'][col] = {
                'min': round(float(series.min()), 3),
                'max': round(float(series.max()), 3),
                'mean': round(float(series.mean()), 3),
                'std': round(float(series.std()), 3),
                'median': round(float(series.median()), 3)
            }

    if target_col and target_col in df.columns and target_col in numeric_cols:
        t_series = df[target_col].dropna()
        report['target_stats'] = {
            'target_name': target_col,
            'count': len(t_series),
            'min': round(float(t_series.min()), 3),
            'max': round(float(t_series.max()), 3),
            'mean': round(float(t_series.mean()), 3),
            'std': round(float(t_series.std()), 3),
            'median': round(float(t_series.median()), 3)
        }

    return report

def clean_and_preprocess(df: pd.DataFrame, target_col: str, remove_duplicates: bool = True, impute_strategy: str = "median"):
    """
    Cleans dataset transparently and returns cleaned dataframe along with change logs.
    Does not silently drop data without documenting actions.
    """
    cleaned_df = df.copy()
    logs = []

    # 1. Filter to numeric columns only
    initial_cols = list(cleaned_df.columns)
    cleaned_df = cleaned_df.select_dtypes(include=[np.number])
    dropped_non_numeric = [c for c in initial_cols if c not in cleaned_df.columns]
    if dropped_non_numeric:
        logs.append(f"Excluded {len(dropped_non_numeric)} non-numeric columns: {', '.join(dropped_non_numeric)}")

    # 2. Check target presence
    if target_col not in cleaned_df.columns:
        raise ValueError(f"Target column '{target_col}' not found among numeric columns.")

    # 3. Handle duplicates
    dup_count = int(cleaned_df.duplicated().sum())
    if remove_duplicates and dup_count > 0:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        logs.append(f"Removed {dup_count} exact duplicate rows.")
    elif dup_count > 0:
        logs.append(f"Retained {dup_count} duplicate rows per user configuration.")

    # 4. Handle target missing values (target missing rows must be dropped)
    target_missing = int(cleaned_df[target_col].isnull().sum())
    if target_missing > 0:
        cleaned_df = cleaned_df.dropna(subset=[target_col]).reset_index(drop=True)
        logs.append(f"Dropped {target_missing} rows where target '{target_col}' was missing.")

    # 5. Handle feature missing values
    feature_cols = [c for c in cleaned_df.columns if c != target_col]
    imputed_count = 0
    for col in feature_cols:
        col_missing = int(cleaned_df[col].isnull().sum())
        if col_missing > 0:
            if impute_strategy == "median":
                fill_val = cleaned_df[col].median()
            elif impute_strategy == "mean":
                fill_val = cleaned_df[col].mean()
            else:
                fill_val = 0.0
            cleaned_df[col] = cleaned_df[col].fillna(fill_val)
            imputed_count += col_missing
            logs.append(f"Imputed {col_missing} missing values in '{col}' using {impute_strategy} ({fill_val:.2f}).")

    if not logs:
        logs.append("Dataset passed inspection with zero missing values or duplicates requiring modification.")

    return cleaned_df, logs

def prepare_train_test(cleaned_df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets using consistent seed.
    """
    feature_cols = [c for c in cleaned_df.columns if c != target_col]
    X = cleaned_df[feature_cols]
    y = cleaned_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    feature_bounds = {}
    for col in feature_cols:
        feature_bounds[col] = {
            'min': float(X_train[col].min()),
            'max': float(X_train[col].max()),
            'mean': float(X_train[col].mean()),
            'std': float(X_train[col].std()),
            'median': float(X_train[col].median())
        }

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols,
        'feature_bounds': feature_bounds,
        'target_name': target_col
    }

def compute_correlation_matrix(df: pd.DataFrame):
    """Compute Pearson correlation matrix for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    return num_df.corr().round(3)
