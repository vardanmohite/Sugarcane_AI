"""
Script to generate realistic baseline and demo sugarcane processing datasets
Reflecting real-world agronomic and mill processing physics:
- Sucrose: 11.0% - 16.5% (primary recovery driver)
- Fiber: 10.5% - 16.0% (impedes milling extraction, bagasse loss)
- Moisture: 65.0% - 74.0%
- Rainfall: 600 - 1500 mm (moderate is good, excess late rains cause dilution)
- Purity (in extended dataset): 78% - 90%
- Trash_Content (in extended dataset): 2% - 10%
- Recovery_Rate: realistically 8.5% - 12.0%
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n_samples = 450

# 1. Generate 4-feature standard dataset: Fiber, Moisture, Rainfall, Sucrose -> Recovery_Rate
sucrose_4 = np.random.normal(13.8, 1.2, n_samples).clip(10.5, 17.0)
fiber_4 = np.random.normal(13.2, 1.1, n_samples).clip(10.0, 16.5)
moisture_4 = np.random.normal(69.5, 2.2, n_samples).clip(64.0, 75.0)
rainfall_4 = np.random.normal(950, 220, n_samples).clip(500, 1600)

# Real-world inspired recovery approximation with natural mill variance & noise:
# Base recovery ~ 0.70 * Sucrose - 0.12 * Fiber - 0.03 * Moisture - 0.0006 * Rainfall + constant + noise
recovery_4 = (
    0.68 * sucrose_4
    - 0.11 * fiber_4
    - 0.035 * (moisture_4 - 68.0)
    - 0.0005 * (rainfall_4 - 900)
    + 1.85
    + np.random.normal(0, 0.45, n_samples)
).round(2).clip(7.5, 13.5)

df_4 = pd.DataFrame({
    'Fiber': fiber_4.round(2),
    'Moisture': moisture_4.round(2),
    'Rainfall': rainfall_4.round(1),
    'Sucrose': sucrose_4.round(2),
    'Recovery_Rate': recovery_4
})

# Add a few realistic duplicate rows or small noise to test data quality section
df_4 = pd.concat([df_4, df_4.iloc[10:13]], ignore_index=True)

out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(out_dir, exist_ok=True)
df_4.to_csv(os.path.join(out_dir, 'sugarcane_data.csv'), index=False)
print(f"Generated sugarcane_data.csv with shape {df_4.shape}")

# 2. Generate extended sample dataset: Fiber, Moisture, Rainfall, Sucrose, Purity, Trash_Content -> Recovery_Rate
n_demo = 600
sucrose_demo = np.random.normal(14.0, 1.25, n_demo).clip(10.8, 17.2)
fiber_demo = np.random.normal(13.0, 1.15, n_demo).clip(9.8, 16.5)
moisture_demo = np.random.normal(69.2, 2.0, n_demo).clip(64.5, 74.5)
rainfall_demo = np.random.normal(920, 200, n_demo).clip(500, 1550)
purity_demo = (72.0 + 1.2 * sucrose_demo - 0.3 * fiber_demo + np.random.normal(0, 1.8, n_demo)).round(2).clip(75.0, 91.5)
trash_demo = np.random.normal(5.5, 1.8, n_demo).clip(1.5, 12.0)

recovery_demo = (
    0.55 * sucrose_demo
    + 0.08 * (purity_demo - 80.0)
    - 0.09 * fiber_demo
    - 0.05 * trash_demo
    - 0.025 * (moisture_demo - 68.0)
    - 0.0004 * (rainfall_demo - 900)
    + 2.10
    + np.random.normal(0, 0.38, n_demo)
).round(2).clip(7.8, 13.8)

df_demo = pd.DataFrame({
    'Sucrose': sucrose_demo.round(2),
    'Fiber': fiber_demo.round(2),
    'Moisture': moisture_demo.round(2),
    'Rainfall': rainfall_demo.round(1),
    'Purity': purity_demo.round(2),
    'Trash_Content': trash_demo.round(2),
    'Recovery_Rate': recovery_demo
})

# Add 2 duplicates and 2 subtle missing values to demonstrate data cleaning transparently
df_demo = pd.concat([df_demo, df_demo.iloc[25:27]], ignore_index=True)
df_demo.loc[5, 'Rainfall'] = np.nan
df_demo.loc[12, 'Fiber'] = np.nan

df_demo.to_csv(os.path.join(out_dir, 'sample_data.csv'), index=False)
print(f"Generated sample_data.csv with shape {df_demo.shape}")
