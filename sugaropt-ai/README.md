# 🌾 SugarOpt AI — Sugarcane Recovery Prediction & Optimization System

> **AI-Powered Sugarcane Recovery Prediction & Decision-Support System**  
> *Transforming empirical cane agronomy and factory extraction data into actionable yield predictions, transparent explainability, and constrained scenario optimizations.*

---

## 1. Project Overview
**SugarOpt AI** is an intelligent decision-support software application designed for sugar mill managers, chemical control engineers, and agricultural procurement planners. Sugarcane recovery rate (the percentage of commercial white/raw sugar extracted per metric ton of cane) is the single most vital economic metric in sugar processing. 

SugarOpt AI bridges the gap between historical factory records and predictive intelligence by providing:
1. Multi-model regression benchmarking with rigorous cross-validation.
2. Real-time point prediction with defensible 95% statistical confidence intervals.
3. Transparent feature attribution using SHAP (SHapley Additive exPlanations) accompanied by plain-English, non-causal interpretations.
4. An interactive What-If Scenario Simulator for evaluating parameter trade-offs.
5. A bounded mathematical optimization engine (SciPy SLSQP / L-BFGS-B / Differential Evolution) to discover realistic parameter configurations that maximize predicted recovery.

---

## 2. Problem Statement
In conventional sugar mills and agro-industrial estates, estimating recovery rates and identifying optimal processing parameters typically relies on empirical rules of thumb, static laboratory tables, or isolated linear formulas (e.g., standard Winter-Carp-Brouwer formulas). 

However:
- Standard formulas often struggle to capture non-linear interactions between cane fiber, juice purity, moisture, extraneous trash, and weather patterns.
- Existing machine learning prototypes often operate as "black boxes," providing predictions without explanations or confidence bounds.
- Industrial teams lack an interactive simulation environment to test "What-If" operational adjustments before making harvest or milling commitments.

---

## 3. Proposed Solution
SugarOpt AI addresses these challenges through a modular, scientifically honest decision-support system:
- **No Over-Engineering:** Pure software application without unneeded IoT hardware, physical sensors, or cloud microservice overhead.
- **Scientific Honesty:** Clear demarcation between **Observed Data**, **Model Prediction**, **Model Explanation**, and **Model-Based Optimization**.
- **Transparent Benchmarking:** Linear models, Random Forests, Gradient Boosters, and XGBoost compete on identical held-out test splits. Complex models are never preferred unless they demonstrate genuine validation gains.
- **Safety Boundaries:** Explicit alerts whenever input parameters drift outside the empirical training distribution.

---

## 4. Key Features

- **📊 Industrial Dashboard:** Real-time KPI cards displaying predicted recovery, best-performing model, validation R², and training sample counts, alongside Plotly interactive distributions.
- **🔍 Comprehensive Data Quality Audit:** Automated detection of missing values, duplicate rows, numeric type verification, IQR-based outlier flagging, and Pearson correlation matrices.
- **⚖️ Fair Model Benchmarking:** Side-by-side comparison of **Linear Regression**, **Random Forest**, **Gradient Boosting**, and **XGBoost** evaluating $R^2$, MAE, RMSE, and MSE.
- **🎯 Bounded Point Prediction:** Real-time yield inference with out-of-distribution warnings and defensible 95% uncertainty intervals.
- **🔬 Model Explainability (SHAP):** Local SHAP contribution charts and global feature hierarchies paired with plain-English narratives free of causal overstatement.
- **🎛️ What-If Simulator:** Interactive sliders allowing process engineers to simulate parameter shifts and immediately visualize predicted recovery deltas.
- **⚙️ Bounded Optimization Engine:** SciPy-powered constrained optimization that identifies realistic parameter settings to maximize predicted recovery within user-defined physical boundaries.
- **🚀 One-Click Demo Mode:** Instant sample dataset loading and auto-training for rapid 3–5 minute demonstrations.

---

## 5. System Architecture

```text
sugaropt-ai/
│
├── app.py                     # Streamlit multi-page dashboard UI
├── requirements.txt           # Python dependencies
├── README.md                  # System documentation
│
├── data/
│   ├── sugarcane_data.csv     # Standard 4-feature baseline dataset
│   └── sample_data.csv        # Multi-parameter demo dataset
│
├── models/                    # Serialized joblib artifacts & metadata
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Data validation, quality audit & train-test split
│   ├── model_training.py      # Model training, benchmarking & serialization
│   ├── prediction.py          # Inference engine & range validation
│   ├── explainability.py      # SHAP attribution & plain-English commentary
│   ├── optimization.py        # SciPy constrained parameter optimization
│   ├── generate_datasets.py   # Dataset synthesis script
│   └── utils.py               # Formatting, constants & safety alerts
│
└── assets/
    └── style.css              # Agricultural-industrial theme styling
```

---

## 6. Technology Stack
- **Frontend / UI:** [Streamlit](https://streamlit.io/) (1.35+)
- **Interactive Visualizations:** [Plotly](https://plotly.com/) (5.15+)
- **Data Engineering:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)
- **Model Explainability:** [SHAP](https://shap.readthedocs.io/)
- **Mathematical Optimization:** [SciPy Optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- **Model Storage:** [Joblib](https://joblib.readthedocs.io/)

---

## 7. Dataset Format & Schema

The application accepts standard tabular CSV files containing numerical process parameters and a target recovery column.

### Standard Core Features:
| Column Name | Type | Typical Range | Description |
| :--- | :--- | :--- | :--- |
| **Sucrose** | Float | 10.0% – 17.5% | Pol % juice or cane; primary driver of crystallization yield |
| **Fiber** | Float | 9.5% – 16.5% | Cane dry fiber; impedes extraction and retains sucrose in bagasse |
| **Moisture** | Float | 64.0% – 75.0% | Inherent cane water content |
| **Rainfall** | Float | 500 – 1600 mm | Cumulative precipitation across crop maturation cycle |
| **Purity** *(Optional)* | Float | 75.0% – 92.0% | Apparent juice purity $(Pol / Brix \times 100)$ |
| **Trash_Content** *(Optional)* | Float | 1.5% – 12.0% | Extraneous leaves, tops, and soil adhering to cane |
| **Recovery_Rate** *(Target)* | Float | 8.0% – 13.5% | Commercial sugar recovered % cane |

---

## 8. Installation

1. Clone or navigate into the repository:
   ```bash
   cd sugaropt-ai
   ```

2. Ensure Python 3.10+ is installed. Then install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 9. How to Run

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

The system will start locally and open in your default browser at `http://localhost:8501`.

---

## 10. How the ML Pipeline Works

```text
CSV Ingestion
      │
      ▼
Data Quality Audit (Missing values, Duplicates, IQR Outlier checks)
      │
      ▼
Preprocessing & Imputation (Transparent median/mean imputation, duplicate removal)
      │
      ▼
Deterministic Train/Test Split (80/20 held-out evaluation)
      │
      ▼
Cross-Model Benchmark (Linear Regression, Random Forest, Gradient Boosting, XGBoost)
      │
      ▼
Validation Evaluation (R², MAE, RMSE, MSE)
      │
      ▼
Automated Selection of Generalization-Superior Regressor
      │
      ▼
Inference & 95% Confidence Interval Generation
```

---

## 11. How Optimization Works

The optimization engine implements constrained mathematical programming against the trained surrogate machine learning model:

$$\max_{\vec{x}} \quad \widehat{f}_{\text{ML}}(\vec{x})$$

$$\text{subject to} \quad \vec{x}_{\text{min}} \le \vec{x} \le \vec{x}_{\text{max}}$$

$$\vec{x}_{\text{fixed}} = \vec{x}_{\text{current}}$$

1. **Surrogate Function:** The objective function queries the active ML model $\widehat{f}(\vec{x})$.
2. **Bounds Specification:** Boundaries are automatically initialized from observed training distributions, with user customization controls to enforce agronomic limits.
3. **Controllability Toggles:** Fixed parameters (such as seasonal Rainfall or historical ambient temperatures) are locked in place.
4. **Solvers:** Implements SciPy `SLSQP` (Sequential Least Squares Programming) for smooth gradient navigation, `L-BFGS-B` for bounded quasi-Newton descent, and `differential_evolution` for global exploration.

---

## 12. Limitations & Operational Boundaries

- **Decision-Support Prototype:** SugarOpt AI does not provide real-time industrial PLC/DCS actuator control.
- **Model-Based Surrogate:** Optimizations reflect mathematical maxima of the trained statistical model. Real factory operations are subject to unmeasured variables (mill grooving wear, microbial dextran formation, boiler steam stability).
- **Out-of-Distribution Warning:** Extrapolations beyond training data ranges are flagged as potentially unreliable.
- **Non-Causal Interpretability:** SHAP values describe statistical associations learned from training records rather than direct physical causation.

---

## 13. Future Scope
- Integration with NIR (Near-Infrared) spectrometry tabular data streams.
- Multi-objective optimization balancing recovery rate against milling power consumption.
- Time-series modeling accounting for post-harvest cane staleness over crushing seasons.
- PDF executive report generation for daily shift handovers.

---

## 📜 Scientific Disclaimer
*SugarOpt AI is an AI-powered decision-support prototype. Predictions and optimization results represent statistical patterns learned from historical data and must be verified by factory chemical control staff and agronomists before operational implementation.*
