# 🧠 Stroke Risk Prediction & Clinical Feature Engineering

An end-to-end Exploratory Data Analysis (EDA) and **Machine Learning-Ready Feature Engineering** project focused on uncovering core physiological stroke drivers, eliminating confounding sociodemographic noise (Simpson's Paradox), and engineering robust clinical risk scores.

---

## 📌 Executive Summary

Many naive models fail on health datasets by learning spurious correlations—such as attributing stroke risk to marital status or occupation. This project demonstrates a domain-informed Data Science workflow that transforms raw observational data into an interpretable, ML-ready feature space:

* **Simpson's Paradox Resolution:** Uncovered that sociodemographic features (`ever_married`, `work_type`, `Residence_type`) carry zero independent predictive power once stratified by Age.
* **Two-Tiered Risk Architecture:** Established that Age acts as the primary predisposing foundation (unlocking baseline risk after age 61), while Hypertension, Heart Disease, and High Glucose ($\ge 140\text{ mg/dL}$) serve as equal-weight direct clinical drivers.
* **Feature Engineering Strategy:** Developed composite clinical risk metrics (`clinical_risk_score`), total cumulative risk indexes, and explicit interaction variables (`Age * Condition`) designed to enhance model interpretability and handle class imbalance for downstream predictive algorithms.

---

## 📊 Key Analytical Insights & Data Artifacts

### 1. Age Predisposition vs. Direct Drivers (Tipping Point)
Age alone does not cause a stroke, but it acts as the primary predisposition factor that unlocks the risk window. Below age 45, stroke incidence is negligible ($< 1.9\%$). Beyond age 61, baseline risk jumps into the critical zone ($10.98\%$).

* **[0–17 Age]** : $0.00\%$ Risk *(Negligible)*
* **[18–30 Age]** : $0.21\%$ Risk *(Negligible)*
* **[31–45 Age]** : $1.92\%$ Risk *(Very Low)*
----------------------------------------------------------- *(45 Age Activation Boundary)*
* **[46–60 Age]** : $4.25\%$ Risk *(Predisposition Stage)*
* **[61+ Age]** : $10.98\%$ Risk *(Critical Zone – Baseline Unlocked)*

---

### 2. Clinical Condition Multipliers (Age 61+)
Once the risk window is unlocked by age, the three key health conditions act as direct drivers that push stroke risk from $\sim 5\%$ to over $20\%$:

| Condition Combination (Age 61+) | Stroke Risk Rate (%) | Role & Growth Multiplier |
| :--- | :--- | :--- |
| **Healthy Baseline** (No conditions) | $5.00\% - 6.00\%$ | Baseline Age Effect |
| **+ Hypertension** | $19.23\%$ | Direct Driver ($\sim 3.5\times$) |
| **+ Heart Disease** | $20.00\%$ | Direct Driver ($\sim 3.5\times$) |
| **+ High Glucose ($\ge 140\text{ mg/dL}$)** | $23.18\%$ | Direct Driver ($\sim 4.0\times$ with BMI) |

---

### 3. Cumulative Risk Stacking
When clinical drivers (High Glucose, Hypertension, Heart Disease, Smoking) stack on top of the age foundation, risk escalates non-linearly:

| Cumulative Risk Factors | Risk Escalation Pattern |
| :--- | :--- |
| **0 Risk Factors** | Baseline Risk |
| **1 Risk Factor** | Moderate Increase |
| **2 Risk Factors** | Double-Digit Threshold Crossed |
| **3 Risk Factors** | High-Risk Zone |
| **4+ Risk Factors** | Critical Risk (~1 in every 5 patients) |

---

## 🛠️ Feature Engineering Pipeline

The preprocessing script (`preprocess_stroke_data`) automates the following steps to construct a clean, ML-ready dataset:

1. **Feature Pruning:** Drops `ever_married`, `work_type`, and `Residence_type` to eliminate sociodemographic noise and prevent overfitting.
2. **Age Stratification:** Bins continuous age into discrete risk cohorts (`0-17`, `18-30`, `31-45`, `46-60`, `61+`).
3. **Diabetic Thresholding:** Flags continuous glucose at the clinical boundary ($\ge 140\text{ mg/dL}$).
4. **Composite Clinical Risk Score:** Computes an unweighted $0–3$ clinical risk index combining Hypertension, Heart Disease, and High Glucose.
5. **Total Cumulative Risk Score:** Combines clinical score, senior status ($\ge 61$), and smoking status (`total_cumulative_risk`).
6. **Explicit Interaction Variables:** Generates interaction terms (`age_x_hypertension`, `age_x_heart_disease`, `age_x_high_glucose`) to capture non-linear risk escalation for downstream modeling.

---

## 💻 Python Implementation

```python
import pandas as pd
import numpy as np

def preprocess_stroke_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses stroke data based on EDA findings:
    - Drops noisy sociodemographic features.
    - Captures the two-tiered structure (Age Predisposition x Clinical Drivers).
    - Constructs composite risk scores and interaction terms.
    """
    df = df.copy()

    # 1. Feature Pruning
    drop_cols = ['ever_married', 'work_type', 'Residence_type']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # 2. Age Cohorts (Predisposition Layers)
    bins = [0, 18, 31, 46, 61, np.inf]
    labels = ['0-17', '18-30', '31-45', '46-60', '61+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    # 3. High Glucose Threshold (Diabetic Boundary = 140 mg/dL)
    df['high_glucose'] = (df['avg_glucose_level'] >= 140).astype(int)

    # 4. Composite Clinical Risk Score (0 - 3)
    df['clinical_risk_score'] = (
        df['hypertension'].astype(int) + 
        df['heart_disease'].astype(int) + 
        df['high_glucose'].astype(int)
    )

    # 5. Total Cumulative Risk Score (Age Baseline + Smoking)
    is_senior_base = (df['age'] >= 61).astype(int)
    is_smoker = df['smoking_status'].isin(['formerly smoked', 'smokes']).astype(int)

    df['total_cumulative_risk'] = (
        df['clinical_risk_score'] + 
        is_senior_base + 
        is_smoker
    )

    # 6. Age x Driver Interactions
    df['age_x_hypertension'] = df['age'] * df['hypertension']
    df['age_x_heart_disease'] = df['age'] * df['heart_disease']
    df['age_x_high_glucose'] = df['age'] * df['high_glucose']

    return df
