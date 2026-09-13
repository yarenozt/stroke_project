# 🧠 Stroke Risk Prediction & Clinical Feature Engineering

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Exploratory Data Analysis (EDA) project focused on uncovering core physiological stroke drivers, eliminating confounding sociodemographic noise (Simpson's Paradox), and engineering clinical risk scores.

---

## 📌 Executive Summary

Many naive models fail on health datasets by learning spurious correlations—such as attributing stroke risk to marital status or occupation. This project demonstrates a **domain-informed data science workflow**:

1. **Simpson's Paradox Resolution:** Uncovered that sociodemographic features (`ever_married`, `work_type`, `Residence_type`) carry zero independent predictive power once stratified by **Age**.
2. **Two-Tiered Risk Architecture:** Established that **Age** acts as the primary predisposing foundation (unlocking baseline risk after age 61), while **Hypertension**, **Heart Disease**, and **High Glucose (>= 140 mg/dL)** serve as equal-weight direct clinical drivers.
3. **Feature Engineering:** Developed composite clinical risk metrics (`clinical_risk_score`) and interaction terms (`Age * Condition`) that significantly improve model interpretability and class-imbalanced prediction.

---

## 📊 Key Analytical Insights

### 1. Age Predisposition vs. Direct Drivers
Stroke risk follows a clear biological threshold. Below age 45, stroke incidence is negligible (< 1.9%). Beyond age 61, baseline risk jumps to **10.98%**.

[0-17 Age]   : 0.00% Risk (Negligible)
[18-30 Age]  : 0.21% Risk (Negligible)
[31-45 Age]  : 1.92% Risk (Very Low)
----------------------------------------------------------- (45 Age Activation Boundary)
[46-60 Age]  : 4.25% Risk (Predisposition Stage)
[61+ Age]    : 10.98% Risk (Critical Zone - Baseline Unlocked)

### 2. Clinical Condition Multipliers (Age 61+)
Once the risk window is unlocked by age, the three key health conditions push stroke risk from ~5% to over 20%:

| Condition Combo (Age 61+) | Stroke Risk Rate (%) | Role |
| :--- | :--- | :--- |
| **Healthy Baseline (No conditions)** | **5.00% - 6.00%** | Baseline Age Effect |
| **+ Hypertension** | **19.23%** | Direct Driver (~3.5x) |
| **+ Heart Disease** | **20.00%** | Direct Driver (~3.5x) |
| **+ High Glucose (>= 140 mg/dL) & BMI** | **23.18%** | Direct Driver (~4.0x) |

---

## 🛠️ Feature Engineering Pipeline

The preprocessing pipeline (`preprocess_stroke_data`) automates the following steps:

* **Feature Pruning:** Drops noisy features (`ever_married`, `work_type`, `Residence_type`) to prevent overfitting.
* **Glucose Thresholding:** Converts continuous glucose level into a binary indicator at the diabetic threshold (140 mg/dL).
* **Composite Risk Score:** Computes an unweighted 0-3 clinical risk index combining Hypertension, Heart Disease, and High Glucose.
* **Interaction Features:** Generates explicit interaction variables (`age_x_hypertension`, `age_x_heart_disease`, `age_x_high_glucose`) to aid tree-based and linear estimators.

---

## 🚀 Quick Start & Usage

### 1. Installation

git clone [https://github.com/your-username/stroke-prediction-ml.git](https://github.com/your-username/stroke-prediction-ml.git)
cd stroke-prediction-ml
pip install -r requirements.txt

### 2. Data Preprocessing Example

import pandas as pd
from src.preprocessing import preprocess_stroke_data

# Load raw dataset
raw_df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

# Process features based on clinical insights
df_processed = preprocess_stroke_data(raw_df)

print(f"Processed shape: {df_processed.shape}")

## 🛠️ Tech Stack & Tools

* **Data Manipulation:** Python, Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib
