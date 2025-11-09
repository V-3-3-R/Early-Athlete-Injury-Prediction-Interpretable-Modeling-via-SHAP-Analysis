# 🏋️‍♂️ Early Athlete Injury Prediction (EAIP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Colab%20%7C%20Jupyter-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📘 Overview
This **machine learning capstone project** focuses on **early prediction of athlete injuries** using a **synthetic dataset**.  
The aim is to identify **high-risk athletes** based on demographic, training, recovery, and performance data — enabling coaches and medical staff to take **preventive action** before injuries occur.

### 🔍 Key Highlights
- **Problem Addressed:** Imbalanced binary classification (*No Injury* vs. *Injury*).  
- **Approach:** Full ML pipeline — **EDA**, **SMOTE balancing**, **ensemble modeling** (Logistic Regression, Random Forest, XGBoost, Stacking Classifier), **interpretability** via SHAP, and **robustness testing** (label noise + jitter).  
- **Performance:**  
  ✅ **98.77% CV accuracy**  
  ✅ **0.99 ROC-AUC**  
  ✅ **Realistic probabilities** through regularization & noise.  
- **Impact:** Deployable injury risk prediction model for real-time use in sports analytics.  

This project demonstrates **end-to-end ML best practices** — from **synthetic data generation** to **explainable model evaluation**.

---

## 📊 Dataset

A **synthetic dataset** is used to simulate real-world athlete data while maintaining privacy and reproducibility.  
Generated dataset: **`synthetic_athlete_injury_dataset.csv`** (6,000 rows × 16 columns) derived from a **200-row seed dataset** using an augmentation pipeline.

---

### 🧬 Data Generation Method
(Implemented in `EAIP.py` → Data Preparation section)

#### Steps:
1. **Seed Data:**  
   Start with `seed_200.csv` containing features like `Age`, `Gender`, `Position`, `Training_Intensity`.

2. **Synthetic Expansion:**
   - Apply **SMOTE** to oversample the minority (Injury) class → ~3,000 rows.  
   - Add **Gaussian jitter** (~5% std dev) to numeric columns (`Height_cm`, `Fatigue_Score`, etc.).  
   - Perform **bootstrap resampling** for categorical variation and derived risk metrics.

3. **Final Scaling:**  
   Repeat oversampling + jitter until 6,000 rows (~10–15% injury rate).

4. **Jittered Variant:**  
   `synthetic_athlete_injury_dataset_jitter.csv` introduces **2% label noise** for robustness evaluation.

---

### 🧾 Code Snippet (Data Generation)
Run this in **Google Colab** or **Jupyter Notebook**:

```python
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder

# Load 200-row seed dataset
seed_df = pd.read_csv('seed_200.csv')

# Encode categorical variables
encoder = OneHotEncoder(sparse_output=False, drop='first')
seed_encoded = pd.DataFrame(
    encoder.fit_transform(seed_df[['Gender', 'Position']]),
    columns=encoder.get_feature_names_out(['Gender', 'Position']),
    index=seed_df.index
)

seed_X = pd.concat([seed_df.drop(['Gender', 'Position', 'Injury_Indicator'], axis=1), seed_encoded], axis=1)
seed_y = seed_df['Injury_Indicator']

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(seed_X, seed_y)

# Add jitter (Gaussian noise)
numerical_cols = ['Age', 'Height_cm', 'Weight_kg', 'Training_Intensity']
for col in numerical_cols:
    X_res[col] += np.random.normal(0, 0.05 * X_res[col].std(), len(X_res))

# Combine and save
df_synthetic = pd.DataFrame(X_res)
df_synthetic['Injury_Indicator'] = y_res
df_synthetic.to_csv('synthetic_athlete_injury_dataset.csv', index=False)

# Jitter variant (flip 2% labels)
flip_idx = np.random.choice(df_synthetic.index, size=int(0.02 * len(df_synthetic)), replace=False)
df_jitter = df_synthetic.copy()
df_jitter.loc[flip_idx, 'Injury_Indicator'] = 1 - df_jitter.loc[flip_idx, 'Injury_Indicator']
df_jitter.to_csv('synthetic_athlete_injury_dataset_jitter.csv', index=False)
```

# DATASET STRUCTURE

| Column                   | Type  | Description                             | Example/Range          |
| ------------------------ | ----- | --------------------------------------- | ---------------------- |
| Athlete_ID               | int   | Unique identifier                       | 1–6000                 |
| Age                      | float | Athlete’s age                           | 18–35                  |
| Gender                   | str   | Male/Female                             | Male                   |
| Height_cm                | float | Height in cm                            | 160–200                |
| Weight_kg                | float | Weight in kg                            | 50–100                 |
| Position                 | str   | Playing position                        | Guard, Forward, Center |
| Training_Intensity       | float | Weekly training intensity (1–10)        | 1–10                   |
| Training_Hours_Per_Week  | float | Hours trained per week                  | 5–25                   |
| Recovery_Days_Per_Week   | int   | Rest/recovery days                      | 0–7                    |
| Match_Count_Per_Week     | int   | Matches played per week                 | 0–5                    |
| Rest_Between_Events_Days | int   | Days rest between events                | 1–7                    |
| Fatigue_Score            | float | Fatigue level (0–100)                   | 0–100                  |
| Performance_Score        | float | Recent performance (0–100)              | 0–100                  |
| Team_Contribution_Score  | float | Team impact (0–100)                     | 0–100                  |
| Load_Balance_Score       | float | Training load balance (0–100)           | 0–100                  |
| ACL_Risk_Score           | float | ACL injury risk (0–100)                 | 0–100                  |
| Injury_Indicator         | int   | Target variable (0=No Injury, 1=Injury) | ~15% injuries          |


# ⚙️ Installation & Setup
# 1️⃣ Clone the Repository
git clone https://github.com/V-3-3-R/Early-Athlete-Injury-Prediction-Interpretable-Modeling-via-SHAP-Analysis.git
cd Early-Athlete-Injury-Prediction-Interpretable-Modeling-via-SHAP-Analysis

# 2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3️⃣ Install Dependencies
pip install -r requirements.txt


# requirements.txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
shap>=0.42.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter

# 4️⃣ Generate Dataset (Optional)
If dataset files are missing:
python generate_data.py

# 5️⃣ Launch Jupyter Notebook
jupyter notebook EAIP.ipynb

# 🚀 Usage
# ▶️ Notebook Mode (Recommended)
Open EAIP.ipynb and run all cells sequentially:
EDA → Data Balancing → Modeling → Evaluation → Prediction
Outputs include:
Histograms, Heatmaps, and ROC Curves
Classification Metrics
SHAP Visualizations for model interpretability
⚡ Script Mode
To run the full pipeline:
python EAIP.py
Outputs (ROC curves, SHAP plots, accuracy reports) will be automatically saved.

# 🧠 Making Predictions
Use the trained model to predict injury risk for a new athlete profile:
```
import pandas as pd

new_athlete = {
    'Age': 23, 'Gender': 'Male', 'Height_cm': 185, 'Weight_kg': 80,
    'Position': 'Guard', 'Training_Intensity': 8, 'Training_Hours_Per_Week': 15,
    'Recovery_Days_Per_Week': 1, 'Match_Count_Per_Week': 3,
    'Rest_Between_Events_Days': 1, 'Fatigue_Score': 85,
    'Performance_Score': 70, 'Team_Contribution_Score': 90,
    'Load_Balance_Score': 40, 'ACL_Risk_Score': 95
}

new_df = pd.DataFrame([new_athlete])
categorical_features = ['Gender', 'Position']
new_encoded = pd.get_dummies(new_df, columns=categorical_features, drop_first=True)
new_final = new_encoded.reindex(columns=X_train.columns, fill_value=0)

prediction = stacking_clf_restricted.predict(new_final)
proba = stacking_clf_restricted.predict_proba(new_final)[:, 1]
print(f"Predicted Injury: {prediction[0]} (Probability: {proba[0]:.4f})")
```

Example Output:
Predicted Injury: 1 (Probability: 0.7088)

# 📈 Results & Evaluation
| Model                   | CV Accuracy | ROC-AUC  | Key Strength            |
| ----------------------- | ----------- | -------- | ----------------------- |
| Logistic Regression     | 0.92        | 0.95     | Fast, interpretable     |
| Random Forest           | 0.96        | 0.98     | Handles non-linearity   |
| XGBoost                 | 0.97        | 0.98     | Gradient boosting power |
| **Stacking Classifier** | **0.9877**  | **0.99** | Ensemble robustness     |

# 🔬 Insights
High Correlation: ACL_Risk_Score ↔ Injury_Indicator (r ≈ 0.85).
Injury Class Imbalance: ~15% injury cases.
Robustness: 2% label noise → Accuracy drops to ~0.95 (prevents overfitting).
Feature Importance (SHAP):
ACL_Risk_Score → 0.35
Fatigue_Score → 0.22
Training_Hours_Per_Week → 0.15

# 🖼️ Visual Outputs
(Placeholder images – generated in notebook execution)
final_roc_curve.png
SHAP Summary Plot
Feature Importance Bar Chart


# 🧭 Future Work
Real Data Integration: Pull real-time injury stats (e.g., NBA/MLS APIs).
Deployment: Streamlit or Flask app for live athlete risk dashboards.
Advanced Models: LSTM-based models for time-series injury patterns.
Extensions: Multi-label injury prediction (ACL, ankle, hamstring).
Fairness: Analyze bias across gender, position, and age groups.

# 🤝 Contributing
Contributions are always welcome!
Fork the repository
Create a new branch
Commit changes
Submit a Pull Request
Please open issues for bug reports or feature requests.

# 📜 License
This project is licensed under the MIT License.
See the LICENSE file for details.

# 💬 Acknowledgements
Built with ❤️ by Veer Javadia
for advancing Sports Analytics with AI & Machine Learning.
“Preventing one injury is worth a thousand recoveries.”
