# Student Mental Health Risk Prediction

A data science and machine learning project dedicated to analyzing and predicting student mental health risk levels based on academic, lifestyle, psychosocial, and demographic indicators.

---

## 📁 Project Structure

```bash
brief-croise/
├── data/
│   └── student_mental_health.csv       # Raw dataset (2,000 records, 17 columns)
├── notebooks/
│   └── 01_eda.ipynb                    # Phase 01: Exploratory Data Analysis (EDA)
├── outputs/
│   └── EDA_plots/                      # Generated visualization artifacts
│       ├── 01_Target_distribution.png
│       ├── 02_Num_Feats_distribution.png
│       ├── 03_Cat_Feats_distribution.png
│       ├── 04_Missing_Values.png
│       ├── 05_Outliers_exploration.png
│       ├── 06_Relation_COLvsCOL.png
│       ├── 07_Correlation_Matrix.png
│       ├── 08_Num_VS_Target.png
│       └── 09_Cat_VS_Target.png
├── app/                                # Application deployment code
├── dag/                                # Workflow / Airflow DAG definitions
├── src/                                # Reusable source modules
└── requirements.txt                    # Project dependencies
```

---

## 📊 Phase 01: Exploratory Data Analysis (EDA) Summary

### 1. Dataset Dimensions & Scope
- **Total Records:** 2,000 individual students.
- **Features:** 17 columns (10 numerical, 5 categorical, 1 identifier, 1 target).
- **Target Variable:** `Mental_Health_Risk` (Ordinal: `Low`, `Medium`, `High`).
- **Data Integrity:** **0 duplicate rows**, **2,000 unique `Student_ID`s**.

### 2. Target Variable Class Balance
The target variable is virtually balanced, ensuring that downstream classification algorithms will not require synthetic resampling:
- **Low Risk:** 660 students (**33.0%**)
- **Medium Risk:** 660 students (**33.0%**)
- **High Risk:** 680 students (**34.0%**)

### 3. Key Findings & Correlates with Mental Health Risk
- **Strongest Protective Factors (Negative Correlation with Risk):**
  - **`Sleep_Hours` ($r = -0.54$):** Sleep deficit is the primary correlate; High Risk averages 6.04 hrs/night vs. 7.52 hrs/night for Low Risk ($\Delta = -1.48$ hrs).
  - **`GPA` ($r = -0.35$):** High Risk students average a 2.42 GPA vs. 2.72 for Low Risk students.
  - **`Family_Support` ($r = -0.25$):** Elevated family support acts as a strong protective buffer.
  - **`Study_Hours_Per_Week` ($r = -0.24$):** Consistent academic investment correlates with reduced risk.
  - **`Physical_Activity` ($r = -0.23$):** Moderate-to-high exercise aligns with lower risk.
- **Strongest Risk Drivers (Positive Correlation with Risk):**
  - **`Financial_Stress` ($r = +0.31$):** Financial strain averages 6.69 in High Risk vs. 5.24 in Low Risk.
  - **`Previous_Mental_Health_History` ($r = +0.24$):** ~50% of students with prior mental health history are High Risk (vs. ~27% without).
  - **`Social_Media_Hours` ($r = +0.21$) & `Screen_Time_Hours` ($r = +0.17$):** Heavy digital exposure associates with elevated risk.
- **Neutral Features:**
  - **`Age` ($r = +0.005$)** and **`Gender` ($|r| < 0.02$)** exhibit near-zero correlation with risk.

---

## 🛠️ Actionable Roadmap for Phase 02 (Data Cleaning & Preprocessing)

Based on the empirical audit of the raw data, the following sequential pipeline is ready to be executed:

| Step | Objective | Details / Target Columns |
| :---: | :--- | :--- |
| **1** | **Primary Key Exclusion** | Drop `Student_ID` to prevent leakage. |
| **2** | **Logical Discrepancy Resolution** | Reconcile the **37 records (1.85%)** where `Social_Media_Hours` exceeds `Screen_Time_Hours` by capping social media to screen time. |
| **3** | **Missing Value Imputation** | Impute **419 missing entries** across 7 features (all < 5% missing):<br>• Median imputation for numerical features (`Financial_Stress`, `Sleep_Hours`, `Social_Media_Hours`, `Family_Support`, `Attendance_Rate`, `GPA`).<br>• Mode or separate category for `Diet_Quality` (69 missing). |
| **4** | **Outlier Mitigation** | Handle the **61 IQR outlier values** (`Sleep_Hours`, `Age`, `Attendance_Rate`) using `RobustScaler` or percentile Winsorization (1st/99th percentiles) rather than row deletion. |
| **5** | **Feature Encoding** | • **One-Hot Encoding:** `Gender`, `Part_Time_Job`, `Previous_Mental_Health_History`.<br>• **Ordinal Encoding:** `Physical_Activity` (Low=0, Moderate=1, High=2), `Diet_Quality` (Poor=0, Average=1, Good=2). |
| **6** | **Feature Scaling** | Apply `StandardScaler` or `RobustScaler` fitted strictly on training data. |
| **7** | **Stratified Splitting** | Partition into train/test sets preserving the 33/33/34% risk balance. |
