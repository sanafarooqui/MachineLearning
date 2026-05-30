
# Hospital Readmission Risk Prediction — Diabetic Patients

## Abstract
This repository contains an analysis and predictive modeling pipeline to identify diabetic patients at high risk of 30-day hospital readmission. The goal is to support targeted interventions at discharge to reduce avoidable readmissions, improve patient outcomes, and lower costs.

## Dataset
- Source: UCI Machine Learning Repository (Diabetes 130-US hospitals for years 1999-2008).
- Local copy: [data/diabetic_data.csv](data/diabetic_data.csv)
- Description: Electronic health record (EHR) data including demographics, admission details, diagnoses, lab results, and medication information covering 1999–2008 across multiple hospitals.

Reference: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

## Project Structure
- [hospital_readmissions.ipynb](hospital_readmissions.ipynb): Primary exploratory analysis, preprocessing, modeling, evaluation, and visualizations.
- [data/diabetic_data.csv](data/diabetic_data.csv): Raw dataset used for the analysis.
- [IDS_mapping.csv](data/IDS_mapping.csv): Supplementary mapping file.

## Goals and Scope
- Predict 30-day readmission (binary) for diabetic patients at discharge.
- Compare model families (logistic regression, tree-based, gradient boosting).
- Evaluate models on Precision, Recall and ROC-AUC and discuss clinical relevance.

## Methods

- Exploratory Data Analysis (EDA): distribution analysis, missingness patterns, class balance, and correlation analysis.

- Preprocessing and Feature Engineering:
	- Handle missing values using domain-aware imputation strategies.
	- Encode categorical variables (one-hot or ordinal where appropriate).
	- Create clinically meaningful features (e.g., comorbidity counts, medication counts).
	- Address class imbalance via resampling or class-weighted models.

- Models Evaluated:
	- Baseline: logistic regression, KNN, Decision Tree.
	- Tree-based: Random Forests.
	- Gradient boosting: XGBoost.
	- Calibration and probability smoothing as needed.

- Model Selection & Tuning:
	- Cross-validation (stratified) for hyperparameter tuning.
	- Grid/Randomized search for key hyperparameters.
    - SMOTE and under sampling for unbalanced data

## Evaluation
- Primary metrics: ROC AUC, Precision, Recall (Sensitivity), F1-score.
- Secondary metrics: Precision-Recall AUC, ROC AUC plots, confusion matrices at chosen thresholds.
- SHAP analysis
- Clinical considerations: trade-offs between sensitivity (catching high-risk patients) and precision (resource constraints for interventions).

## Results
All model training, full metric tables, and visualizations are provided inside [hospital_readmissions.ipynb](hospital_readmissions.ipynb). The notebook includes:
- Model performance summaries and comparison tables.
- ROC and Precision-Recall curves.
- Feature importance and SHAP (or permutation) explanations where applicable.

# Business Impact
**Patient Outcomes**\
At 0.43 threshold, the model catches 75% of 30-day readmissions before they happen — enabling targeted discharge interventions for 1,703 high-risk patients per 20,000 encounters who would otherwise go unidentified.\
**Operational Efficiency**\
Replaces inconsistent clinician judgment with a standardised, automated risk score applied at every discharge. Care coordinators can prioritise the highest-risk patients rather than spreading resources broadly — making finite capacity go further.\
**Explainability**\
SHAP analysis shows why each patient was flagged — prior inpatient visits, discharge destination, emergency history — giving clinicians actionable context, not just a score. This directly improves adoption and trust.\
**Key Caveat for Stakeholders**\
Precision is 16% at the recommended threshold — for every 6 alerts, 1 is a true readmission. This false alarm burden must be weighed against the cost of a missed readmission. The model is a decision-support tool, not an autonomous decision-maker, and requires retraining on recent data before clinical deployment.

## Limitations
- Possible missing or incorrectly recorded EHR fields.
- Temporal changes across years and hospitals may reduce generalizability.
- Readmission prediction is only one component; operational and ethical considerations must guide interventions.

## Reproducibility — How to Run
Recommended Python version: 3.8 or newer.

1) Create and activate a virtual environment (macOS / Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies (example set):

```bash
pip install -U pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab xgboost shap imbalanced-learn
```

3) Launch the analysis notebook:

```bash
jupyter lab hospital_readmissions.ipynb
```

4) Re-run notebook cells to reproduce figures and tables. If you prefer a script-based run, convert the notebook with `nbconvert` or extract relevant cells into a Python script.

## File Map (short)
- README.md — this document.
- [hospital_readmissions.ipynb](hospital_readmissions.ipynb) — notebook containing EDA, preprocessing, modeling, and results.
- data/ — folder containing raw CSV files used in the analysis.

## Next Steps and Extensions
- Incorporate external data (social determinants, post-discharge follow-up) for improved predictions.
- Deploy a calibrated model as a clinical decision support prototype with safety checks.
- Prospective validation and A/B testing to measure clinical impact.

## Contact
For questions or collaboration, open an issue in this repository or contact the author listed in the notebook metadata.

## License
Specify license here (e.g., MIT) if you choose to publish this repository publicly.

