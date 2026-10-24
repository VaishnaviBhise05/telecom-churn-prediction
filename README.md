# Telecom Customer Churn Prediction

Predicts which telecom customers are likely to churn so the retention team can act early.
AIDS Department PBL project.

## Dataset
IBM Telco Customer Churn (7,043 rows, 21 columns). Kaggle: `blastchar/telco-customer-churn`.
The notebook downloads it automatically; if offline, put `Telco-Customer-Churn.csv` beside the notebook.

## Pipeline
Problem definition → data understanding → cleaning → EDA + statistical tests → feature engineering →
Logistic Regression / Random Forest / Gradient Boosting (stratified 5-fold CV, grid search, threshold tuning) →
evaluation (recall, F1, ROC-AUC, confusion matrix) → business impact simulation → Streamlit app.

## Results (test set)
Churn recall ≈ 0.75, F1 ≈ 0.63, ROC-AUC ≈ 0.85.

## Run
```
pip install -r requirements.txt
jupyter notebook churn_project.ipynb     # run all cells (creates churn_model.pkl)
streamlit run app.py                     # live demo
```

## Team contributions
Vaishnavi Bhise :-	problem definition, data understanding, documentation

Srushti Jamdar  :-	preprocessing, EDA and visualisations

Pranjal Dhamane :- 	feature engineering, model building


