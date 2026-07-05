# Customer Churn Prediction

## Business Problem
Telecom companies lose significant revenue when customers churn (cancel service).
Identifying customers who are likely to churn *before* they leave lets a business
target them with retention offers. This project builds and compares three
classification models to predict churn probability from customer account data.

## Dataset
**Telco Customer Churn** — download from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the link above.
2. Rename it to `telco_churn.csv` and place it inside the `data/` folder.

## Approach
1. **Cleaning**: Fixed `TotalCharges` (loaded as text due to blank values for
   new customers), dropped the non-predictive `customerID` column.
2. **Encoding**: Label-encoded all categorical columns.
3. **Modeling**: Trained and compared three models —
   - Logistic Regression (baseline, interpretable)
   - Random Forest (handles non-linearity, feature importance)
   - XGBoost (gradient boosting, typically strongest on tabular data)
4. **Evaluation**: Since churn is an imbalanced problem (most customers don't
   churn), models are compared on **Precision, Recall, F1, and ROC-AUC** —
   not plain accuracy, which would be misleading here.
5. **Deployment**: Best model (by F1 score) is saved and served through a
   Streamlit web interface for interactive predictions.

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python train.py
```
This prints a comparison table of all three models and saves the best one
(plus the scaler and encoders) into the `models/` folder.

### 3. Launch the interface
```bash
streamlit run app.py
```
Opens a local web app where you can enter a customer's details and get a
churn probability.

## Deploy for Free (so you can put a live link on your resume)
1. Push this whole project folder to a GitHub repo (include the `models/`
   folder after running `train.py` once — Streamlit Cloud needs the trained
   model files, not just the code).
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app", select your repo and `app.py` as the entry point.
4. It deploys automatically and gives you a public URL — add that to your
   resume/portfolio next to this project.

## Results
After running `train.py`, see `models/model_comparison.csv` for the exact
Precision/Recall/F1/ROC-AUC of each model on your run — use these real numbers
in your resume bullet instead of a generic "achieved good accuracy" claim.

## Possible Extensions
- Add SHAP values to explain *why* a specific customer is predicted to churn
- Try an ANN (Keras) as a fourth model to directly compare ML vs DL on the
  same tabular problem
- Add a "batch prediction" mode to the Streamlit app (upload a CSV of many
  customers at once)
