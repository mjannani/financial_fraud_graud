# FraudGuard AI — Fixed Advanced Version

## Run locally
1. Extract this ZIP.
2. Open Command Prompt in this folder.
3. Install packages: `pip install -r requirements.txt`
4. Start the dashboard: `streamlit run app.py`

## Deploy to Streamlit Community Cloud
Push `app.py` and `requirements.txt` to your GitHub repository and set `app.py` as the main file.

## Fixes
- String transaction IDs such as `TXN-00586` are excluded from model features.
- Categorical columns are one-hot encoded rather than being passed to scikit-learn as raw strings.
- Fraud target columns are detected from common names.
- If a usable labelled fraud column is missing, anomaly-based scoring is used and the app does not display misleading supervised accuracy.
- Keeps the dark dashboard, risk scoring, queue, charts, and CSV exports.

Note: This is an educational decision-support demo, not a production payment-blocking system. Validate on representative labelled data before real-world use.
