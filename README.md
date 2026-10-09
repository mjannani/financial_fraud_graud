# FraudGuard AI — Advanced Financial Fraud Detection

## Run on Windows
1. Install Python 3.10+
2. Open CMD in this folder
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

## Features
- Advanced dark dashboard UI
- CSV upload
- Random Forest supervised fraud detection
- Isolation Forest anomaly detection
- 0–100 risk score
- Configurable high-risk threshold
- Approve / Manual Review / Block decisions
- Investigation queue
- Risk explanations / reason codes
- Analytics charts
- Model metrics
- Download scored CSV and investigation queue

If your dataset has a target column with a different name, rename it to `Fraud` (0/1) for supervised model metrics.
