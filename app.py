import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#07111f}
[data-testid="stHeader"]{background:rgba(0,0,0,0)}
.block-container{padding-top:1.2rem}
.hero{padding:22px;border-radius:18px;background:linear-gradient(135deg,#10243b,#0b1728);border:1px solid #203a57}
.hero h1{margin:0;color:#fff;font-size:34px}.hero p{color:#9fb3c8}
div[data-testid="stMetric"]{background:#0e1d2e;border:1px solid #203a57;padding:16px;border-radius:14px}
.stButton>button{border-radius:10px;font-weight:700}
</style>""", unsafe_allow_html=True)

@st.cache_data
def demo_data(n=1200):
    rng=np.random.default_rng(42)
    types=rng.choice(['PAYMENT','TRANSFER','CASH_OUT','PURCHASE'],n,p=[.45,.22,.18,.15])
    amount=np.round(rng.lognormal(7,1,n),2)
    old=rng.uniform(100,200000,n)
    hour=rng.integers(0,24,n)
    new_device=rng.binomial(1,.15,n)
    new_loc=rng.binomial(1,.12,n)
    risk=(amount>np.quantile(amount,.92)).astype(int)+(hour<5).astype(int)+new_device+new_loc
    fraud=(risk>=2).astype(int)
    return pd.DataFrame({'TransactionID':[f'TXN-{i:05d}' for i in range(n)],'Type':types,'Amount':amount,'OldBalance':old,'NewBalance':np.maximum(old-amount,0),'Hour':hour,'NewDevice':new_device,'NewLocation':new_loc,'Fraud':fraud})

def find_target(df):
    lookup={str(c).strip().lower():c for c in df.columns}
    for name in ['fraud','is_fraud','isfraud','label','target','class','fraudulent','is_fraudulent']:
        if name in lookup:
            return lookup[name]
    return None

def normalise_target(series):
    if pd.api.types.is_numeric_dtype(series):
        vals=pd.to_numeric(series,errors='coerce').fillna(0)
        return (vals > 0).astype(int)
    s=series.astype(str).str.strip().str.lower()
    fraud_words={'1','true','yes','fraud','fraudulent','positive','chargeback'}
    legit_words={'0','false','no','genuine','legit','legitimate','normal','negative','non-fraud','nonfraud'}
    mapped=s.map(lambda v: 1 if v in fraud_words else (0 if v in legit_words else np.nan))
    if mapped.notna().all():
        return mapped.astype(int)
    unique=list(s.dropna().unique())
    if len(unique)==2:
        fraud_like=[v for v in unique if any(w in v for w in ['fraud','chargeback','positive'])]
        positive=fraud_like[0] if fraud_like else unique[-1]
        return (s==positive).astype(int)
    return None

def score(df, threshold):
    raw=df.copy()
    target_col=find_target(raw)
    y=None
    if target_col is not None:
        y=normalise_target(raw[target_col])
    # Drop target and identifier-like fields from predictors
    excluded=set()
    if target_col is not None: excluded.add(target_col)
    for c in raw.columns:
        cl=str(c).strip().lower().replace(' ','').replace('_','')
        if cl in {'transactionid','transaction_id','id','userid','customerid','accountid','referenceid','name'} or cl.endswith('uuid'):
            excluded.add(c)
    X=raw.drop(columns=list(excluded),errors='ignore').copy()
    # Make features numeric/categorical explicitly; keep IDs out of model.
    for c in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c]=X[c].astype(str)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c]=X[c].astype('string').fillna('Unknown').astype(str)
    numeric_cols=X.select_dtypes(include=['number','bool']).columns.tolist()
    categorical_cols=[c for c in X.columns if c not in numeric_cols]
    transformers=[]
    if numeric_cols:
        transformers.append(('num',Pipeline([('imputer',SimpleImputer(strategy='median'))]),numeric_cols))
    if categorical_cols:
        transformers.append(('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),categorical_cols))
    if not transformers:
        X=pd.DataFrame({'placeholder':np.zeros(len(raw))})
        numeric_cols=['placeholder']; categorical_cols=[]
        transformers=[('num',SimpleImputer(strategy='median'),numeric_cols)]
    pre=ColumnTransformer(transformers,remainder='drop')
    valid_y = y is not None and y.nunique()==2 and y.value_counts().min() >= 2
    metrics={}
    if valid_y:
        pipeline=Pipeline([('prep',pre),('model',RandomForestClassifier(n_estimators=220,class_weight='balanced',random_state=42,n_jobs=-1))])
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
        pipeline.fit(Xtr,ytr)
        prob=pipeline.predict_proba(X)[:,1]
        pred=pipeline.predict(Xte)
        metrics={'Accuracy':accuracy_score(yte,pred),'Precision':precision_score(yte,pred,zero_division=0),'Recall':recall_score(yte,pred,zero_division=0),'F1':f1_score(yte,pred,zero_division=0)}
        prediction_mode='Supervised Random Forest'
    else:
        transformed=pre.fit_transform(X)
        iso=IsolationForest(contamination=.08,random_state=42)
        anomaly_flag=(iso.fit_predict(transformed)==-1).astype(int)
        # Heuristic score is explicitly identified as anomaly-based, not trained fraud probability.
        prob=np.where(anomaly_flag==1,.78,.12)
        prediction_mode='Isolation Forest anomaly scoring'
    transformed=pre.fit_transform(X)
    anomaly_flag=(IsolationForest(contamination=.08,random_state=42).fit_predict(transformed)==-1).astype(int)
    risk=np.clip(prob*70+anomaly_flag*30,0,100)
    out=raw.copy()
    out['Fraud Probability']=np.round(prob*100,1)
    out['Risk Score']=np.round(risk,1)
    out['Anomaly']=np.where(anomaly_flag==1,'Anomalous','Normal')
    out['Decision']=np.select([risk>=threshold,risk>=45],['BLOCK / INVESTIGATE','MANUAL REVIEW'],'APPROVE')
    return out,metrics,prediction_mode,target_col

st.markdown('<div class="hero"><h1>🛡️ FraudGuard AI</h1><p>Advanced Financial Fraud Risk Intelligence & Investigation Dashboard</p></div>',unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    uploaded=st.file_uploader("Upload transaction CSV",type=["csv"])
    threshold=st.slider("High-risk threshold",50,95,75)
    page=st.radio("Navigation",["Overview","Transaction Scoring","Investigation Queue","Analytics","Model Performance"])
    st.caption("Upload your transaction CSV. A demo dataset is used when no file is uploaded.")

try:
    if uploaded:
        df=pd.read_csv(uploaded)

        # Remove duplicate column names from uploaded CSV.
        # Example: two columns named "Fraud_Label".
        df=df.loc[:, ~df.columns.duplicated(keep="first")]

        # Clean column-name whitespace.
        df.columns=df.columns.astype(str).str.strip()
    else:
        df=demo_data()
except Exception as e:
    st.error(f"Could not read the uploaded CSV: {e}")
    st.stop()

# Common transaction amount column names
rename={}
for c in df.columns:
    if str(c).strip().lower() in ["amount","transaction_amount","transaction amount","amt","amount_paid"]:
        rename[c]="Amount"
if rename:
    df=df.rename(columns=rename)
if "Amount" not in df.columns:
    df["Amount"]=0.0
df["Amount"]=pd.to_numeric(df["Amount"],errors="coerce").fillna(0)

if len(df)<1:
    st.error("The uploaded file has no transaction rows.")
    st.stop()
try:
    res,metrics,mode,target_col=score(df,threshold)
except Exception as e:
    st.error("The uploaded data could not be scored. Check that the CSV has transaction rows and consistent columns.")
    st.exception(e)
    st.stop()

if page=="Overview":
    total=len(res)
    high=int((res["Decision"]=="BLOCK / INVESTIGATE").sum())
    review=int((res["Decision"]=="MANUAL REVIEW").sum())
    amount=float(pd.to_numeric(res["Amount"],errors="coerce").fillna(0).sum())
    a,b,c,d=st.columns(4)
    a.metric("Transactions",f"{total:,}")
    b.metric("High Risk",f"{high:,}")
    c.metric("Manual Review",f"{review:,}")
    d.metric("Transaction Value",f"₹{amount:,.0f}")
    st.markdown("### 📊 Risk Intelligence")
    c1,c2=st.columns(2)
    with c1:
        fig=px.pie(res,names="Decision",title="Decision Distribution",hole=.55,template="plotly_dark")
        st.plotly_chart(fig,width='stretch')
    with c2:
        fig=px.histogram(res,x="Risk Score",nbins=25,title="Risk Score Distribution",template="plotly_dark")
        st.plotly_chart(fig,width='stretch')
    st.markdown("### 🚨 Top Risk Transactions")
    st.dataframe(res.sort_values("Risk Score",ascending=False).head(10),width='stretch',hide_index=True)

elif page=="Transaction Scoring":
    st.subheader("🔍 Transaction Risk Scoring")
    idx=st.number_input("Transaction row",min_value=0,max_value=max(0,len(res)-1),value=0,step=1)
    r=res.iloc[int(idx)]
    scorev=float(r["Risk Score"])
    a,b,c=st.columns(3)
    a.metric("Risk Score",f"{scorev:.1f}/100")
    b.metric("Fraud Score",f"{float(r['Fraud Probability']):.1f}%")
    c.metric("Decision",str(r["Decision"]))
    st.progress(int(scorev))
    reasons=[]
    if float(r.get("Amount",0))>res["Amount"].quantile(.9): reasons.append("Unusually high transaction amount")
    if "Hour" in r.index and pd.to_numeric(pd.Series([r.get("Hour")]),errors="coerce").fillna(12).iloc[0]<5: reasons.append("Unusual transaction time")
    if "NewDevice" in r.index and str(r.get("NewDevice")).strip().lower() in ["1","true","yes"]: reasons.append("New device indicator is active")
    if "NewLocation" in r.index and str(r.get("NewLocation")).strip().lower() in ["1","true","yes"]: reasons.append("New location indicator is active")
    if r["Anomaly"]=="Anomalous": reasons.append("Isolation Forest flagged an unusual feature pattern")
    st.markdown("### Why was it flagged?")
    for reason in reasons or ["No rule-based warning was identified for this row. Review the score alongside the source data."]:
        st.write("• "+reason)
    st.download_button("⬇️ Download scored CSV",res.to_csv(index=False).encode("utf-8"),"fraud_scored_transactions.csv","text/csv")

elif page=="Investigation Queue":
    st.subheader("🚨 Investigator Workbench")
    q=res[res["Decision"]!="APPROVE"].sort_values("Risk Score",ascending=False)
    st.write(f"{len(q):,} transactions require attention.")
    st.dataframe(q,width='stretch',hide_index=True)
    st.download_button("⬇️ Export Investigation Queue",q.to_csv(index=False).encode("utf-8"),"investigation_queue.csv","text/csv")

elif page == "Analytics":
    st.subheader("📈 Fraud Analytics")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(
            res,
            x="Decision",
            y="Amount",
            title="Amount by Decision",
            template="plotly_dark"
        )
        st.plotly_chart(fig, width='stretch')

    with c2:
        type_col = next(
            (
                c for c in res.columns
                if str(c).strip().lower()
                in ["type", "transaction_type", "transaction type"]
            ),
            None
        )

        if type_col:
            g = res.groupby(type_col, as_index=False)["Risk Score"].mean()

            fig = px.bar(
                g,
                x=type_col,
                y="Risk Score",
                title="Average Risk by Transaction Type",
                template="plotly_dark"
            )
            st.plotly_chart(fig, width='stretch')
        else:
            fig = px.histogram(
                res,
                x="Decision",
                title="Transactions by Decision",
                template="plotly_dark"
            )
            st.plotly_chart(fig, width='stretch')

    hour_col = next(
        (
            c for c in res.columns
            if str(c).strip().lower() in ["hour", "transaction_hour"]
        ),
        None
    )

    if hour_col:
        h = res.groupby(hour_col, as_index=False)["Risk Score"].mean()

        fig = px.line(
            h,
            x=hour_col,
            y="Risk Score",
            markers=True,
            title="Risk by Hour",
            template="plotly_dark"
        )
        st.plotly_chart(fig, width='stretch')

else:
    st.subheader("🤖 Model Performance")
    st.caption(f"Scoring method: {mode}")
    if metrics:
        cols=st.columns(4)
        for col,(k,v) in zip(cols,metrics.items()):
            col.metric(k,f"{v:.2%}")
        st.info("Metrics are measured on a held-out test split. Risk scores also incorporate anomaly flags.")
    else:
        st.warning("No usable two-class fraud label was detected. The app uses anomaly-based risk scoring instead; it does not claim supervised model accuracy.")
    if target_col:
        st.write(f"Detected target column: `{target_col}`")
    else:
        st.write("No fraud-label column detected. Add a labelled column such as Fraud / is_fraud to evaluate supervised model metrics.")
