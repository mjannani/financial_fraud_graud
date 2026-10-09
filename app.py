import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px

st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#07111f} [data-testid="stHeader"]{background:rgba(0,0,0,0)}
.block-container{padding-top:1.2rem}.hero{padding:22px;border-radius:18px;background:linear-gradient(135deg,#10243b,#0b1728);border:1px solid #203a57}
.hero h1{margin:0;color:#fff;font-size:34px}.hero p{color:#9fb3c8}.card{padding:18px;border-radius:16px;background:#0e1d2e;border:1px solid #203a57}.risk{font-size:38px;font-weight:800}.small{color:#9fb3c8;font-size:13px}.stButton>button{border-radius:10px;font-weight:700}
</style>""", unsafe_allow_html=True)

@st.cache_data
def demo_data(n=1200):
    rng=np.random.default_rng(42); types=rng.choice(['PAYMENT','TRANSFER','CASH_OUT','PURCHASE'],n,p=[.45,.22,.18,.15])
    amount=np.round(rng.lognormal(7,1,n),2); old=rng.uniform(100,200000,n); new=np.maximum(old-amount,0)
    hour=rng.integers(0,24,n); new_device=rng.binomial(1,.15,n); new_loc=rng.binomial(1,.12,n)
    risk=(amount>np.quantile(amount,.92)).astype(int)+(hour<5).astype(int)+new_device+new_loc
    fraud=(risk>=2).astype(int)
    return pd.DataFrame({'TransactionID':[f'TXN-{i:05d}' for i in range(n)],'Type':types,'Amount':amount,'OldBalance':old,'NewBalance':new,'Hour':hour,'NewDevice':new_device,'NewLocation':new_loc,'Fraud':fraud})

def prepare(df):
    d=df.copy();
    for c in d.columns:
        if d[c].dtype=='object': d[c]=LabelEncoder().fit_transform(d[c].astype(str))
    return d.replace([np.inf,-np.inf],0).fillna(0)

def score(df):
    d=df.copy(); numeric=prepare(d)
    target='Fraud' if 'Fraud' in numeric else None
    if target:
        X=numeric.drop(columns=[target]); y=numeric[target]
        if y.nunique()<2: y=None
    else: X=numeric; y=None
    if y is not None:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
        model=RandomForestClassifier(n_estimators=250,class_weight='balanced',random_state=42,n_jobs=-1); model.fit(Xtr,ytr)
        prob=model.predict_proba(X)[:,1]; metrics={'Accuracy':accuracy_score(yte,model.predict(Xte)),'Precision':precision_score(yte,model.predict(Xte),zero_division=0),'Recall':recall_score(yte,model.predict(Xte),zero_division=0),'F1':f1_score(yte,model.predict(Xte),zero_division=0)}
    else:
        model=None; prob=np.zeros(len(X)); metrics={}
    iso=IsolationForest(contamination=.08,random_state=42); anomaly=iso.fit_predict(X); anomaly_flag=(anomaly==-1).astype(int)
    if target is None: prob=np.clip(.25+anomaly_flag*.55,0,1)
    risk=np.clip(prob*70+anomaly_flag*30,0,100)
    out=d.copy(); out['Fraud Probability']=np.round(prob*100,1); out['Risk Score']=np.round(risk,1); out['Anomaly']=np.where(anomaly_flag==1,'Anomalous','Normal'); out['Decision']=np.select([risk>=75,risk>=45],['BLOCK / INVESTIGATE','MANUAL REVIEW'],'APPROVE'); return out,metrics

st.markdown('<div class="hero"><h1>🛡️ FraudGuard AI</h1><p>Advanced Financial Fraud Risk Intelligence & Investigation Dashboard</p></div>',unsafe_allow_html=True)
with st.sidebar:
    st.markdown('## ⚙️ Control Center')
    uploaded=st.file_uploader('Upload transaction CSV',type=['csv'])
    threshold=st.slider('High-risk threshold',50,95,75)
    page=st.radio('Navigation',['Overview','Transaction Scoring','Investigation Queue','Analytics','Model Performance'])
    st.caption('Works with your CSV. If no file is uploaded, a demo dataset is used.')

df=pd.read_csv(uploaded) if uploaded else demo_data();
# map common columns when available
rename={c:'Amount' for c in df.columns if c.lower() in ['amount','transaction_amount','amt']}
df=df.rename(columns=rename)
if 'Amount' not in df: df['Amount']=np.random.lognormal(7,1,len(df))
if 'Fraud' not in df: df['Fraud']=0
res,metrics=score(df); res['Decision']=np.select([res['Risk Score']>=threshold,res['Risk Score']>=45],['BLOCK / INVESTIGATE','MANUAL REVIEW'],'APPROVE')

if page=='Overview':
    total=len(res); fraud=int((res['Decision']=='BLOCK / INVESTIGATE').sum()); review=int((res['Decision']=='MANUAL REVIEW').sum()); amount=res['Amount'].sum()
    a,b,c,d=st.columns(4)
    a.metric('Transactions',f'{total:,}'); b.metric('High Risk',f'{fraud:,}'); c.metric('Manual Review',f'{review:,}'); d.metric('Transaction Value',f'₹{amount:,.0f}')
    st.markdown('### 📊 Risk Intelligence')
    c1,c2=st.columns(2)
    with c1:
        fig=px.pie(res,names='Decision',title='Decision Distribution',hole=.55,template='plotly_dark'); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.histogram(res,x='Risk Score',nbins=25,title='Risk Score Distribution',template='plotly_dark'); st.plotly_chart(fig,use_container_width=True)
    st.markdown('### 🚨 Top Risk Transactions')
    st.dataframe(res.sort_values('Risk Score',ascending=False).head(10),use_container_width=True,hide_index=True)

elif page=='Transaction Scoring':
    st.subheader('🔍 Transaction Risk Scoring')
    st.write('Upload a dataset to score every transaction, or inspect a transaction from the table below.')
    idx=st.number_input('Transaction row',0,max(0,len(res)-1),0)
    r=res.iloc[int(idx)]; scorev=float(r['Risk Score']);
    st.metric('Risk Score',f'{scorev:.1f}/100'); st.progress(int(scorev))
    st.write('**Decision:**',r['Decision']); st.write('**Fraud Probability:**',f"{r['Fraud Probability']:.1f}%")
    reasons=[]
    if float(r.get('Amount',0))>res['Amount'].quantile(.9): reasons.append('Unusually high transaction amount')
    if int(r.get('Hour',12))<5: reasons.append('Unusual transaction time')
    if int(r.get('NewDevice',0))==1: reasons.append('New device detected')
    if int(r.get('NewLocation',0))==1: reasons.append('New location detected')
    if r['Anomaly']=='Anomalous': reasons.append('Isolation Forest anomaly detected')
    st.markdown('### Why is it suspicious?'); [st.write('• '+x) for x in reasons or ['No major risk signal detected.']]
    st.download_button('⬇️ Download scored CSV',res.to_csv(index=False),'fraud_scored_transactions.csv','text/csv')

elif page=='Investigation Queue':
    st.subheader('🚨 Investigator Workbench')
    q=res[res['Decision']!='APPROVE'].sort_values('Risk Score',ascending=False)
    st.write(f'{len(q):,} transactions require attention.')
    st.dataframe(q,use_container_width=True,hide_index=True)
    st.download_button('⬇️ Export Investigation Queue',q.to_csv(index=False),'investigation_queue.csv','text/csv')

elif page=='Analytics':
    st.subheader('📈 Fraud Analytics')
    c1,c2=st.columns(2)
    with c1:
        fig=px.box(res,x='Decision',y='Amount',title='Amount by Decision',template='plotly_dark'); st.plotly_chart(fig,use_container_width=True)
    with c2:
        if 'Type' in res:
            g=res.groupby('Type',as_index=False)['Risk Score'].mean(); fig=px.bar(g,x='Type',y='Risk Score',title='Average Risk by Transaction Type',template='plotly_dark'); st.plotly_chart(fig,use_container_width=True)
    if 'Hour' in res:
        h=res.groupby('Hour',as_index=False)['Risk Score'].mean(); fig=px.line(h,x='Hour',y='Risk Score',markers=True,title='Risk by Hour',template='plotly_dark'); st.plotly_chart(fig,use_container_width=True)

else:
    st.subheader('🤖 Model Performance')
    if metrics:
        cols=st.columns(4)
        for col,(k,v) in zip(cols,metrics.items()): col.metric(k,f'{v:.2%}')
        st.info('Model: Random Forest + Isolation Forest. Risk score combines supervised fraud probability and unsupervised anomaly detection.')
    else: st.warning('Upload a CSV containing a Fraud/label column with at least two classes to calculate supervised metrics.')
