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
    st.markdown('<h3 class="workbench-title">🚨 Investigator Workbench</h3>', unsafe_allow_html=True)
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
