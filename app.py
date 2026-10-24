import joblib, pandas as pd, streamlit as st

art = joblib.load('churn_model.pkl')
model, th = art['model'], art['threshold']

st.title('Telecom Customer Churn Predictor')
c1, c2 = st.columns(2)
with c1:
    SeniorCitizen = int(st.checkbox('Senior citizen'))
    Partner = st.selectbox('Partner', ['Yes', 'No'])
    Dependents = st.selectbox('Dependents', ['No', 'Yes'])
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    MultipleLines = st.selectbox('Multiple lines', ['No', 'Yes'])
    InternetService = st.selectbox('Internet service', ['Fiber optic', 'DSL', 'No'])
    Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
with c2:
    OnlineSecurity = st.selectbox('Online security', ['No', 'Yes'])
    OnlineBackup = st.selectbox('Online backup', ['No', 'Yes'])
    DeviceProtection = st.selectbox('Device protection', ['No', 'Yes'])
    TechSupport = st.selectbox('Tech support', ['No', 'Yes'])
    StreamingTV = st.selectbox('Streaming TV', ['No', 'Yes'])
    StreamingMovies = st.selectbox('Streaming movies', ['No', 'Yes'])
    PaperlessBilling = st.selectbox('Paperless billing', ['Yes', 'No'])
    PaymentMethod = st.selectbox('Payment method', ['Electronic check', 'Mailed check',
                                 'Bank transfer (automatic)', 'Credit card (automatic)'])
MonthlyCharges = st.number_input('Monthly charges', 18.0, 120.0, 70.0)
TotalCharges = st.number_input('Total charges', 0.0, 9000.0, float(MonthlyCharges * max(tenure, 1)))

if st.button('Predict'):
    d = dict(SeniorCitizen=SeniorCitizen, Partner=Partner, Dependents=Dependents, tenure=tenure,
             MultipleLines=MultipleLines, InternetService=InternetService, OnlineSecurity=OnlineSecurity,
             OnlineBackup=OnlineBackup, DeviceProtection=DeviceProtection, TechSupport=TechSupport,
             StreamingTV=StreamingTV, StreamingMovies=StreamingMovies, Contract=Contract,
             PaperlessBilling=PaperlessBilling, PaymentMethod=PaymentMethod,
             MonthlyCharges=MonthlyCharges, TotalCharges=TotalCharges)
    X = pd.DataFrame([d])
    # same feature engineering as the notebook
    X['tenure_group'] = pd.cut(X.tenure, [-1, 12, 24, 48, 72], labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']).astype(str)
    addons = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    X['num_addons'] = (X[addons] == 'Yes').sum(axis=1)
    X['avg_monthly_spend'] = X.TotalCharges / X.tenure.replace(0, 1)
    X['charge_gap'] = X.MonthlyCharges - X.avg_monthly_spend
    X['autopay'] = X.PaymentMethod.str.contains('automatic').astype(int)
    p = model.predict_proba(X)[0, 1]
    st.metric('Churn probability', f'{p:.1%}')
    if p >= th:
        st.error('HIGH RISK: offer a retention deal (e.g., contract upgrade discount).')
    else:
        st.success('Low risk: customer likely to stay.')
