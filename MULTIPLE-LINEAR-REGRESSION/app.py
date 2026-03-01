import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.markdown('# `💸 Income Prediction`')
st.markdown('> based on age and experience')

tab1, tab2 = st.tabs(['🔰 Model data', '🚀 Predictions'])


with tab1:
    st.markdown('### 📊 Dataset preview')
    st.caption('random rows')
    df = pd.read_csv('income.csv')
    st.dataframe(df.sample(5))

    st.markdown('### 🔎 Statistical summary')
    st.caption('statistics metadata')
    st.dataframe(df.describe().T)
    st.write('---')

    st.markdown('### 👉 Features and target')
    
    sub_col1, sub_col2, sub_col3 = st.columns(3)
    with sub_col1:
        st.markdown("""
            <div style = 'color:#FF4B4B;font-size:16px'>Feature 1</div>
            <div style = 'color:white;font-size:24px; font-weight:bold'>age</div>
        """, unsafe_allow_html=True)

    with sub_col2:
        st.markdown("""
            <div style = 'color:#FF4B4B;font-size:16px'>Feature 2</div>
            <div style = 'color:white;font-size:24px; font-weight:bold'>experience</div>
        """, unsafe_allow_html=True)
    
    with sub_col3:
        st.markdown("""
            <div style = 'color:#2cc755;font-size:16px'>Target</div>
            <div style = 'color:white;font-size:24px; font-weight:bold'>income</div>
        """, unsafe_allow_html=True)
    st.write('---')

    st.markdown('### 🚀 Model Insights')
    
    # feature and target
    x = df.iloc[:, 0:2]
    y = df.iloc[:, -1]
    # data split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)
    # model train
    mlr = LinearRegression()
    mlr.fit(x_train, y_train)
    y_pred = mlr.predict(x_test)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label='R2_score',
            value=round(r2_score(y_test,y_pred), 3)
        )
        
    with col2:
        st.metric(      # intercept
            label='Intercept',
            value=round(mlr.intercept_, 3)
        )
    
    # coefficients
    st.markdown('Coefficient Table')
    coef_df = pd.DataFrame({
    'Feature':x_train.columns,
    'Value':mlr.coef_
    })
    st.table(coef_df)

    # correlation matrix
    st.markdown('Correlation Matrix')
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), cmap = 'Blues',annot=True, fmt='.2f')
    st.pyplot(fig)
    st.write('---')

# SCATTER PLOT
    feat1 = df.columns[0]
    feat2 = df.columns[1]
    target = df.columns[2]
    st.markdown('### 📈 Scatter plot')
    st.caption('True data')

    fig2 = plt.subplot()
    fig2 = px.scatter_3d(df, x = feat1, y = feat2, z = target, color = target)
    st.plotly_chart(fig2, use_container_width=True)
    
with tab2:
    st.caption('Enter following values to predict income')
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age',min_value=0, max_value=100, step=1)
    with col2:
        xp = st.number_input('Experience', min_value=0.0, max_value=84.0, step = 0.5)
    
    # predict button
    if st.button('🚀 Predict Income'):
        # st.info('Predicting...')
        pred = mlr.predict([[age,xp]])
        st.success('✅ Done Prediction')
        st.balloons()
        st.metric(
            label='👇 Predicted Income',
            value=round(pred[0],0)
        )

st.write('---')
st.markdown("""
<div style='text-align: center; color: gray; font-size: 14px;'>
© <a href="https://www.linkedin.com/in/dhyey-savaliya-632bb4246/" target="_blank">
Dhyey Savaliya</a> • Built with Streamlit ❤️
</div>
""", unsafe_allow_html=True)