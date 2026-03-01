import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

# setting the page title
st.title('💸 Placement LPA predictor')

# different tabs
# tab 1: model info
# tab 2: user input predictions
tab1, tab2 = st.tabs(["ℹ️ Model info","🙍‍♂️ User Prediction"])


# all tab 1 content : All model and metadata
with tab1:
    df = pd.read_csv('placement.csv')
    st.markdown("### 🔰 Dataset Preview")
    st.caption('Random Tuples')
    st.dataframe(df.sample(5))

    st.markdown("### 🔎 Datset summary")
    st.caption('Statistical summary')
    st.dataframe(df.describe().T)
    st.write('---')

    # Linear Regression Model
    
    # features and target
    st.markdown("### 👉 Input and target")
    x = df.iloc[:, 0:1]
    y = df.iloc[:, -1]
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label='Feature',
            value='cgpa'      
            )
    with col2:
        st.metric(
            label='Target',
            value='package'

        )
    st.write('---')
    
    # data split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)

    # model train
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # pred
    y_pred = lr.predict(x_test)

    # hyperparameters and evaluation
    r2_sc = r2_score(y_test, y_pred)
    
    st.markdown('### 📊 Model Results')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(      # evaluation
            label='R2_Score',
            value=round(r2_sc,3)
        )
    
    with col2:
        st.metric(      # intercept
            label='Intercept',
            value=round(lr.intercept_, 3)
        )
    
    with col3:
        st.metric(      # slope
            label='Slope',
            value=round(lr.coef_[0],3)
        )
    
    # coeff of correlation
    correlation = np.corrcoef(x.values.flatten(),y.values)[0,1]
    with col4:
        st.metric(
            label='Correlation',
            value=f"{correlation:.3f}"

        )
    st.write('---')

    st.markdown('### 📈 Best fit Line')
    fig,ax = plt.subplots()
    ax.scatter(x,y,color = 'deepskyblue', s = 70, label = 'True data', edgecolors='navy', lw = 0.2)
    ax.plot(x_test, y_pred, color = 'k', label = 'best fit line', lw = 2)
    ax.set_xlabel('CGPA')
    ax.set_ylabel('PAckage(LPA)')
    ax.set_title('best fit line Plot')
    ax.legend()
    ax.grid(axis='both', ls = 'dotted')
    st.pyplot(fig)

with tab2:
    cgpa = st.number_input('Enter CGPA', min_value=0.00, max_value=10.00, step=0.01)
    st.info('ℹ️ Enter CGPA to predict the Approx LPA')
    
    if st.button('Predict LPA'):
        pred = lr.predict([[cgpa]])
        st.success('✅ Prediction complete')
        st.metric(
            label='Predicted LPA',
            value=round(pred[0],2)
        )


    

