import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression



st.title('👾 Lasso Regression')
st.markdown('##### `📑 calories.csv`')
st.write("")

# tabs
tab1, tab2 = st.tabs(['🧮 Model info','😃 User input Prediction'])

with tab1:
    # dataset preview
    df = pd.read_csv('calories.csv')
    st.markdown('#### 🔍 Dataset preview')
    st.caption('First 5 rows')
    st.dataframe(df.head())

    # datset summary
    st.markdown('#### ℹ️ Data Summary')
    st.caption('Statistical summary')
    st.dataframe(df.describe().T)


    st.write("---")
    st.markdown('#### 🔰 Features and Target')

    # features and target
    col1, col2, col3, col4 = st.columns(4)

    
    with col1:
        st.metric(label = "Feature 1", value = 'age')

    with col2:
        st.metric(label = "Feature 2", value = 'exercise_min')

    with col3:
         st.metric(label = "Feature 3", value = 'heart_rate')

    with col4:
        st.metric(label = "Target", value = 'calories_burned')


    st.write("---")

    # model evaluation
    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

    l = Lasso(alpha = 0.01)
    l.fit(x_train,y_train)
    y_PRED = l.predict(x_test)


    st.markdown("#### 🚀 Model Evaluation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption('Coef')
        st.dataframe(l.coef_)

    with col2:
        st.caption('Intercept')
        st.metric(label = "",value = round(l.intercept_,2))

    with col3:
        st.caption('R2 score')
        st.metric(label = "",value = round(r2_score(y_test, y_PRED),2))


                  
    st.write("---")

    st.markdown('#### 📊 Plots')
    # plane of Regression
    x_range = np.linspace(df['age'].min(), df['age'].max(), 15)
    y_range = np.linspace(df['exercise_min'].min(), df['exercise_min'].max(), 15)

    xx,yy = np.meshgrid(x_range, y_range)
    heart_mean = df['heart_rate'].mean()
    heart_col = np.full(xx.ravel().shape, heart_mean)

    pred_input = np.c_[xx.ravel(), yy.ravel(), heart_col]
    zz = l.predict(pred_input).reshape(xx.shape)

    fig = px.scatter_3d(df, x = df['age'],
                        y = df['exercise_min'],
                        z = df['calories_burned'],
                        color=df['calories_burned'],
                        color_continuous_scale='thermal')
    fig.add_surface(x = x_range, y = y_range, z = zz, colorscale='thermal',showscale=False)
    st.plotly_chart(fig, use_container_width=True)


with tab2:
    # User input prediction
    st.markdown('### 🏁 User input Prediction')
    st.info('👇 Enter the following details...')
    st.write("")
    col1, col2, col3 = st.columns(3)


    with col1:
        age_ = st.number_input('😃 Enter age:',min_value = 0, max_value = 100, step = 1)

    with col2:
        exercise_min_ = st.number_input('🏃 Enter Exercise Min:',min_value = 0, max_value = 360, step = 1)

    with col3:
        heart_rate_ = st.number_input('🫀 Enter Heart rate',min_value=90.0, max_value=170.0, step=0.2)

        
    st.write("")
    if st.button('PREDICT 🚀'):
        PRED = l.predict([[age_, exercise_min_, heart_rate_]])[0]
        st.success('✅ Prediction complete')
        st.metric(
            label = 'Calories burned Prediction',
            value = round(PRED,2)
        )
    
        


