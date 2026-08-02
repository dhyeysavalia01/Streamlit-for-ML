import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import streamlit as st

st.title('🚀 Ridge Regression')
st.markdown(
    "### <span style='color:deepskyblue;'>Advertising sales prediction</span>",
    unsafe_allow_html=True,
)
st.markdown('> `advertising.csv`')

tab1, tab2 = st.tabs(["ℹ️  Model info","🚀  User input"])


# TAB 1  ->
# dataset preview
# featureand target
# intercept, coef, r2 score

with tab1:
    # dataset preview
    df = pd.read_csv('advertising.csv')
    st.markdown('#### 🔰 Dataset Preview')
    st.caption('First 5 rows:')
    st.dataframe(df.head())

    # dataset summary
    st.markdown('#### 🔍 Datset Summary')
    st.caption('Statistical summary')
    st.dataframe(df.describe().T)

    st.markdown('---')

    # fearturse and target
    st.markdown('#### 🧮 Features and Target')
    x = df.iloc[:,0:-1]
    y = df.iloc[:, -1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label = 'Feature 1', value = 'TV')

    with col2:
        st.metric(label = 'Feature 2', value = 'Radio')

    with col3:
        st.metric(label = 'Feature 3', value = 'Newspaper')

    with col4:
        st.metric(label = 'Target', value = 'Sales')

    st.markdown('---')

    # data split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
    # model train
    r = Ridge()
    r.fit(x_train, y_train)
    y_Pred = r.predict(x_test)

    st.markdown("#### ✅ Model Evaluation")
    col1, col2, col3 = st.columns(3)

    with col2:
        st.caption('Intercept')
        st.metric(label="",value = round(r.intercept_,2))

    with col1:
        st.caption('Coef(s)')
        st.dataframe(r.coef_)

    with col3:
        st.caption('R2 score')
        st.metric(label="", value = round(r2_score(y_test, y_Pred),2))

    st.write('---')

    # graph (plane of Regression)
    st.markdown('#### 🗺️ Plane Of Regression')
    x_range = np.linspace(df['TV'].min(), df['TV'].max(), 15)
    y_range = np.linspace(df['Radio'].min(), df['Radio'].max(), 15)

    xx,yy = np.meshgrid(x_range, y_range)
    news_mean = df['Newspaper'].mean()
    news_col = np.full(xx.ravel().shape, news_mean)

    pred_input = np.c_[xx.ravel(), yy.ravel(), news_col]
    zz = r.predict(pred_input).reshape(xx.shape)

    fig = px.scatter_3d(df, x = df['TV'],
                    y = df['Radio'],
                    z = df['Sales'],
                    color=df['Sales'],
                    color_continuous_scale='twilight')
    fig.add_surface(x = x_range, y = y_range, z = zz, colorscale='magma',showscale=False,opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # User input prediction
    st.markdown('### 🏁 User input Prediction')
    st.info('👇 Enter the following details...')
    st.write("")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        TV = st.number_input('📺 Enter TV budget:',min_value = 0.0, max_value = 300.4, step = 0.1)

    with col2:
        RADIO = st.number_input('📻 Enter Radio budget:',min_value = 0.0, max_value = 50.0, step = 0.1)

    with col3:
        NEWS = st.number_input('📰 Enter Newspaper budget:',min_value=0.0, max_value=115.0, step=0.1)

    st.write("")
    if st.button('PREDICT 🚀'):
        PRED = r.predict([[TV,RADIO,NEWS]])[0]
        st.success('✅ Prediction complete')
        st.metric(
            label = 'Predicted Sales',
            value = round(PRED,2)
        )



