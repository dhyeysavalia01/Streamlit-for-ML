import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Batch Gradient Descent",
    layout="wide"
)

st.title('🚀 Batch gradient Descent')
st.caption('Student performance Index')

tab1, tab2 = st.tabs(['📊 Model insights', '👨 User prediction'])

with tab1: 
    st.markdown('### 📄 Dataset preview')
    st.caption('random rows')
    df = pd.read_csv('student.csv')
    df.pop('Extracurricular Activities')
    df.rename(columns = {
      'Hours Studied':'study_hrs',
      'Previous Scores':'prev_score',
      'Sleep Hours':'sleep_hrs',
      'Sample Question Papers Practiced':'ques_paper',
      'Performance Index':'perf_index'
        },inplace=True)
    st.dataframe(df.sample(5))

    st.markdown('### 🔰 Statistical summary')
    st.dataframe(df.describe().T)
    st.write('---')

    st.markdown('### 🔍 Feature and target')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('###### Features 👇')
        st.dataframe(df.columns[0:-1])
    with col2:
        st.markdown('###### Target 🎯')
        st.markdown("""
        <div style = 'color:white; font-size:30px'>perf_index</div>
        """,unsafe_allow_html=True)
    st.write('---')

    st.markdown('### 💀 Model metadata')
    # features and target
    x = df.iloc[:,0:4]
    y = df.iloc[:, -1]

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    # data split
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)

    # custom class
    class BGD:
        def __init__(self, learning_rate, epochs):
            self.coef_ = None
            self.intercept_ = None
            self.epochs = epochs
            self.n = learning_rate
            self.loss_list = []
        
        def fit(self, x_train, y_train):
            self.coef_ = np.ones(x_train.shape[1])
            self.intercept_ = 0

            for i in range(self.epochs):
                y_pred = self.intercept_ + np.dot(x_train, self.coef_)

                # udpate self.intercept_
                grad_intercept_ = (-2/x_train.shape[0]) * np.sum(y_train - y_pred)
                self.intercept_ -= (self.n * grad_intercept_)

                # update self.coef_
                grad_coef_ = (-2/x_train.shape[0]) * np.dot((y_train - y_pred), x_train)
                self.coef_ -= (self.n * grad_coef_)

                loss = np.mean((y_pred - y_train)**2)
                self.loss_list.append(loss)
                # print(f'step {i} | loss = {loss}')
            # print('final parameters...')
            # print('coef_ = ',self.coef_)
            # print('intercept_ = ',self.intercept_)
            return self.intercept_, self.coef_
        
        def predict(self,x_test):
            return self.intercept_ + np.dot(x_test, self.coef_)
        
    batch = BGD(0.01, 100)
    batch.fit(x_train, y_train)

    y_pred = batch.predict(x_test)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label='R2 Score',
            value=f'{r2:.3f}'
        )
    with col2:
        st.metric(
            label='Intercept',
            value=f'{batch.intercept_:.3f}'
        )
    st.write('Coefficients')
    col1, col2, col3, col4 = st.columns(4)
    st.write('')
    with col1:
        st.metric(
            label='study_hrs coef',
            value=f'{batch.coef_[0]:.3f}'
        )
    
    with col2:
        st.metric(
            label='prev_score coef',
            value=f'{batch.coef_[1]:.3f}'
        )
    
    with col3:
        st.metric(
            label='sleep_hrs coef',
            value=f'{batch.coef_[2]:.3f}'
        )
    
    with col4:
        st.metric(
            label='ques_paper coef',
            value=f'{batch.coef_[3]:.3f}'
        )
    st.write('---')
    st.markdown('### 💀 Loss curve')
    st.info('This loss curve is for learning rate: 0.01 and 100 epochs')
    fig,ax = plt.subplots()
    ax.plot(range(batch.epochs), batch.loss_list, color = 'navy', label = 'loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss - MSE')
    ax.set_title('Loss curve')
    ax.legend()
    ax.grid(axis = 'both', ls = 'dotted')
    st.pyplot(fig)


with tab2:
    st.markdown('### Model configuration')
    col1, col2 = st.columns(2)
    with col1:
        lr = st.slider('Learning rate', min_value=0.01, max_value=0.1, value = 0.01)
    with col2: 
        ep = st.number_input('Epochs', min_value=100, max_value=1000, value = 1000, step = 1)
    

    st.markdown('### Feature input')
    col1, col2 = st.columns(2)
    with col1:
        study_hrs = st.number_input('Daily Study hours', min_value=0.0, max_value=15.0, value = 4.0, step = 0.5)
        prev_scores = st.number_input('Previous Score', min_value=0, max_value=100, value = 70, step = 1)

    with col2:
        sleep_hrs = st.number_input('Daily Sleep hours', min_value=2.0, max_value=15.0, value = 7.0, step = 0.5)
        question_paper = st.number_input('Question paper solved', min_value=0, max_value=50, value = 5, step = 1)

    st.write('')
    if st.button('Predict performance Index'):
        st.info('Making Prediction...')
        batch2 = BGD(lr, ep)
        batch2.fit(x_train, y_train)
        user_input = np.array([[study_hrs,
                                prev_scores,
                                sleep_hrs,
                                question_paper]])

        # Normalize input
        user_input_norm = (user_input - x.mean().values) / x.std().values

        # Predict normalized
        y_pred_norm = batch2.predict(user_input_norm)

        # Denormalize output
        pred_real = (y_pred_norm * y.std()) + y.mean()

        st.success(f"✅ Predicted Performance Index: {pred_real[0]:.2f}")

