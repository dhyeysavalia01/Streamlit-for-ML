import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


st.markdown(
    '''
    <div style='color:darkorange;font-weight:Bold;font-size:70px;font-family:Arial'>
    💸 Salary Prediction
    </div>
    ''',unsafe_allow_html=True)

st.markdown(
    '''
    <div style='color:deepskyblue;font-weight:Bold;font-size:15px;font-family:JetBrains Mono'>
    ~ Gradient Descent
    </div>
    ''',unsafe_allow_html=True)

# line break
st.write('')

# tabs
tab1, tab2 = st.tabs(['Model Info', 'User Prediction'])

with tab1:
    st.markdown('## 👀 Dataset Preview')
    df = pd.read_csv('salary.csv')
    df.pop('S_no')
    st.dataframe(df.sample(5))

    st.markdown('## 📄 Statistical Summary')
    st.dataframe(df.describe().T)
    st.write('---')

    # Model insights
    # feature and target
    x = df.iloc[:, 0]
    y = df.iloc[:, -1]
    st.markdown('## 👉 Feature and Target')
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label='Feature',
            value='YearsExperience'
        )
    with col2:
        st.metric(
            label='Target',
            value='Salary'
        )

    # Save these BEFORE normalizing
    x_raw = df.iloc[:, 0]
    y_raw = df.iloc[:, -1]

    x_mean, x_std = x_raw.mean(), x_raw.std()
    y_mean, y_std = y_raw.mean(), y_raw.std()

    # Now perform normalization using these constants
    x = (x_raw - x_mean) / x_std
    y = (y_raw - y_mean) / y_std
    # split
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2)

    # custom class GD
    class GD:
        def __init__(self, learning_rate, epochs):
            self.m = None
            self.b = None
            self.n = learning_rate
            self.epochs = epochs
            self.loss_list = []
        
        def fit(self, x_train, y_train):
            # random init
            self.m = 1
            self.b = 0

            for i in range(self.epochs):
                y_pred = self.m * x_train + self.b
                # update m
                grad_m = (2/x_train.shape[0]) * np.sum((y_pred - y_train)*(x_train))
                self.m -= (self.n * grad_m)
                # update b
                grad_b = (2/x_train.shape[0]) * np.sum(y_pred - y_train)
                self.b -= (self.n * grad_b)

                loss = np.mean((y_pred - y_train)**2)
                self.loss_list.append(loss)
            
            return self.m, self.b
        
        def predict(self, x_test):
            return self.m * x_test + self.b
        
    gd = GD(0.1, 100)
    gd.fit(x_train, y_train)

    y_pred = gd.predict(x_test)
    st.write('')
    st.markdown('## 🚀 Model Evaluation and Parameters')
    r2_sc = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label='R2_Score',
            value=round(r2_sc,3)
        )
    with col2:
        st.metric(
            label='Slope',
            value=f'{gd.m:.3f}'
        )
    with col3:
        st.metric(
            label='Intercept',
            value=f'{gd.b:.3f}'
        )
    st.write('---')
    st.markdown('## 📈 Plots')
    col1, col2 = st.columns(2)
    with col1:
        fig,ax = plt.subplots()
        ax.scatter(x,y,color = 'deepskyblue', s = 70, label = 'True data')
        ax.plot(x_test, y_pred, color = 'k', lw = 2, label = 'Best fit Line')
        ax.set_xlabel('Experience')
        ax.set_ylabel('Salary')
        ax.set_title('best fit line')
        ax.legend()
        ax.grid(axis = 'both', ls = 'dotted')
        st.pyplot(fig)
    with col2:
        fig,ax = plt.subplots()
        ax.plot(range(gd.epochs), gd.loss_list, color = 'navy', lw = 2, label = 'Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Loss Curve')
        ax.legend()
        ax.grid(axis = 'both', ls = 'dotted')
        st.pyplot(fig)
        st.warning('This loss curve is for: n = 0.1 and 100 epochs')
        


with tab2:
    st.markdown('### Hyperparameters')
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('')
        lr = st.slider('Learning Rate', min_value=0.00, max_value = 0.10, value = 0.01)
    with col2:
        st.write('')
        ep = st.number_input('Epochs', min_value=0, max_value=1000, value = 1000)
    
    st.markdown('### Feature input')
    exp = st.number_input('Experience', min_value=0, max_value=84, value = 2, step = 1)

    exp_norm = (exp - x_mean) / x_std
    st.write('')
    if st.button('🚀.  PREDICT'):
        gd_2 = GD(lr, ep)
        gd_2.fit(x_train, y_train)
        y_pred_user = gd_2.predict(x_test)
        r2_user = r2_score(y_test, y_pred_user)
        st.success('✅ Model Trained')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label='slope',
                value=f'{gd_2.m:.3f}'
            )
        with col2:
            st.metric(
                label='intercept',
                value=f'{gd_2.b:.3f}'
            )
        with col3:
            st.metric(
                label = 'R2_score',
                value = f'{r2_user:.3f}'
            )
        # st.markdown('#### Prediction')
        salary_pred_norm = gd_2.predict(np.array([exp_norm]))[0]

        # de-normalize salary
        salary_real = (salary_pred_norm * y_std) + y_mean

        st.metric(label="Predicted Salary", value=f"{salary_real:,.2f}")
        st.write('---')
        st.markdown('#### Loss Curve')
        fig,ax = plt.subplots()
        ax.plot(range(gd_2.epochs), gd_2.loss_list, color = 'coral', lw = 2, label = 'Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Loss Curve')
        ax.legend()
        ax.grid(axis = 'both', ls = 'dotted')
        st.pyplot(fig)
    


    


    


    

    


    

