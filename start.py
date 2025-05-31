import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model=tf.keras.models.load_model('churn_model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    one_hot=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

#streamlit app

st.title("cstomer_chrn prediction")

geography=st.selectbox("Geo",one_hot.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)

age=st.slider("age",18,92)

balance=st.number_input('Balance')
credit_score=st.number_input('credit score')
estimated_salary=st.number_input("salary")
tenure=st.slider('tenure',0,10)
nmofp=st.slider('nmpro',1,4)
has_cr=st.selectbox('has cc',[0,1])
is_active=st.selectbox("is active",[0,1])

#CreditScore	Geography	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary	Exited
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    "Gender" : [label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure': [tenure],
    'Balance' : [balance],
    'NumOfProducts':[nmofp],
    'HasCrCard' : [has_cr],
    'IsActiveMember' : [is_active],
    'EstimatedSalary': [estimated_salary]
})

geo=one_hot.transform([[geography]])
geo_df=pd.DataFrame(geo , columns=one_hot.get_feature_names_out(['Geography']))

inpt=pd.concat([input_data.reset_index(drop=True),geo_df],axis=1)

inpt_scaled=scaler.transform(inpt)

prediction=model.predict(inpt_scaled)
prediction_proba=prediction[0][0]

if prediction_proba > 0.5 :
    st.write('CHURN')
else :
    st.write('NOT CHURN')