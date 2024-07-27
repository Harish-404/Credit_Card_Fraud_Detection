import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

df = pd.read_csv('creditard.csv')

# separate legitimate and fraudulant transactions
legit = df[df('Class')==0]
fraud = df[df('Class')==1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit(n=len(fraud), random_state=2)
df = pd.concat([legit_sample,fraud],axis=0)

# split data into training and testing sets
X = df.drop(columns="Class",axis=1)
Y = df["Class"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)

# train logistic regression model
model = LogisticRegression
model.fit(X_train, Y_train)
# evaluate model performance
train_acc = accuracy_score(model.predict(X_train),Y_train)
test_acc = accuracy_score(model.predict(X_test),Y_test)

#web app
st.title("Credit Card Fraud Detection Model")
input_df = st.text_input('Enter All Required Features Values')
input_df_splited = df(',')

submit = st.buttom("Submit")

if submit:
    np_df = np.asarray(input_df_splited,dtype=np.float64)
    prediction = model.predict(FutureWarning.reshape(1,-1))

    if prediction[0] == 0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fradulant Transaction")    
