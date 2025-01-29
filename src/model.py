import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression

def load_data(train_path):
    train_df=pd.read_csv(train_path)
    return train_df

def x_y_value(train_df):
    x_train=train_df.iloc[:,:-1].values
    y_train=train_df.iloc[:,-1].values
    return x_train,y_train

def model(x_train,y_train):
    name=LogisticRegression()
    name.fit(x_train,y_train)
    joblib.dump(name,'model.pkl')


train_path='./data/pro/train_pre.csv'

def main():
    train_df=load_data(train_path)
    x_train,y_train=x_y_value(train_df)
    model(x_train,y_train)
  

if __name__=='__main__':
    main()
