import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder,StandardScaler

def load_data(train_path,test_path):
    train_df=pd.read_csv(train_path)
    test_df=pd.read_csv(test_path)
    return train_df,test_df

def x_y_value(train_df,test_df):
    x_train=train_df.iloc[:,:-1].values
    y_train=train_df.iloc[:,-1].values
    x_test=test_df.iloc[:,:-1].values
    y_test=test_df.iloc[:,-1].values
    return x_train,y_train,x_test,y_test

def processing_x(name,x_train,x_test):
    x_train_pre=name.fit_transform(x_train)
    x_test_pre=name.transform(x_test)
    return x_train_pre,x_test_pre

def processing_y(name1,y_train,y_test):
    y_train_pre=name1.fit_transform(y_train)
    y_test_pre=name1.transform(y_test)
    return y_train_pre,y_test_pre

def convert(x_train_pre,y_train_pre,x_test_pre,y_test_pre):
    x_train_pre=pd.DataFrame(x_train_pre)
    y_train_pre=pd.DataFrame(y_train_pre)
    x_test_pre=pd.DataFrame(x_test_pre)
    y_test_pre=pd.DataFrame(y_test_pre)
    return x_train_pre,y_train_pre,x_test_pre,y_test_pre

def combine(x_train_pre,y_train_pre,x_test_pre,y_test_pre):
    train_pre=pd.concat([x_train_pre,y_train_pre],axis=1,ignore_index=True)
    test_pre=pd.concat([x_test_pre,y_test_pre],axis=1,ignore_index=True)
    return train_pre,test_pre

def save_data(data_path,train_pre,test_pre):
    os.makedirs(data_path)
    train_pre.to_csv(os.path.join(data_path,'train_pre.csv'))
    test_pre.to_csv(os.path.join(data_path,'test_pre.csv'))

train_path='./data/raw/train.csv'
test_path='./data/raw/test.csv'

def main():
    train_df,test_df=load_data(train_path,test_path)
    x_train,y_train,x_test,y_test=x_y_value(train_df,test_df)
    name=StandardScaler()
    x_train_pre,x_test_pre=processing_x(name,x_train,x_test)
    name1=LabelEncoder()
    y_train_pre,y_test_pre=processing_y(name1,y_train,y_test)
    x_train_pre,y_train_pre,x_test_pre,y_test_pre=convert(x_train_pre,y_train_pre,x_test_pre,y_test_pre)
    train_pre,test_pre=combine(x_train_pre,y_train_pre,x_test_pre,y_test_pre)
    data_path='data/pro'
    save_data(data_path,train_pre,test_pre)

if __name__=='__main__':
    main()