import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split

def load_data(url):
    df=pd.read_csv(url)
    return df

def drop(df):
    df=df.dropna()
    return df
    

def train_test(df,test_size):
    train_df,test_df=train_test_split(df,test_size=test_size,random_state=5)
    return train_df,test_df

def save_data(data_path,train_df,test_df):
    os.makedirs(data_path)
    train_df.to_csv(os.path.join(data_path,'train.csv'))
    test_df.to_csv(os.path.join(data_path,'test.csv'))

def main():
    df=load_data('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
    df=drop(df)
    train_df,test_df=train_test(df,0.2)
    data_path='data/raw'
    save_data(data_path,train_df,test_df)

if __name__=='__main__':
    main()