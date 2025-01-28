import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score,recall_score,precision_score


def load_data(test_path):
    test_df=pd.read_csv(test_path)
    return test_df

def x_y_value(test_df):
    x_test=test_df.iloc[:,:-1].values
    y_test=test_df.iloc[:,-1].values
    return x_test,y_test

def model(x_test,y_test):
    model=joblib.load('model.pkl')
    y_pred=model.predict(x_test)
    mit_dic={
        'accuracy':accuracy_score(y_pred,y_test),
        'recall':recall_score(y_pred,y_test,average='macro'),
        'precision':precision_score(y_pred,y_test,average='macro')
    }

    with open('metrics.json','w') as file:
        json.dump(mit_dic,file,indent=3)

test_path='./data/pro/test_pre.csv'

def main():
    test_df=load_data(test_path)
    x_test,y_test=x_y_value(test_df)
    model(x_test,y_test)

if __name__=='__main__':
    main()
