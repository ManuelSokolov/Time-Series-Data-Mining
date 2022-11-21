import math

import pandas as pd
import Data_Analysis
import Data_Preparation
from preprocess_nlp import preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
count = 0
def location(string):
    print(string)
    str = string.split("\n")
    if len(str) <= 2:
        count += 1
        return (float(0.0),float(0.0))
    str = str[2].split(',')
    str[0] = str[0][1:]
    str[1] = str[1][1:]
    str[1] = str[1][:-1]
    print((str[0],str[1]))
    return (float(str[0]),float(str[1]))

def other(string):
    if isinstance(string,float):
        if math.isnan(string):
            return string
    str = string.split("\n")
    s =  ' '.join(str)
    #print(s)
    return s

'''
Index(['Report_No', 'Reported_Date', 'Reported Time', 'From_Date', 'From Time',
       'To_Date', 'To Time', 'Offense', 'IBRS', 'Description', 'Beat',
       'Address', 'City', 'Zip Code', 'Rep_Dist', 'Area', 'DVFlag',
       'Involvement', 'Race', 'Sex', 'Age', 'Firearm Used Flag', 'Location'],
      dtype='object')
'''
#- date of crime , time of crime , tipe of ofense, description, address, area, sex, age,
#, coordinates
def fix(s):
    print(s)
    if isinstance(s, float):
        if math.isnan(s):
            return s
    s = s.replace("\'"," ")
    return s


if __name__ == '__main__':
    # read data set
    df = pd.read_csv("./input/KCPD_Crime_Data_2020.csv", sep=',')
    #df.drop(['From_Date', 'From_Time', 'To_Date', 'To_Time', 'IBRS', 'Beat', 'Rep_Dist','DVFlag', 'Invl_No'], axis=1, inplace=True)
    df = df.sort_values(["Reported_Date","Reported Time"])
    print(df.columns)
    print(df['Location'].head(5))
    df['Location'] = df['Location'].apply(lambda x: other(x))
    df['Address'] = df['Address'].apply(lambda x : fix(x))
    df['Location'] = df['Location'].apply(lambda x : fix(x))
    df['City'] = df['City'].apply(lambda x : fix(x))
    print(count)
    print(df['Location'].head(5))
    df.to_csv('KCPD_Crime_Data_2020_2.csv', index=False)
    '''
    df["Reported_Date"] = pd.to_datetime(df["Reported_Date"])
    print(df.head(10)[['Reported_Date','Reported_Time']])
    #train, test = train_test_split(df, test_size=0.85, shuffle=False)
    df["Reported_Date"] = pd.to_datetime(df["Reported_Date"])
    da = Data_Analysis.Data_Analysis(df)
    da.data_profiling()
    dp = Data_Preparation.Data_Preparation(df)
    df = dp.Data_Prepare()
    df.to_csv('data_preparation.csv', index=False)
    '''


    # preprocess the column with content
   # pd = preprocess(dataset)
   # dataset_clean = pd.preprocess_col('content')
    # da.data_profiling()












