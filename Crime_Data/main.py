import math
import collections
from itertools import islice
import pandas as pd
import datetime
import Data_Analysis
import Data_Preparation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy
from geopy.geocoders import Nominatim

if __name__ == '__main__':
    # read data set
    df = pd.read_csv("input/KCPD_Crime_Data_2020_2.csv", sep=',')

    #Drop deemed unnecessary colls
    df.drop(['From_Date', 'From Time', 'To_Date', 'To Time', 'IBRS', 'Beat', 'Rep_Dist','Description', 'City', 'Area', 'Zip Code', 'Involvement'], axis=1, inplace=True)
    #Sort by Report Number, Date and Time to better visualize each instance of each report at a time
    df = df.sort_values(["Report_No","Reported_Date","Reported Time"])
    
    #Remove the text from the Location and just keep the GPS Coordinates
    df['Location'] = '(' + df['Location'].str.split('(').str[1]

    #Remove multiple instances of the same crime report
    df = df.drop_duplicates(subset=['Report_No'])
    
    # Null ages become 0 -> Meaning its a crime that involves a Minor
    df['Age'].fillna(0, inplace=True)

    # Null values of Sex become Undefined 
    df['Sex'].fillna('U', inplace=True)

    # Null values of Race become Undefined 
    df['Race'].fillna('U', inplace=True)
    
    # Null coordinates of a Crime Location become 0 
    df['Location'].fillna(0, inplace=True)

    #change index to the Report Number
    df.set_index('Report_No', inplace=True, drop=True)

    #df = Data_Preparation.Data_Preparation.fill_know_missing_values(df)
    
    #Data_Preparation.Data_Preparation.count_uniques_without_location(df)

    #Has 40803 missing location values
    dt = pd.read_csv("teste_andre.csv", sep=',')

    #Remove rows with location '0'
    dt = dt[dt.Location != '0']

    dt = dt.iloc[: , 4:]
    dt = dt.iloc[:, [1,2,0,3,4,5,6,7,8,9,10]]


    #Format Date and Time in the same column
    dt['Reported_Date'] = pd.to_datetime(dt.Reported_Date, format='%d-%m-%Y')
    dt['Reported_Date'] = dt['Reported_Date'].dt.strftime('%Y-%m-%d')
    dt['Reported Time'] = dt['Reported Time'].astype(str) + ':00'
    dt['Timestamps'] = pd.to_datetime(dt['Reported_Date'] + dt['Reported Time'], format='%Y-%m-%d%H:%M:%S')
    dt = dt.drop(['Reported_Date'], axis=1)
    dt = dt.drop(['Reported Time'], axis=1)

    #Format location into 2 columns
    column_x, column_y = Data_Preparation.Data_Preparation.parse_location(dt)
    dt['Location_X'] = column_x
    dt['Location_Y'] = column_y
    dt = dt.drop(['Location'], axis=1)

    #ID to numeric values
    dt = Data_Preparation.Data_Preparation.parse_ID(dt)
    dt["Report_No"] = pd.to_numeric(dt["Report_No"])

    #Timestamp in the first row
    dt = dt.iloc[:, [8,1,2,0,3,4,5,6,7,9,10]]
    print(dt.head())  
    
    #convert Firearm to numeric
    dt["Firearm Used Flag"] = dt["Firearm Used Flag"].astype(int)

    #Convert DVFlag to numeric
    d = {'Y': True, 'N': False}
    dt['DVFlag'] = dt['DVFlag'].map(d)
    dt["DVFlag"] = dt["DVFlag"].astype(int)

    #Convert sex to numeric
    d = {'F': 1, 'M': 0, 'U':2}
    dt['Sex'] = dt['Sex'].map(d)
    dt["Sex"] = dt["Sex"].astype(int)

    #Convert  race to numeric
    d = {'B': 1, 'W': 0, 'U':2, 'I':3}
    dt['Race'] = dt['Race'].map(d)
    dt["Race"] = dt["Race"].astype(int)

    #age to age groups 
    bins= [0,18,30,45,60,75,110]
    labels = [0,1,2,3,4,5]
    dt['AgeGroup'] = pd.cut(dt['Age'], bins=bins, labels=labels, right=False)
    dt["AgeGroup"] = pd.to_numeric(dt["AgeGroup"])
    dt = dt.drop(['Age'], axis=1)

    #Location to numeric
    dt["Location_X"] = pd.to_numeric(dt["Location_X"])
    dt["Location_Y"] = pd.to_numeric(dt["Location_Y"])

    print(dt.dtypes)
    
    #dataframe to csv
    dt.to_csv('data_crime.csv',index=False)
    