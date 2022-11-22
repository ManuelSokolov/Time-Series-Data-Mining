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

if __name__ == '__main__':
    # read data set
    df = pd.read_csv("./Crime_Data/input/KCPD_Crime_Data_2020_2.csv", sep=',')

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

    #Change the date format
    df['Reported_Date'] = pd.to_datetime(df.Reported_Date)
    df['Reported_Date'] = df['Reported_Date'].dt.strftime('%d-%m-%Y')

    print(df.head(10))

    #Export to csv to better visualize everything
    #df.to_csv('./Crime_Data/dataset_clean.csv', encoding='utf-8')

    #print(df.head(10))
    
    # % of missing values per collumn
    #print (df.isnull().mean() * 100)

    #Export to csv to better visualize everything
    #df.to_csv('changes.csv', encoding='utf-8')

    #remove all collumns with more than 75% missing values
    # perc = 75.0
    # min_count =  int(((100-perc)/100)*df.shape[0] + 1)
    # mod_df = df.dropna( axis=1,thresh=min_count)
    
    #remote rows with more than 75% missing values
    # threshold = df.shape[1] * 0.75
    # df.dropna(thresh=threshold, inplace=True)
   
    #preprocess the column with content
    #pd = preprocess(dataset)
    #dataset_clean = pd.preprocess_col('content')
    #da.data_profiling()
