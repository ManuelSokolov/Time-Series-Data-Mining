import pandas as pd
import Data_Analysis
import os

if __name__ == '__main__':
	newpath = r'Data_Profiling' 
	newpath2 = r'Data_Profiling/Distribution' 
if not os.path.exists(newpath):
	os.makedirs(newpath)
if not os.path.exists(newpath2):
	os.makedirs(newpath2)

dataset = pd.read_csv('train.csv')
da = Data_Analysis.Data_Analysis(dataset)
da.data_profiling()
