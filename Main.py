import pandas as pd
import Data_Analysis

if __name__ == '__main__':
    dataset = pd.read_csv('train.csv')
    da = Data_Analysis.Data_Analysis(dataset)
    da.data_profiling()
