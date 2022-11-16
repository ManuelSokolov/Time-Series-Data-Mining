import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from seaborn import heatmap






class Data_Analysis:

    def __init__(self, dataset):
        self.data = dataset.copy()
    '''
    Data Dimensionality gives number of records vs number of variables
    '''
    def data_dimensionality(self):
        data = self.data
        plt.figure(figsize=(4, 2))
        objects = ('nr records', 'nr variables')
        y_pos = np.arange(len(objects))
        values = [data.shape[0], data.shape[1]]
        plt.bar(y_pos, values, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.title('Number Records vs Number Variables')
        plt.savefig('Data_Profiling/NumberRecordsVsVariables.png')

    '''
       Number of variables per type 
    '''
    def number_of_variables_per_type(self):
        data = self.data
        cat_vars = data.select_dtypes(include='object')
        data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
        variable_types: dict = {
            'Numeric': [],
            'Binary': [],
            'Date': [],
            'Symbolic': []
        }
        for c in data.columns:
            uniques = data[c].dropna(inplace=False).unique()
            if len(uniques) == 2:
                variable_types['Binary'].append(c)
                data[c].astype('bool')
            elif data[c].dtype == 'datetime64':
                variable_types['Date'].append(c)
            elif c == "CRASH_DATE":
                variable_types['Date'].append(c)
            elif c == "CRASH_TIME":
                variable_types['Date'].append(c)
            elif data[c].dtype == 'int':
                variable_types['Numeric'].append(c)
            elif data[c].dtype == 'float':
                variable_types['Numeric'].append(c)
            else:
                data[c].astype('category')
                variable_types['Symbolic'].append(c)
            counts = {}
            for tp in variable_types.keys():
                counts[tp] = len(variable_types[tp])
            plt.figure(figsize=(6, 3))
            plt.bar(tuple(counts.keys()), list(counts.values()))
            plt.title('Nr of variables per type')
            plt.savefig('Data_Profiling/NumberVariablesPerType.png')
    '''
    Number of missing values per variable
    '''
    def missing_values(self):
        data = self.data
        mv = {}
        for var in data:
            nr = data[var].isna().sum()
            if nr > 0:
                mv[var] = nr

        plt.figure(figsize=(6, 3))
        plt.bar(tuple(mv.keys()),list(mv.values()))
        plt.title('Nr of missing values per variable')
        plt.xlabel('variables')
        plt.ylabel('nr missing values')
       # bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
        #          xlabel='variables', ylabel='nr missing values', rotation=True)
        plt.savefig('Data_Profiling/MissingValues.png', bbox_inches="tight")

    '''
    Data granularity for numerical variables with 10, 100, 1000 bins
    '''
    def data_granularity(self):
        data = self.data
        numeric_vars = [col for col in data.columns if data[col].dtype == 'int' or data[col].dtype =='float']
        if [] == numeric_vars:
            raise ValueError('There are no numeric variables.')
        HEIGHT = 10
        rows = len(numeric_vars)
        bins = (10, 100, 1000)
        cols = len(bins)
        fig, axs = plt.subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
        for i in range(rows):
            for j in range(cols):
                axs[i, j].set_title('Histogram for %s %d bins' % (numeric_vars[i], bins[j]))
                axs[i, j].set_xlabel(numeric_vars[i])
                axs[i, j].set_ylabel('Nr records')
                axs[i, j].hist(data[numeric_vars[i]].values, bins=bins[j], range=(0, 110))
        plt.savefig('Data_Profiling/granularity_study_numeric_variables.png')

    '''
    For the data distribution 
    -> NUMERICAL VARIABLES: we start by generating box plots for each  variable
    then we analise number of outliers per each numerical variable by IQR and STDEV; finnaly number of records 
    per value
    -> Symbolic variable: we analyse the number of records per each variable and we analyse the class distribution 
    '''
    def data_distribution(self):
        data = self.data
        # Separate variables into types (split later into method)
        cat_vars = data.select_dtypes(include='object')
        data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
        variable_types: dict = {
            'Numeric': [],
            'Binary': [],
            'Date': [],
            'Symbolic': []
        }
        for c in data.columns:
            uniques = data[c].dropna(inplace=False).unique()
            if len(uniques) == 2:
                variable_types['Binary'].append(c)
                data[c].astype('bool')
            elif data[c].dtype == 'datetime64':
                variable_types['Date'].append(c)
            elif c == "CRASH_DATE":
                variable_types['Date'].append(c)
            elif c == "CRASH_TIME":
                variable_types['Date'].append(c)
            elif data[c].dtype == 'int':
                variable_types['Numeric'].append(c)
            elif data[c].dtype == 'float':
                variable_types['Numeric'].append(c)
            else:
                data[c].astype('category')
                variable_types['Symbolic'].append(c)
        # First Part numerical variables
        # Boxplots for each variable and outliers by std (desvio padrao) e IQR
        outliers_iqr = []
        outliers_stdev = []
        summary5 = data.describe(include='number')
        NR_STDEV: int = 2
        for num_var in variable_types['Numeric']:
            fig = plt.figure()
            ax = data.boxplot(column=[num_var], return_type='axes')
            fig.savefig('Data_Profiling/Distribution/{}.png'.format(num_var+'box_plot'))
            iqr = 1.5 * (summary5[num_var]['75%'] - summary5[num_var]['25%'])
            outliers_iqr += [
                data[data[num_var] > summary5[num_var]['75%'] + iqr].count()[num_var] +
                data[data[num_var] < summary5[num_var]['25%'] - iqr].count()[num_var]]
            std = NR_STDEV * summary5[num_var]['std']
            outliers_stdev += [
                data[data[num_var] > summary5[num_var]['mean'] + std].count()[num_var] +
                data[data[num_var] < summary5[num_var]['mean'] - std].count()[num_var]]
        outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
        plt.figure(figsize=(12, utils.HEIGHT))
        utils.multiple_bar_chart(variable_types['Numeric'], outliers, title='Nr of outliers per variable', xlabel='variables',
                           ylabel='nr outliers', percentage=False)
        plt.savefig('Data_Profiling/Distribution/outliers.png')
        '''
        now number of records per value for both numerical and symbolic variables
        '''
    def correlations(self):
        data = self.data
        variable_types: dict = {
            'Numeric': [],
            'Binary': [],
            'Date': [],
            'Symbolic': []
        }
        for c in data.columns:
            uniques = data[c].dropna(inplace=False).unique()
            if len(uniques) == 2:
                variable_types['Binary'].append(c)
                data[c].astype('bool')
            elif data[c].dtype == 'datetime64':
                variable_types['Date'].append(c)
            elif c == "CRASH_DATE":
                variable_types['Date'].append(c)
            elif c == "CRASH_TIME":
                variable_types['Date'].append(c)
            elif data[c].dtype == 'int':
                variable_types['Numeric'].append(c)
            elif data[c].dtype == 'float':
                variable_types['Numeric'].append(c)
            else:
                data[c].astype('category')
                variable_types['Symbolic'].append(c)
        # delete later repeated code
        symbolic_vars = data[variable_types['Symbolic']].copy()
        corr_mtx = symbolic_vars.apply(lambda x: pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
        fig = plt.figure(figsize=[12, 12])
        heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Greens')
        plt.title('Correlation analysis')
        plt.savefig('Data_Profiling/Correlation/correlation_analysis_symbolic.png')
        plt.show()
        # now for the other types
        # question: can we do correlations between types of different kind?

    def data_profiling(self):
        self.data_dimensionality()
        self.number_of_variables_per_type()
        self.missing_values()
        self.data_granularity()
        self.data_distribution()
        self.correlations()





