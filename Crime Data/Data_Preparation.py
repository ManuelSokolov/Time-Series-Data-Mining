import statistics as st

class Data_Preparation():
    def __init__(self,data):
        self.data = data.copy()


    # {'From_Date': 214, 'From_Time': 317, 'To_Date': 70438, 'To_Time': 70895, 'IBRS': 639, 'Beat': 178, 'Address': 31,
    # 'City': 31, 'Zip Code': 4246, 'Rep_Dist': 265, 'Area': 265, 'Race': 15641, 'Sex': 15641, 'Age': 44755, 'Location 1': 31}
    def missing_values(self):
        # FIND VARIABLES WITH MISSING VALUES
        mv = {}
        for var in self.data:
            nr = self.data[var].isna().sum()
            if nr > 0:
                mv[var] = nr

        # DISCARD COLUMNS WITH MORE THEN 85% MISSING VALUES
        threshold = self.data.shape[0] * 0.85

        missings = [c for c in mv.keys() if mv[c] > threshold]
        self.data.drop(columns=missings, inplace=True)
        print('Dropped variables', missings)

        # DISCARD RECORDS WITH MAJORITY OF MISSING VALUES
        threshold = self.data.shape[1] * 0.50

        self.data.dropna(thresh=threshold, inplace=True)
        print(self.data.shape)

        # PERSON_AGE - all the NaN are minor age <21
        #mean_ages = int(person_age.mean())
        self.data['Age'].fillna(21, inplace=True)

        # RACE ? preencher com unknown ?
        self.data['Race'].fillna('Unknown', inplace=True)

        # Sex ?
        self.data['Sex'].fillna('U', inplace=True)

        # Zip Code mais comum ou ir ver qual e o codigo da zona? -> ma ideia
        #self.data['Zip Code'].fillna(most_common, inplace=True)

    def drop_unnecessary_vars(self):
        #drop row with not a number values
        self.data.dropna(subset=[1])
        self.data = self.data[self.data.notnull(self.data['Zip'])]


    def Data_Prepare(self):
        pass