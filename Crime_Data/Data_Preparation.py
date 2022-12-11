import statistics as st
from geopy.geocoders import Nominatim
import time
import datetime

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
        self.head(10)
        # transformar location em coordenada so
        # remover duplicados!! alguns teem victima e suspeito, pertencem ao mesmo report mas teem roles diferentes
        # Zip Code mais comum ou ir ver qual e o codigo da zona? -> ma ideia
        #self.data['Zip Code'].fillna(most_common, inplace=True)

    def drop_unnecessary_vars(self):
        #drop row with not a number values
        #self.data.dropna(subset=[1])
        #self.data = self.data[self.data.notnull(self.data['Zip Code'])]
        self.data.drop(['Report_No', 'From_Time', 'To_Date', 'To_Time', 'IBRS', 'Beat', 'City','Location 1'],
                axis=1, inplace=True)

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


    def Data_Prepare(self):
        self.missing_values()
        self.drop_unnecessary_vars()
        return self.data
    
    def fill_know_missing_values(df):
        list_adress = []
        list_location = []

        for i, j in df.iterrows():
        
            location = j['Location']
            address = j['Address']

            if address not in list_adress and location != 0:
                list_adress.append(address)
                list_location.append(location)

        k = " Kansas City"
        #lista_addK = [x + k for x in list_adress]
        geolocator = Nominatim(user_agent="BiomedicalProj", timeout=100)
        
        for i, j in df.iterrows():
            if(j['Location'] == 0):
                if(j['Address'] in list_adress):
                    df.at[i,'Location'] = list_location[list_adress.index(j['Address'])]
                else:
                    try:
                        n = geolocator.geocode(str(j['Address']) + k)
                        if n != None:
                            a = (n.latitude, n.longitude)
                            print(j['Location'])
                            df.at[i,'Location'] = a
                            #print(df[i]['Location'])
                            print()
                    
                    except:
                        print("Este não deu!!!")
    
        return df

    def count_uniques_without_location(df):
        unique_adress=df['Address'].unique()
        count_unique_zero = 0
        count = 0
        for i, j in df.iterrows():
            if (j['Address'] in unique_adress):
                if(j['Location'] == 0):
                    count_unique_zero += 1
                    unique_adress = unique_adress[unique_adress != j['Address']]

            if(j['Location'] == 0):
                count += 1

        print("number of uniques with no location coordinates:")
        print(count_unique_zero)
        print("number without location coordinates:")
        print(count)

    def parse_location(df):
        column_x = []
        column_y = []
        for i, j in df.iterrows():
            location = j['Location']
            location = location.replace('(', '')
            location = location.replace(')', '')
            location = location.replace(' ', '')
            location_arr = location.split(",")
            column_x.append(location_arr[0])
            column_y.append(location_arr[1])

        return column_x, column_y

    def parse_ID(df):
        for i, j in df.iterrows():
            ID = j['Report_No']
            ID = ID.replace('K', '')
            ID = ID.replace('C', '')
            df.at[i,'Report_No'] = ID
    
        return df



    