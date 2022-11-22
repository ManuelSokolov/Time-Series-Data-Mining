Reported_Date         0.000000
Reported Time         0.000000
From_Date             0.001039  V Its not worth to keep if we dont have To_date and Time
From Time             0.001039  V Its not worth to keep if we dont have To_date and Time
To_Date              66.637913  V Too many missing values
To Time              66.637913  V Too many missing values
Offense               0.000000
IBRS                 10.012471  V Too many missing values
Description          10.012471  V Too many missing values
Beat                  0.002079  V Irrelevant
Address               0.000000  
City                  0.001039  V Irrelevant
Zip Code             12.077531  V Too many missing values
Rep_Dist             30.434421  V Too many missing values
Area                  0.008314  V Maybe worth to use to PREDICT AREA INSTEAD OF LOCATION
DVFlag                0.000000  
Involvement           0.000000  V After thinking about it, it doesn't matter too much.
Race                  0.000000
Sex                   0.000000
Age                   0.000000
Firearm Used Flag     0.000000
Location              0.000000
dtype: float64

    #df.drop(['From_Date', 'From Time', 'To_Date', 'To Time', 'IBRS', 'Beat', 'Rep_Dist','DVFlag','Description', 'Address', 'City', 'Area', 'Zip Code'], axis=1, inplace=True)
