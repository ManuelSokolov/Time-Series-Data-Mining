# Basic Analysis and Visualization
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from datetime import timedelta
# Mapping
import geopy
from geopy.geocoders import Nominatim
import folium
from geopy.extra.rate_limiter import RateLimiter
from folium import plugins
from folium.plugins import MarkerCluster
# Statistical OLS Regression Analysis
import statsmodels.api as sm
from statsmodels.compat import lzip
from statsmodels.formula.api import ols
#Scipy sklearn Predictions
from sklearn.ensemble import GradientBoostingRegressor


# Pull in JSON and set index as case number (unique key)
df = pd.read_csv("data_crime_final.csv", sep=',')
df = df.set_index("Report_No")
print(df.head())


# Convert time objects
df['Timestamps'] = pd.to_datetime(df['Timestamps'])
df['date'] = [d.date() for d in df['Timestamps']]
df['time'] = [d.time() for d in df['Timestamps']]
df['day'] = df['Timestamps'].dt.day_name()
# Find Fractions of Day
df['timeint'] = (df['Timestamps']-df['Timestamps'].dt.normalize()).dt.total_seconds()/timedelta(days=1).total_seconds()

df.groupby("Offense")["Offense"].count().sort_values()

df.to_csv('temp.csv',index=False)
