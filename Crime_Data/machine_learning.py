import pandas as pd
import matplotlib.pyplot as mpl
from datetime import timedelta
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
# Mapping
import geopandas
import geopy
from geopy.geocoders import Nominatim
import folium
from geopy.extra.rate_limiter import RateLimiter
from folium import plugins
from folium.plugins import MarkerCluster


def apply_gradient_boosting_regressor(fraction_of_the_day, location, label, lim_lower, lim_high):
    #  First the noiseless case
    X = np.atleast_2d(fraction_of_the_day.values).T
    # Observations
    y = np.atleast_2d(location.values).T
    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    xx = np.atleast_2d(np.linspace(0, 1, 913)).T
    xx = xx.astype(np.float32)
    alpha = 0.95
    clf = GradientBoostingRegressor(loss='quantile', alpha=alpha,
                                    n_estimators=250, max_depth=3,
                                    learning_rate=.1, min_samples_leaf=9,
                                    min_samples_split=9)
    clf.fit(X, y)
    # Make the prediction on the meshed x-axis
    y_upper = clf.predict(xx)
    clf.set_params(alpha=1.0 - alpha)
    clf.fit(X, y)
    # Make the prediction on the meshed x-axis
    y_lower = clf.predict(xx)
    clf.set_params(loss='ls')
    clf.fit(X, y)
    # Make the prediction on the meshed x-axis
    y_pred = clf.predict(xx)
    # Plot the function, the prediction and the 90% confidence interval based on
    # the MSE
    fig = mpl.figure()
    mpl.figure(figsize=(20, 10))
    mpl.plot(X, y, 'b.', markersize=10, label=u'Observations')
    mpl.plot(xx, y_pred, 'r-', label=u'Prediction')
    mpl.plot(xx, y_upper, 'k-')
    mpl.plot(xx, y_lower, 'k-')
    mpl.fill(np.concatenate([xx, xx[::-1]]),
             np.concatenate([y_upper, y_lower[::-1]]),
             alpha=.5, fc='b', ec='None', label='90% prediction interval')
    mpl.xlabel('$Time of Day by Fraction$')
    mpl.ylabel(label)
    mpl.ylim(lim_lower, lim_high)
    mpl.legend(loc='upper right')
    mpl.show()
    return y_pred

def apply_logistic_regression(coord):
    # Create the model
    model = LinearRegression()

    # Reshape the data to be in the form [x(t), x(t+1)]
    X = np.column_stack((coord[:-1], coord[1:]))

    # Fit the model to the data
    model.fit(X[:, 0].reshape(-1, 1), X[:, 1])

    # Predict the next x coordinate
    return model.predict(coord.iloc[-1].reshape(-1, 1))[0]

data = pd.read_csv("data_crime_final.csv")

timestamps = pd.to_datetime(data['Timestamps'])
fraction_of_the_day = (timestamps-timestamps.dt.normalize()).dt.total_seconds()/timedelta(days=1).total_seconds()

location_x = data['Location_X']

location_y = data['Location_Y']

plot = mpl.scatter(timestamps, location_x)
mpl.title("Location X vs Time")
mpl.show()

plot = mpl.scatter(timestamps, location_y)
mpl.title("Location Y vs Time")
mpl.show()

plot = mpl.scatter(fraction_of_the_day, location_x)
mpl.title("Location X vs Fraction of the day")
mpl.show()

plot = mpl.scatter(fraction_of_the_day, location_y)
mpl.title("Location Y vs Fraction of the day")
mpl.show()

# filter more outliers found
# Filter the data to remove values outside the specified range
mask_x = (location_x >= 36.0) & (location_x <= 42.0)
mask_y = (location_y >= -105.0) & (location_y <= -85.0)
mask = mask_x & mask_y
filtered_timestamps = timestamps[mask]
filtered_fraction_of_the_day = fraction_of_the_day[mask]
filtered_location_x = location_x[mask]
filtered_location_y = location_y[mask]
prediction_x = apply_gradient_boosting_regressor(filtered_fraction_of_the_day, filtered_location_x,"Location x", 38.0,40.0)
prediction_y = apply_gradient_boosting_regressor(filtered_fraction_of_the_day, filtered_location_y,"Location y", -96.0,-94.0)
print("Coordinates of the next crime might be (" + str(np.mean(prediction_x)) + ", " + str(np.mean(prediction_y)) + ")")
# Map points of events
m7 = folium.Map([39.25,-94.50], zoom_start=14)
for i in range(len(prediction_y)):
    folium.CircleMarker([prediction_y[i], prediction_x[i]],
                        radius=4,
                        popup=str(i),
                        fill_color="#3db7e4", # divvy color
                       ).add_to(m7)
# convert to (n, 2) nd-array format for heatmap
matrix = np.column_stack((prediction_y, prediction_x))
# plot heatmap
m7.add_child(plugins.HeatMap(matrix, radius=15))

m7.save("m7.html")

# approach 2 - Linear Regression - Predict the next x given TimeStamps

from sklearn.linear_model import LinearRegression

x_pred = apply_logistic_regression(filtered_location_x)
y_pred = apply_logistic_regression(filtered_location_y)
print("The coordinates of the next crime according to linear regressio are: (" + str(x_pred) +", "+ str(y_pred) + ")")