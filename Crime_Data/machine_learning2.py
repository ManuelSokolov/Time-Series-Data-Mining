import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import folium
from folium import plugins
from folium.plugins import MarkerCluster
import webbrowser
from IPython.display import display

color_pal = sns.color_palette()

def outlier_removal_and_visualization(df):
    # Read data
    df = df.copy()

    # Create Time series for location x
    df_time_loc_x = df[['Timestamps', 'Location_X']]
    df_time_loc_x = df_time_loc_x.set_index('Timestamps')
    df_time_loc_x.index = pd.to_datetime(df_time_loc_x.index)

    # Create Time series for location y
    df_time_loc_y = df[['Timestamps', 'Location_Y']]
    df_time_loc_y = df_time_loc_y.set_index('Timestamps')
    df_time_loc_y.index = pd.to_datetime(df_time_loc_y.index)

    # Plot the location x and location y
    df_time_loc_x.plot(style=".", color=color_pal[0], figsize=(15, 5), title="Location x vs time")
    plt.show()
    df_time_loc_y.plot(style=".", color=color_pal[0], figsize=(15, 5), title="Location y vs time")
    plt.show()

    # Filter the data to remove values outside the specified range
    df = df[(df['Location_X'] >= 38.8) & (df['Location_X'] <= 39.4)]
    df = df[(df['Location_Y'] >= -94.75) & (df['Location_Y'] <= -94.35)]
    # Create Time series for location x
    df_time_loc_x = df[['Timestamps', 'Location_X']]
    df_time_loc_x = df_time_loc_x.set_index('Timestamps')
    df_time_loc_x.index = pd.to_datetime(df_time_loc_x.index)

    # Create Time series for location y
    df_time_loc_y = df[['Timestamps', 'Location_Y']]
    df_time_loc_y = df_time_loc_y.set_index('Timestamps')
    df_time_loc_y.index = pd.to_datetime(df_time_loc_y.index)

    # Plot the location x and location y
    df_time_loc_x.plot(style=".", color=color_pal[0], figsize=(15, 5), title="Location x vs time")
    plt.show()
    df_time_loc_y.plot(style=".", color=color_pal[0], figsize=(15, 5), title="Location y vs time")
    plt.show()
    return df_time_loc_x, df_time_loc_y

# Feature Creation
def _create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

def train_lin_regression(df_time_loc_x, df_time_loc_y):

    mdl1 = LinearRegression()
    mdl2 = LinearRegression()
    train_x = df_time_loc_x[df_time_loc_x.index < '2020-10-01']
    test_x = df_time_loc_x[df_time_loc_x.index >= '2020-10-01']

    train_y = df_time_loc_y[df_time_loc_y.index < '2020-10-01']
    test_y = df_time_loc_y[df_time_loc_y.index >= '2020-10-01']

    train_x = _create_features(train_x)
    test_x = _create_features(test_x)
    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month']
    TARGET = 'Location_X'

    X_train_x = train_x[FEATURES]
    y_train_x = train_x[TARGET]

    mdl1.fit(X_train_x, y_train_x)

    X_test_x = test_x[FEATURES]

    test_x['prediction'] = mdl1.predict(X_test_x)

    score = np.sqrt(mean_squared_error(test_x['Location_X'], test_x['prediction']))
    print(f'RMSE Score on Test set: {score:0.2f}')

    train_y = _create_features(train_y)
    test_y = _create_features(test_y)
    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month']
    TARGET = 'Location_Y'

    X_train_y = train_y[FEATURES]
    y_train_y = train_y[TARGET]

    mdl1.fit(X_train_y, y_train_y)

    X_test_y = test_y[FEATURES]

    test_y['prediction'] = mdl1.predict(X_test_y)

    score = np.sqrt(mean_squared_error(test_y['Location_Y'], test_y['prediction']))
    print(f'RMSE Score on Test set: {score:0.2f}')

    return mdl1,mdl2






def train_model(df_time_loc_x, df_time_loc_y):

    # Separate data into train and test for location x and location y
    train_x = df_time_loc_x[df_time_loc_x.index < '2020-10-01']
    test_x = df_time_loc_x[df_time_loc_x.index >= '2020-10-01']
    fig, ax = plt.subplots(figsize=(15, 5))
    train_x.plot(ax=ax, label='Training Set', title="Location X split")
    test_x.plot(ax=ax, label='Test set')
    ax.legend(['Training set', 'Test set'])
    plt.show()
    train_y = df_time_loc_y[df_time_loc_y.index < '2020-10-01']
    test_y = df_time_loc_y[df_time_loc_y.index >= '2020-10-01']
    fig, ax = plt.subplots(figsize=(15, 5))
    train_y.plot(ax=ax, label='Training Set', title="Location Y split")
    test_y.plot(ax=ax, label='Test set')
    ax.legend(['Training set', 'Test set'])
    plt.show()

    df_time_loc_x = _create_features(df_time_loc_x)
    df_time_loc_y = _create_features(df_time_loc_y)

    # Visualize Hourly Location X location
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=df_time_loc_x, x='hour', y='Location_X')
    ax.set_title('Location X by Hour')
    plt.show()

    # Visualize Hourly Location X location
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=df_time_loc_y, x='hour', y='Location_Y')
    ax.set_title('Location Y by Hour')
    plt.show()

    # Visualize location X by month
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=df_time_loc_x, x='month', y='Location_X', palette='Blues')
    ax.set_title('Location X by Month')
    plt.show()

    # Visualize location y by month
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=df_time_loc_y, x='month', y='Location_Y', palette='Blues')
    ax.set_title('Location Y by Month')
    plt.show()

    train_x = _create_features(train_x)
    test_x= _create_features(test_x)
    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month']
    TARGET = 'Location_X'

    X_train_x = train_x[FEATURES]
    y_train_x = train_x[TARGET]

    X_test_x = test_x[FEATURES]
    y_test_x = test_x[TARGET]

    reg = xgb.XGBRegressor(base_score=0.05, booster='gbtree',
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:squarederror',
                           max_depth=6,
                           learning_rate=0.1)
    reg.fit(X_train_x, y_train_x,
            eval_set=[(X_train_x, y_train_x), (X_test_x, y_test_x)],
            verbose=100)

    fi = pd.DataFrame(data=reg.feature_importances_,
                 index=reg.feature_names_in_,
                 columns=['importance'])
    fi.sort_values('importance').plot(kind='barh', title='Feature Importance for Location X')
    plt.show()

    test_x['prediction'] = reg.predict(X_test_x)
    df_time_loc_x = df_time_loc_x.merge(test_x[['prediction']], how='left', left_index=True, right_index=True)
    ax = df_time_loc_x[['Location_X']].plot(figsize=(15, 5))
    df_time_loc_x['prediction'].plot(ax=ax, style='.')
    plt.legend(['Truth Data', 'Predictions'])
    ax.set_title('Raw Dat and Prediction')
    plt.show()

    score = np.sqrt(mean_squared_error(test_x['Location_X'], test_x['prediction']))
    print(f'RMSE Score on Test set: {score:0.2f}')

    # Now for location Y

    train_y = _create_features(train_y)
    test_y = _create_features(test_y)

    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month']
    TARGET = 'Location_Y'

    X_train_y = train_y[FEATURES]
    y_train_y = train_y[TARGET]

    X_test_y = test_y[FEATURES]
    y_test_y = test_y[TARGET]

    reg2 = xgb.XGBRegressor(base_score=0.05, booster='gbtree',
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:squarederror',
                           max_depth=6,
                           learning_rate=0.1)
    reg2.fit(X_train_y, y_train_y,
            eval_set=[(X_train_y, y_train_y), (X_test_y, y_test_y)],
            verbose=100)

    fi = pd.DataFrame(data=reg2.feature_importances_,
                 index=reg2.feature_names_in_,
                 columns=['importance'])
    fi.sort_values('importance').plot(kind='barh', title='Feature Importance for Location y')
    plt.show()

    test_y['prediction'] = reg2.predict(X_test_y)

    #aNDRE -----------------------------

    # alpha = 0.95 

    # # over predict
    # reg2_95 = xgb.XGBRegressor(loss="quantile", alpha=alpha,
    #                     n_estimators=1000,
    #                     early_stopping_rounds=50,
    #                     objective='reg:squarederror',
    #                     max_depth=6,
    #                     learning_rate=0.1)
    
    # reg2_95.fit(X_train_y, y_train_y,
    #         eval_set=[(X_train_y, y_train_y), (X_test_y, y_test_y)],
    #         verbose=100)

    # # over predict
    # alpha = 0.05
    # reg2_05 = xgb.XGBRegressor(loss="quantile", alpha=alpha,
    #                     n_estimators=1000,
    #                     early_stopping_rounds=50,
    #                     objective='reg:squarederror',
    #                     max_depth=6,
    #                     learning_rate=0.1)
    
    # reg2_05.fit(X_train_y, y_train_y,
    #         eval_set=[(X_train_y, y_train_y), (X_test_y, y_test_y)],
    #         verbose=100)
    
    # teste_95 = reg2_95.predict(X_test_y)   
    # teste_05 = reg2_05.predict(X_test_y)  

    # # Plot the predictions and confidence intervals
    # plt.plot(teste_95, 'b-', label='Prediction_upper')
    # plt.plot(teste_95, 'b-', label='Prediction_lower')

    # df_time_loc_y = df_time_loc_y.merge(test_y[['prediction']], how='left', left_index=True, right_index=True)
    # ax = df_time_loc_y[['Location_Y']].plot(figsize=(15, 5))
    # df_time_loc_y['prediction'].plot(ax=ax, style='.')
    # df_real_confianca = pd.DataFrame(
    # {'95': teste_95,
    #  '05': teste_05
    # })
    # df_real_confianca['95'].plot(ax=ax, style='.')
    # df_real_confianca['05'].plot(ax=ax, style='.')
    # plt.legend(['Truth Data', 'Predictions'])
    # ax.set_title('Raw Dat and Prediction')
    # plt.show()
    #plt.legend("Andre")

    # fig = plt.figure(figsize=(10, 10))
    # ax = df_time_loc_y[['Location_Y']].plot(figsize=(15, 5))
    # plt.plot(ax, test_y[['prediction']], "r-", label="Predicted mean")
    # plt.plot(ax, teste_95, "k-")
    # plt.plot(ax, teste_05, "k-")
    # plt.fill_between(
    #     ax.ravel(), teste_05, teste_95, alpha=0.4, label="Predicted 90% interval"
    # )
    # plt.xlabel("$x$")
    # plt.ylabel("$f(x)$")
    # plt.ylim(-10, 25)
    # plt.legend(loc="upper left")
    #plt.show()

    # Show the plot
    
    #aNDRE -----------------------------


    df_time_loc_y = df_time_loc_y.merge(test_y[['prediction']], how='left', left_index=True, right_index=True)
    ax = df_time_loc_y[['Location_Y']].plot(figsize=(15, 5))
    df_time_loc_y['prediction'].plot(ax=ax, style='.')
    plt.legend(['Truth Data', 'Predictions'])
    ax.set_title('Raw Dat and Prediction')
    plt.show()

    score = np.sqrt(mean_squared_error(test_y['Location_Y'], test_y['prediction']))

    map = folium.Map([38.984518,-94.511084], zoom_start=14)
    df_prediction = pd.DataFrame(
    {'Location_X': test_x['prediction'],
     'Location_Y': test_y['prediction']
    })
    #df3=pd.merge(test_x['prediction'],test_y['prediction'])
    print(df_prediction.head())
    plugins.HeatMap(df_prediction[['Location_X', 'Location_Y']].values, radius=15).add_to(map)
    #Display the map
    map.save('result_andre_prediction.html')
    df_real = pd.DataFrame(
    {'Location_X': test_x['Location_X'],
     'Location_Y': test_y['Location_Y']
    })
    #df3=pd.merge(test_x['prediction'],test_y['prediction'])
    print(df_real.head())
    plugins.HeatMap(df_real[['Location_X', 'Location_Y']].values, radius=15).add_to(map)
    #Display the map
    map.save('result_andre_real.html')
    print(f'RMSE Score on Test set: {score:0.2f}')
    return reg, reg2

def apply_model(model_for_x, model_for_y, list_of_timestamps):
    df = pd.DataFrame()
    df['Timestamps'] = list_of_timestamps
    df = df.set_index('Timestamps')
    df.index = pd.to_datetime(df.index)
    df = _create_features(df)
    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month']
    df_final = df[FEATURES]
    x_coords = model_for_x.predict(df_final)
    y_coords = model_for_y.predict(df_final)
    return [(x,y) for x,y in zip(x_coords, y_coords)]




if __name__ == '__main__':
    df = pd.read_csv('data_crime_final.csv')
    df_time_loc_x, df_time_loc_y = outlier_removal_and_visualization(df)
    model_for_x, model_for_y = train_model(df_time_loc_x, df_time_loc_y)
    # para treinar com linear regression passar train_lin_regression(df_time_loc_x, df_time_loc_y)
    list_of_timestamps = ['2021-01-01 02:00:00', '2021-06-03 14:00:00', '2022-05-01 12:00:00', '2022-01-05 13:00:00']
    print("Target Locations :", apply_model(model_for_x, model_for_y, list_of_timestamps))
