# Stock Price Prediction

# Import libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Data Preparation
def prepare_data(df, forecast_col, forecast_out, test_size):
    label = df[forecast_col]  # creating new column called label with the last 5 rows are nan
    X = np.array(df[[forecast_col]])  # creating the feature array
    X = preprocessing.scale(X)  # processing the feature array
    X_lately = X[-forecast_out:]  # creating the column i want to use later in the predicting method
    # X = X[:-forecast_out]  # X that will contain the training and testing
    label.dropna(inplace=True)  # dropping na values
    y = np.array(df['Close'])  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0)  # cross validation

    response = [X_train, X_test, Y_train, Y_test, X_lately]
    return response


df = pd.read_csv('stock.csv')
# df = df['High']
forecast_col = 'High'
forecast_out = 5
test_size = 0.2

# Now I will split the data and fit into the linear regression model:

X_train, X_test, Y_train, Y_test, X_lately = prepare_data(df, forecast_col, forecast_out, test_size)

learner = LinearRegression()  # initializing linear regression model

learner.fit(X_train, Y_train)  # training the linear regression model

# Now let’s predict the output and have a look at the prices of the stock prices:

score = learner.score(X_test, Y_test)  # testing the linear regression model
forecast = learner.predict(X_lately)  # set that will contain the forecasted data
response = {}  # creating json object
response['test_score'] = score
response['forecast_set'] = forecast
print(response)
