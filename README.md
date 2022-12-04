# ECS171FinalProject

## Table of Contents:
  - Introduction
  - Figures
  - Methods
  - Results
  - Discussion
  - Conclusion
  - Collaboration



## Introduction <br/>
&emsp;The purpose of this project is to build a regression model for predicting flights delay in the U.S.  For a better prediction model, passengers can have better schedule planning instead of waiting in the airport. Related to daily experience, people tend to believe that airport congestion or specific low-cost flight is the main cause of flight delays. The model should be able to reveal how each factor impacts the flight's delay and how long the delay might be. <br/>
&emsp;The dataset is found from Kaggle's “2015 Flight Delays and Cancellations”, originally abstracted from the U.S. Department of Transportation (DOT). For this project, based on different goals compared to the Kaggle Event, not full of the data was used to build the prediction model for delays. The model mainly relies on the  “flights.csv”, which includes detail of each flight (5819079 samples) in the USA from 2015. <br/>
 
 ## Methods <br/>
 #### Data Downloads
```python
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import warnings
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
warnings.filterwarnings('ignore')
```

```python
link = 'https://drive.google.com/file/d/1_H1j57EahEEXpZtQ413BpF2JiAAr1Anv/view?usp=share_link'
id = "1_H1j57EahEEXpZtQ413BpF2JiAAr1Anv"
print (id) # Verify that you have everything after '='
downloaded = drive.CreateFile({'id':id}) 
downloaded.GetContentFile('flights.csv')  
```
Pandas and Numpy is required for this project for basic data processing.
```python
import  pandas as pd
import  numpy as np

flights = pd.read_csv("flights.csv")
```
 #### Data Preprocessing
  Note: Full code with comments for Data Preprocessing can be found in "Data_Preprocessing.ipynb"
 
 1. Drop features unrelated features.
 ```python
columns = flights.columns
print("Features of current dataset",columns)
flights=flights.drop(['WEATHER_DELAY','LATE_AIRCRAFT_DELAY','AIRLINE_DELAY','SECURITY_DELAY','AIR_SYSTEM_DELAY','CANCELLATION_REASON',
 "CANCELLED", "DIVERTED","TAXI_IN", "TAXI_OUT"],axis=1)
 ```
 2. Drop overlapped features. 
```python
flights=flights.drop(['SCHEDULED_ARRIVAL','ARRIVAL_TIME','DEPARTURE_TIME','SCHEDULED_DEPARTURE','TAIL_NUMBER','FLIGHT_NUMBER', 'WHEELS_OFF', 'WHEELS_ON', "SCHEDULED_TIME", "ELAPSED_TIME"],axis=1)
 ```
 4. Drop feature of the year.
```python
print(flights["YEAR"].unique())
flights=flights.drop(['YEAR'],axis=1)
columns = flights.columns

print(flights.dtypes)
 ```
 6. Extract categrical features, Encode categrical features with Ordinal Encoder.
Library preprocessing is used for Encoding.
```python
from sklearn import preprocessing
num_flights = flights.loc[:,['DEPARTURE_DELAY',"AIR_TIME","DISTANCE"]]
one_hot_original = flights.loc[:,['AIRLINE',"DAY_OF_WEEK","MONTH"]]
ordinal_orginal = flights.loc[:,['DAY',"ORIGIN_AIRPORT","DESTINATION_AIRPORT"]]

display(one_hot_original)
one_hot_encode=one_hot_original.astype("str")
encoder = preprocessing.OrdinalEncoder()
encoder.fit(one_hot_encode)
ordinal_cat = encoder.transform(one_hot_encode)
print(ordinal_cat, ordinal_cat.shape)

display(ordinal_orginal)
ordinal_orginal["DAY"] = ordinal_orginal["DAY"].astype("str")
ordinal_orginal.dtypes
ordinal_orginal=ordinal_orginal.astype("str")
encoder = preprocessing.OrdinalEncoder()
encoder.fit(ordinal_orginal)
ordinal = encoder.transform(ordinal_orginal)
print(ordinal, ordinal.shape)
```

 7. Build the Dataframe after partically encoding dataset.

```python
df_ordinal = pd.DataFrame(ordinal, columns = ['DAY','ORIGIN_AIRPORT','DESTINATION_AIRPORT'])
df_cat = pd.DataFrame(ordinal_cat, columns = ['AIRLINE','DAY_OF_WEEK','MONTH'])

df_flights = num_flights.join(df_ordinal)
df_flights = df_flights.join(df_cat)
```

 8. Scale the dataset and keep the dataset as Dataframe.
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df_flights)
df_flight_noscale =df_flights
df_flights = scaler.transform (df_flights)

# Make df_flights into Dataframe
df_flights = pd.DataFrame(df_flights)
df_flights = df_flights.rename(columns={0:'DEPARTURE_DELAY',1:'AIR_TIME',2:'DISTANCE',3:'DAY',4:'ORINGIN_AIRPORT',5:'DESTINATION_AIRPORT',6:'AIRLINES',7:'DAY_OF_WEEK',8:'MONTH' })
target = flights["ARRIVAL_DELAY"]
df_flights = df_flights.join(target)
df_flights
```
 
 9. Clean up NAN Values
```python
# check NAN values
display(df_flights)
display(df_flights.isna().any())

df_flights['DEPARTURE_DELAY'] = df_flights['DEPARTURE_DELAY'].fillna(0)
df_flights['ARRIVAL_DELAY'] = df_flights['ARRIVAL_DELAY'].fillna(0)

print("After filling DEPARTURE_DELAY, and ARRIVAL_DELAY:")
display(df_flights.isna().any())

df_flights = df_flights.dropna()
display(df_flights)
print("after deleting all rows that contain NaN values,")
display(df_flights.isna().any())
```
 #### Models Training
 
 Note: Only the code of building models is displaying below. Plotting/graphs and Full code with comments for Models Training can be found in "Model_Training.ipynb"
 
```python
```
 1. Linear Regression (by SGD Regressor)

```python
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
reg = SGDRegressor()
reg.fit(X=X_train, y=y_train)

y_pred = reg.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred)

print("Model weights: ")
print(reg.coef_)
print('Testing MSE:',mse1)
print("Model score:",reg.score(X_test, y_test) )
```

 2. Poylnomial Regression

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform((X_train))
plr = LinearRegression()
# Note that I didn't do reshape on X_poly as it's already a matrix.
plr.fit(X_poly, (y_train))
    
predicted = plr.predict(poly.transform((X_test)))
    
display(plr.intercept_)
display(plr.coef_[0:3])

print(f'Polynomial regression with degree = {3}')
print(f'Training MSE error is:',mean_squared_error(plr.predict(X_poly), y_train))
print(f'Testing MSE error is:', mean_squared_error(predicted, y_test))
print("Model score:",plr.score(X_test, y_test))
```

 3. Lasso Regression

```python
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
reg.fit(X_train, y_train)

y_train_pred = reg.predict(X_train)

train_mse = mean_squared_error(y_train,y_train_pred)
print('Training MSE:',train_mse)
y_hat = reg.predict(X_test)
print(f'Testing MSE error is: {round(mean_squared_error(y_hat, y_test),4)}')
print("Model score:",reg.score(X_test, y_test))
```

 4. Ridge Regression

```python
from sklearn import linear_model
reg = linear_model.Ridge(alpha=.5)
reg.fit(X_train, y_train)

y_train_pred = reg.predict(X_train)
# plt.figure(figsize=(7, 7))
# sns.scatterplot(X_train, y_train)
# plt.show()

train_mse = mean_squared_error(y_train,y_train_pred)
print('Training MSE:',train_mse)
y_hat = reg.predict(X_test)
print(f'Testing MSE error is: {round(mean_squared_error(y_hat, y_test),4)}')

print("Model weights: ")
print(reg.coef_)

z = reg.score(X_test, y_test)
print("Accuracy score: ",z)
```

## Results <br/>

## Discussion <br/>

## Conclusion <br/>
 
## Group Members:
Zhaolin Zhong, 
Shang Wu,
Xueqi Zhang,
Huaiyu Jiang,
Kejing Chen

