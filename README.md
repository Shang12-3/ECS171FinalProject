# ECS171FinalProject

## Table of Contents:
  - Project Purpose
  - Dataset information
  - Setup (download data)
  - Preprocessing
  - Plot
  - Group Members
 


## Project Purpose:
The goal of this project is to predict flight delay conditions mainly based on locations of destination and origin, flight distance and dates. The output is expected to be a classification index. 

## Dataset information:
The dataset is found from Kaggle's “2015 Flight Delays and Cancellations”, originally abstracted from the U.S. Department of Transportation (DOT). 
Link: "https://www.kaggle.com/datasets/usdot/flight-delays?select=flights.csv"

Three csv files are included:
  airlines.csv: include two columns with Airline identifier and Airline full name.
  airports.csv: detail information for each airport (includes LATITUDE and LONGITUDE)
  flights.csv: detail of each flight (5819079 samples) in USA from 2015. Model will mainly relay on this data. 

## Setup(download data):
First two cell from FinalProject.ipynb is for setting up the data.Three csv files mentioned previously are saved in the googledrive. Please run the cells to download the data files. 

If not avaliable for download please use the Kaggle link or visit "https://drive.google.com/drive/folders/1vx8tEnQxPC4WL4bhLlQ0A6NpDX0VQQtT?usp=share_link".

## Preprocessing:

#### 1. Drop unrelated features:
      These following features are about delay reasons which implies flight is already delayed, since our goal is to predict if the flight is delayed, these should not be trained in the model as features.
      Drop 'WEATHER_DELAY','LATE_AIRCRAFT_DELAY','AIRLINE_DELAY','SECURITY_DELAY','AIR_SYSTEM_DELAY','CANCELLATION_REASON', "CANCELLED", "DIVERTED","TAXI_IN", "TAXI_OUT"
      
#### 2. Drop overlap features:
     These following features are about overlapped, for example 'SCHEDULED_ARRIVAL' and 'ARRIVAL_TIME', the difference between these two features is also included as "ARRIVAL_DELAY". There is no need to have them. Also, 'TAIL_NUMBER' and 'FLIGHT_NUMBER' has the same function to identify the index of each sample. 
    Drop 'SCHEDULED_ARRIVAL','ARRIVAL_TIME','DEPARTURE_TIME','SCHEDULED_DEPARTURE','TAIL_NUMBER','FLIGHT_NUMBER'
 
 #### 3. Drop 'Year':
     Since all data is collected in 2015.
     
* After dropping columns: Dataframe 'flights' has shape 5819079 rows × 10 columns.

 #### 4. Encode object features 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT':
    By observeing these three features, they are using string(code name) to represent airline and airports. So it needs to be encoded as numerical types for training model. Based on various types of airlines and airports, here OrdinalEncoder is more practial than OnehotEncoder().
    
* After encoding, dataframe prepared is 'df_flights'.
 #### 5. Scaler:
    Implemented MinMaxScaler on the dataframe 'df_flights'. Prepare for model training process.
    
## Plot:
    Plotting is applied with correlations and sns.pairplot. (Same procedure). Correlations do not provided a clear assumption due to the multiple features and real life conditions. Based on pairplot distributions, most of features have the graph as Unimodal Distributions which only contains one peak (reasonable guess is the peak could impact more on our prediction). Also, there are some Multimodal Distributions, for example on "day of week", "AIRLINE", "ORIGIN_AIRPOT", "DESTINATION_AIRPORT". Before identifying the impact of each airport, reasonable guess could be a few specific airports could cause higher probability of delaying flights.


## Future Work:
1. The dataset from airlines and airports should not be applied until we have trained the models and explaining the features with coefficeients. (These two datasets are more related as supplementary material  for flights)
2. Based on the distribution graph, "MONTH" and "DAY" seems having very little impact. There could be some adjustment after viewing how the model is built.
3. We were thinking to build a model with different regressions, but due to the huge sample amounts, we might applied neural network to take the advantage of batch_sizes. 


## Group Members:
Zhaolin Zhong, 
Shang Wu,
Xueqi Zhang,
Huaiyu Jiang,
Kejing Chen
