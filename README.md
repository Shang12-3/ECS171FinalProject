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
 ### Data Downloads
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
 ### Data Preprocessing
 
 ### Models Training
 

## Group Members:
Zhaolin Zhong, 
Shang Wu,
Xueqi Zhang,
Huaiyu Jiang,
Kejing Chen

