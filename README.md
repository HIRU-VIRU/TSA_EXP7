# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 30-9-2025

#### NAME:HIRUTHIK SUDHAKAR
#### REGISTER NUMBER:212223240054

### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```PYTHON

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from statsmodels.datasets import sunspots

data = sunspots.load_pandas().data
data.columns = ['date', 'sunspots']

# Convert year to datetime and set as index
data['date'] = pd.to_datetime(data['date'], format='%Y')
data.set_index('date', inplace=True)

ts = data['sunspots']

# ADF Test
result = adfuller(ts.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

# Split into train/test
x = int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

plot_acf(ts.dropna(), lags=30)
plot_pacf(ts.dropna(), lags=30)
plt.show()

train_size = int(len(ts) * 0.8)
train, test = ts[0:train_size], ts[train_size:]

# Ensure lag < number of training points
lags = min(5, len(train)-1)

# Fit AR model
model = AutoReg(train, lags=lags).fit()
print(model.summary())

# Forecast
preds = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# MSE
error = mean_squared_error(test, preds)
print("MSE:", error)

# Plot
plt.figure(figsize=(10,5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, preds, label='Predicted', color='red')
plt.legend()
plt.show()


```
### OUTPUT:

### GIVEN DATA
<br>
<img width="229" height="178" alt="image" src="https://github.com/user-attachments/assets/cbe42be7-cba8-4adf-aa95-0cc24cf1b066" />
<br>

### PACF - ACF
<br>
<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/4b08daa4-4969-4d9d-8a56-ab07aea634b0" />
<br>
<br>
<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/f27ceb7b-d286-41cd-be84-9d0383f13644" />
<br>

### PREDICTION
<br>
<img width="302" height="87" alt="image" src="https://github.com/user-attachments/assets/a9f3f6f1-b934-4479-afda-0f4f9fb2335b" />

<br>

### FINIAL PREDICTION
<img width="836" height="428" alt="image" src="https://github.com/user-attachments/assets/3d701547-aad8-4ea3-b431-bd8fea596c5a" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
