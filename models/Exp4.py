# Just to explore types of anomalies in Yahoo benchmark dataset

# Import libraries
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow import keras
import seaborn as sns

# Load Dataset
os.chdir('/Users/sylviachadha/Desktop/Anomaly_Detection/Practice_Dataset')
df = pd.read_csv('real_20.csv', index_col = 'timestamp')
print(df.head(6))

# Index column timestamp data type should be datetime (hourly data)
df.index.freq = 'H'
df.index
df.index = pd.to_datetime(df.index, origin=pd.Timestamp('2020-01-01'), unit = 'h')
df.index

# Plot data (showing point anomalies)
ax = df['value'].plot(figsize = (12,6), title = 'yahoo traffic data');
xlabel = 'hourly data'
ylabel = 'traffic - yahoo properties'
ax.set(xlabel = xlabel, ylabel = ylabel)
plt.show()

# We’ll use 90% of the data and train our model on it:
train_size = int(len(df) * 0.90)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size],df.iloc[train_size:len(df)]
print(train.shape,test.shape)

# To check how many anomalies are present in Training and how many in test
a = train.loc[train.is_anomaly == 1]
print(a)
total_rows = a['is_anomaly'].count()
print('is_anomaly_training_count',total_rows)

b = test.loc[test.is_anomaly == 1]
print(b)
total_rows = b['is_anomaly'].count()
print('is_anomaly_test_count',total_rows)

print(min(df['value']))
print(max(df['value']))
a = np.mean(df['value'])
b = np.std(df['value'])
c = (a + 3*b)
print('c is',c)
