# Anomaly Detection using LSTM Autoencoder - LSTM Encoder Decoder Architecture
# Here Decoder is used as predictor?????
# Mostly anomalies seem to be collective anomalies

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
df = pd.read_csv('real_10.csv', index_col = 'timestamp')
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

# Remove is_anomaly = 1 rows from only train data
train = train.drop(train[train.is_anomaly == 1].index)
print('new_train_count',train['value'].count())

# Just to confirm no anomaly in training set
c = train.loc[train.is_anomaly == 1]
total_rows = c['is_anomaly'].count()
print('is_anomaly_training_count',total_rows)

# Anomalies still present in test set
d = test.loc[test.is_anomaly == 1]
total_rows = d['is_anomaly'].count()
print('is_anomaly_test_count',total_rows)

# Remove is_anomaly column from train and test data since semi-supervised learning
del train['is_anomaly']
del test['is_anomaly']

print(train.shape, test.shape)

# Scaling of data
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler = scaler.fit(train[['value']])

train['value'] = scaler.transform(train[['value']])
test['value'] = scaler.transform(test[['value']])

# We’ll split the data into subsequences - changing input to a shape as accepted by
# lstm autoencoder

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 10
# It tries to reconstruct 11th sample taking into account previous 10 timestamps
# If only autoencoder it would try to reconstruct 1st sample same as 1st sample,
# wnd same as 2nd & so on
# If it is LSTM Autoencoder it tries to reconstruct taking previous timestamps into account
# If timesteps = 10 then when it reconstructs 11th sample it takes into account
# 1 to 10 samples, then based on 2 to 11 it takes 12th sample & try to reconstruct

# Reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(
  train[['value']],
  train.value,
  TIME_STEPS
)
X_test, y_test = create_dataset(
  test[['value']],
  test.value,
  TIME_STEPS
)
print(X_train.shape)

# Model Architecture

model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=64,
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(
  keras.layers.TimeDistributed(
    keras.layers.Dense(units=X_train.shape[2])
  )
)
model.compile(loss='mae', optimizer='adam')

# Training Model

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=10,
    validation_split=0.1,
    shuffle=False
)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();
plt.show()

X_train_pred = model.predict(X_train)

train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

sns.distplot(train_mae_loss, bins=50, kde=True);
plt.show()
# From distribution plot mostly all data comes within  value upto 1.5/0.6

X_test_pred = model.predict(X_test)

test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
train_mae_loss = train_mae_loss.flatten()

def statistical_threshold(loss_function):
    import statistics

    m = statistics.mean(loss_function)
    sd = statistics.stdev(loss_function, xbar=m)

    # 3 standard deviation as outlier
    new_std_dev = sd * 3
    threshold = m + (sd*3)

    print("Mean:", m, "SD:", sd, "New SD:", new_std_dev)
    return threshold


THRESHOLD = statistical_threshold(train_mae_loss)
print('statistical threshold is', THRESHOLD)

# Even statistical threshold comes to be same 1.44 as seen from distribution plot

test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['value'] = test[TIME_STEPS:].value

plt.plot(test_score_df.index, test_score_df.loss, label='loss')
plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
plt.xticks(rotation=25)
plt.legend();
plt.show()

anomalies = test_score_df[test_score_df.anomaly == True]
print(anomalies)


