# https://github.com/curiousily/Deep-Learning-For-Hackers/blob/master/14.time-series-anomaly-detection.ipynb# Using Tensorflow2 and Keras

# LSTM Autoencoder using LSTM Encoder-decoder architecture
# Encoder will always encode
# Decoder will predict here

# -------------------------
# Step1 Import Libraries
# -------------------------
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

# -------------------------
# Step2 Read csv dataset
# -------------------------
# Load Dataset
os.chdir('/Users/sylviachadha/Desktop/Anomaly_Detection/Practice_Dataset')
df = pd.read_csv('real_2.csv', index_col='timestamp')
print(df.head(6))

# Index column timestamp data type should be datetime (hourly data)
df.index.freq = 'H'
df.index
df.index = pd.to_datetime(df.index, origin=pd.Timestamp('2020-01-01'), unit='h')
df.index
print(df.shape)

# ---------------------------
# Step3 Plot of dataset
# ---------------------------

ax = df['value'].plot(figsize=(12, 6), title='yahoo traffic data')
xlabel = 'hourly data'
ylabel = 'traffic - yahoo properties'
ax.set(xlabel=xlabel, ylabel=ylabel)
plt.show()

# ----------------------------------------
# Step 4 Split into Train and Test set
# ----------------------------------------
train_size = int(len(df) * 0.90)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)

del train['is_anomaly']
del test['is_anomaly']

print(train.shape,test.shape)

# ----------------------------------------------------------------------------
# Step 5 Scale data, important for algorithms using gradient descent approach
# ----------------------------------------------------------------------------
scaler = RobustScaler()
scaler = scaler.fit(train[['value']])

train['value'] = scaler.transform(train[['value']])
test['value'] = scaler.transform(test[['value']])

# View of scaled data
train.head()


# -------------------------------------------------------------------
# Step 6 Need to prepare input in a way as expected by LSTM Model
# LSTM is type of RNN, so it expects data in sequences
# -------------------------------------------------------------------
# We will create sequences taking 1 day of history
# 1-24 ->> 25th hour
# 2-25 ->> 26th hour

def create_dataset(X, Y, time_steps=1):
    Xs, Ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        Ys.append(Y.iloc[i + time_steps])
    return np.array(Xs), np.array(Ys)


TIME_STEPS = 10

X_train, y_train = create_dataset(
    train[['value']],
    train.value,
    TIME_STEPS
)

print(X_train.shape)
print(X_train.shape[0])  # Samples
print(X_train.shape[1])  # Timesteps
print(X_train.shape[2])  # Features

X_test, y_test = create_dataset(
    test[['value']],
    test.value,
    TIME_STEPS
)

print(X_train.shape)

# -------------------------------------------------------------------
# Step 7 Architecture
# LSTM Autoencoder with decoder as Predictor ~ LSTM
# They are rnn & need some sort of sequence as an input
# This is Encoder Decoder LSTM Architecture
# -------------------------------------------------------------------
# Autoencoder is not a layer in keras, its an architecture or a
# type of nn that tries to reconstruct its data
# Autoencoder with encoder-decoder as LSTM ~ LSTM Autoencoder
# --------------------------------------------------------------------
model = keras.Sequential()

model.add(keras.layers.LSTM(
    units=64,
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dropout(rate=0.2))
# Dropout is a technique used to prevent a model from overfitting.
# Dropout works by randomly setting the outgoing edges of hidden units
# (neurons that make up hidden layers) to 0 at each update of the training phase

model.add(keras.layers.RepeatVector(n=X_train.shape[1]))

model.add(keras.layers.LSTM(
    units=64,
    return_sequences=True)
)
model.add(keras.layers.Dropout(rate=0.2))

# Output is time distributed fully connected layer
# This layer will provide us with same units we passed as i/p
#
model.add(
    keras.layers.TimeDistributed(
        keras.layers.Dense(units=X_train.shape[2])
    )
)

model.compile(loss='mae', optimizer='adam')

# --------------------------------------------
# Step 8 Training model / Fit Model
# --------------------------------------------

fitted_model = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)
