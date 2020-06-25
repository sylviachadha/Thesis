# Anomaly detection on Yahoo Benchmark

#--------------------------
# Step 1 Import Libraries
#--------------------------

import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pdf as pdf
import tensorflow as tf
import os
from tensorflow import keras
import seaborn as sns
from scipy.stats import norm
from scipy.stats import lognorm
from scipy import stats
import math

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

#----------------------
# Step 2 Load Dataset
# ---------------------
os.chdir('/Users/sylviachadha/Desktop/Anomaly_Detection/Practice_Dataset')
df = pd.read_csv('real_51.csv', index_col = 'timestamp')
print(df.head(6))

#-------------------
# Step3 Plot data
#-------------------
ax = df['value'].plot(figsize = (12,6), title = 'yahoo traffic data');
xlabel = 'hourly data'
ylabel = 'traffic - yahoo properties'
ax.set(xlabel = xlabel, ylabel = ylabel)
plt.show()


# ------------------------------------
# Step4 Train, Validation, Test Split
# ------------------------------------

train_size = 0.70
validation_size = 0.15
test_size = 0.15


train, remain = train_test_split(df, test_size=(validation_size + test_size),shuffle=False)

new_test_size = np.around(test_size / (validation_size + test_size), 2)
# To preserve (new_test_size + new_val_size) = 1.0
new_val_size = 1.0 - new_test_size

val, test = train_test_split(remain, test_size=new_test_size,shuffle=False)

print(train)
print(val)
print(test)

len(df)

# Store Actual labels for val data to use for threshold decision later

no_of_time_steps=10
print(val)
len(val)
val_actual = val.iloc[no_of_time_steps:]
print(val_actual)

# Store Actual labels for test data to use for confusion matrix evaluation

no_of_time_steps=10
print(test)
len(test)
test_actual = test.iloc[no_of_time_steps:]
print(test_actual)

# Check ratio of classes (Normal vs Anomalies)
df['is_anomaly'].value_counts()

# To check how many anomalies are present in Training and how many in validation and test
a = train.loc[train.is_anomaly == 1]
print(a)
total_rowsa = a['is_anomaly'].count()
print('is_anomaly_training_count',total_rowsa)


b = val.loc[val.is_anomaly == 1]
print(b)
total_rowsb = b['is_anomaly'].count()
print('is_anomaly_val_count',total_rowsb)

c = test.loc[test.is_anomaly == 1]
print(c)
total_rowsc = c['is_anomaly'].count()
print('is_anomaly_test_count',total_rowsc)

# Anomalies still present in test set
d = test.loc[test.is_anomaly == 1]
total_rows = d['is_anomaly'].count()
print('is_anomaly_test_count',total_rows)

# Remove is_anomaly = 1 rows from only train data
train = train.drop(train[train.is_anomaly == 1].index)
print('new_train_count',train['value'].count())

# Just to confirm no anomaly in training set
c = train.loc[train.is_anomaly == 1]
total_rows = c['is_anomaly'].count()
print('is_anomaly_training_count',total_rows)

# Remove is_anomaly column from train and test data since semi-supervised learning
# For validation do not remove "is_anomaly" since need actual labels to decide on threshold
del train['is_anomaly']
del test['is_anomaly']

print(train.shape, test.shape)

#-------------------------
# Step 5 Scaling of data
#-------------------------
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler = scaler.fit(train[['value']])

train['value'] = scaler.transform(train[['value']])
val['value'] = scaler.transform(val[['value']])
test['value'] = scaler.transform(test[['value']])


#-----------------------------------
# Step 6 - Prepare Input for LSTM
#-----------------------------------
# We’ll split the data into sub-sequences - changing input to a shape as accepted by
# lstm autoencoder

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS=10

# Reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(
  train[['value']],
  train.value,
  TIME_STEPS
)

X_val, y_val = create_dataset(
  val[['value']],
  val.value,
  TIME_STEPS
)

X_test, y_test = create_dataset(
  test[['value']],
  test.value,
  TIME_STEPS
)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

#-----------------------------------
# Step 7 Define Model Architecture
#-----------------------------------

model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=32,
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
model.add(keras.layers.LSTM(units=32, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(
  keras.layers.TimeDistributed(
    keras.layers.Dense(units=X_train.shape[2])
  )
)
model.compile(loss='mae', optimizer='adam')

#----------------------------
# Step 8 Training/Fit Model
#----------------------------

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=10,
    validation_split=0.1,
    shuffle=False
)

#------------------------------------------------------------
# Step 9 Predict & decide parameters using MLE on Training data
#------------------------------------------------------------
X_train_pred = model.predict(X_train)

train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

# Show distribution plot of training loss
sns.distplot(train_mae_loss, bins=50, kde=True);
plt.show()

train_mae_loss = train_mae_loss.flatten()

# We normally assume distributions on errors/residuals
# Assume guassian distribution for prediction errors on training data
train_mae_loss

# Use MLE to estimate best parameters for above prediction errors
parameters = norm.fit(train_mae_loss)
print ("Mean and S.D obtained using MLE on training prediction errors is",parameters)

#-----------------------------------------------
# Step 10 - Method to decide on Threshold value
#-----------------------------------------------

# Apply trained model on validation set and get validation prediction errors
# Adding anomalies of Training as well to validation


X_val_pred = model.predict(X_val)

val_mae_loss = np.mean(np.abs(X_val_pred - X_val), axis=1)
len(val_mae_loss)

# Compute probability values for errors on validation set

prob_val_mae_loss = norm.pdf(val_mae_loss, loc=parameters[0], scale=parameters[1])
prob_val_mae_loss = prob_val_mae_loss.flatten()
prob_val_mae_loss_log = np.log(prob_val_mae_loss)
prob_val_mae_loss_log

# A threshold is set on the log probability distribution values

#----------------#
# Assumptions
#----------------#
# 1. An anomaly is an observation which is suspected of being partially or wholly irrelevant
# because it is not generated by the stochastic model assumed.Above we assumed guassian
# distribution on training errors train_mae_loss

# 2. Normal data instances occur in high probability regions of a stochastic model,
# while anomalies occur in the low probability regions of the stochastic model.
# This assumption is checked below where low Probability density instances should be
# captured as anomalies

error_df = pd.DataFrame({'probability_val_set': prob_val_mae_loss_log,
                        'True_class_val_set': val_actual['is_anomaly']})

precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class_val_set,error_df.probability_val_set)

plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)

plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)

plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

# See values of different thresholds as per precision_recall_curve function
print(threshold_rt)
print(precision_rt)
print(recall_rt)

# Choose threshold which maximize the F1 Score

f1_scores = 2*recall_rt*precision_rt/(recall_rt+precision_rt)

selected_threshold = threshold_rt[np.argmax(f1_scores)]
print('Best threshold: ', threshold_rt[np.argmax(f1_scores)])
print('Best F1-Score: ', np.max(f1_scores))

#-----------------------------------------------
# Step 13 - Get predictions on test set
#-----------------------------------------------

X_test_pred = model.predict(X_test)

test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
len(test_mae_loss)

# Compute probability values for errors on test set

prob_test_mae_loss = norm.pdf(test_mae_loss, loc=parameters[0], scale=parameters[1])
prob_test_mae_loss = prob_test_mae_loss.flatten()
prob_test_mae_loss_log = np.log(prob_test_mae_loss)
prob_test_mae_loss_log

#-----------------------------------------------
# Step 12 - Evaluate this threshold on test set
#-----------------------------------------------

test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
test_score_df['loss'] = prob_test_mae_loss_log
test_score_df['threshold'] = selected_threshold
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['anomaly'] = test_score_df['anomaly'].astype(int)
test_score_df['value'] = test[TIME_STEPS:].value

#--------------------------------------------
# Step 13 Plot Test Loss and Threshold
#---------------------------------------------
plt.plot(test_score_df.index, test_score_df.loss, label='loss')
plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
plt.xticks(rotation=25)
plt.legend();
plt.show()

#---------------------------------------------
# Step 14 Print Anomalies Actual and Predicted
#---------------------------------------------

anomalies = test_score_df[test_score_df.anomaly == 1]
print ('ACTUAL ANOMALIES count',len(d))
print('ACTUAL ANOMALIES',d)
print ('DETECTED ANOMALIES count',len(anomalies))
print('DETECTED ANOMALIES', anomalies)

#---------------------------------------------------------------------
# Step 15 Evaluation metrics with test_actual and test_score_df
# 1. Confusion Matrix
# 2. Accuracy
# 3. Precision
# 4. Recall
#---------------------------------------------------------------------

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(test_actual['is_anomaly'],test_score_df['anomaly'])
print('Confusion Matrix')
print(cf)

# Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(test_actual['is_anomaly'],test_score_df['anomaly'])
print('Accuracy')
print(acc)

# Recall
from sklearn.metrics import recall_score
recall = recall_score(test_actual['is_anomaly'],test_score_df['anomaly'])
print('Recall')
print(recall)

# Precision
from sklearn.metrics import precision_score
precision = precision_score(test_actual['is_anomaly'],test_score_df['anomaly'])
print('Precision')
print(precision)
