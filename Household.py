# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:10:17 2020

@author: deepak
"""

from pandas import read_csv
from keras.models import *
from keras.layers import *
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.pipeline import Pipeline
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

df = read_csv("household.txt", delim_whitespace = True)

feature_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df.columns = feature_name
print(df.head())

df = df.rename(columns = {'MEDV' : 'Price'})
print(df.describe())

##### Heatmap

mask = np.zeros_like(df.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True

plt.figure(figsize = (16, 10))
sns.heatmap(df.corr(), mask = mask, annot = True)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()

X = df.drop('Price', axis = 1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


###Scalerization

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


### define model 

model = Sequential()
model.add(Dense(256, input_dim = 13, activation = 'relu'))
model.add(Dense(128, init = 'uniform', activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(56, activation = 'relu'))
model.add(BatchNormalization())
#output layer
model.add(Dense(1, activation = 'linear'))

model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mae'])
model.summary()

history = model.fit(X_train_scaled, y_train, validation_split = 0.2, epochs = 100)

from matplotlib import pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'b', label = 'Training_loss')
plt.plot(epochs, val_loss, 'r', label = 'validation_loss')
plt.title('Training and validation loss')
plt.xlabel ("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


acc = history.history['mae']
val_acc = history.history['val_mae']
plt.plot(epochs, acc, 'b', label = 'Trainig_MAE')
plt.plot(epochs, val_acc, 'r', label = 'Validation_MAE')
plt.title("Training and Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

##########################################################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

### pridict on test set

predictions = model.predict(X_test_scaled[:5])
print("Predicted values are : ", predictions)
print("Real values are : ", y_test[:5])

##############################################
mse_neural, mae_neural = model.evaluate(X_test_scaled, y_test)
y_pred = model.predict(X_test_scaled)
r2_neural = r2_score(y_test, y_pred)
print("R_2 value for Neural Network is :", r2_neural)
print("Mean Squared Error from Neural Network ", mse_neural)
print("Mean Absolute Error from Neural Network", mae_neural)

################################################
## Linear Model

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
###linear Regression
lr_model = linear_model.LinearRegression(normalize = True, n_jobs = -1)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("R_2 value for linear Regression is :", r2_lr)
print("Mean Squared Error from linear Regression ", mse_lr)
print("Mean Absloute Error froim linear Regression ", mae_lr)


########################
## Decision Tree

tree = DecisionTreeRegressor()
tree.fit(X_train_scaled, y_train)
y_pred_tree = tree.predict(X_test_scaled)
mse_tree = mean_squared_error(y_test, y_pred_tree)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)
print("R_2 value for Decision Tree is :", r2_tree)
print("Mean Squared Error from Decision Tree Regression ", mse_tree)
print("Mean Absloute Error froim Decision Tree Regression ", mae_tree)

##################################
##Random Forest

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators = 25, random_state = 40)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("R_2 value for Random Forest is :", r2_rf)
print("Mean Squared Error from Random Forest Regression ", mse_rf)
print("Mean Absloute Error froim Random Forest Regression ", mae_rf)

######Heatmap for independent variables

#######
