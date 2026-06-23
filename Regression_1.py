# importing needed lib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

train_df = pd.read_csv('D:/Downloads/train.csv')
print(train_df.head())

test_df = pd.read_csv('D:/Downloads/test.csv')
print(test_df.head())

print(train_df['SalePrice'].mean())

test_df['SalePrice'] = pd.Series([train_df['SalePrice'].mean()]*len(train_df))
print(test_df)

test_df[['Id', 'SalePrice']].to_csv('average_prediction.csv', index=False)

from copy import deepcopy

train_df_sub = deepcopy(train_df[['MSSubClass', 'LotFrontage', 'SalePrice']])

print(train_df_sub)

print(train_df_sub.mean())

train_df_sub = train_df_sub.fillna(train_df_sub.mean())
X_train, y_train = train_df_sub.to_numpy()[:, :-1], train_df_sub.to_numpy()[:, -1]
lr = LinearRegression().fit(X_train, y_train)
print(lr)

print(lr.coef_)
print(lr.intercept_)
train_df_sub['Linear_predictions'] = lr.predict(X_train)
print(train_df_sub)

print(mean_absolute_error(train_df_sub['SalePrice'], train_df_sub['Linear_predictions']))
test_df_sub = deepcopy(test_df[['MSSubClass', 'LotFrontage']])
print(test_df_sub)

test_df_sub = test_df_sub.fillna(test_df_sub.mean())
print(test_df_sub)

x_test = test_df_sub.to_numpy()
print(x_test)

test_df['SalePrice'] = lr.predict(x_test)
print(test_df)

