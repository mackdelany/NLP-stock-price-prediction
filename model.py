import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

data_folder = Path('data/interim')
text_file_path = data_folder / 'text_features.csv'
numbers_file_path = data_folder / 'clean_DJIA.csv'
labels_file_path =  data_folder / 'prototype_labels.csv'

text = pd.read_csv(text_file_path)
numbers = pd.read_csv(numbers_file_path)
labels = pd.read_csv(labels_file_path)

data = text.merge(labels, how='inner', left_on='Date', right_on='Date')\
    .merge(numbers, how='inner', left_on='Date', right_on='Date')

data = data.dropna()

data['Date'] = pd.to_datetime(data['Date'])
data['dayofweek'] = data.Date.dt.dayofweek
data['dayofmonth'] = data.Date.dt.day
data['monthofyear'] = data.Date.dt.month

data = data.drop('Date', axis=1)
data = data.drop(['Open','High','Low','Close','Volume','Adj Close', 'Lag_Vol'], axis=1)

scaler = MinMaxScaler()
data['1st_PC'] = scaler.fit_transform(data['1st_PC'].values.reshape(-1,1))
data['2nd_PC'] = scaler.fit_transform(data['2nd_PC'].values.reshape(-1,1))
data['dayofweek'] = scaler.fit_transform(data['dayofweek'].values.reshape(-1,1))
data['dayofmonth'] = scaler.fit_transform(data['dayofmonth'].values.reshape(-1,1))
data['monthofyear'] = scaler.fit_transform(data['monthofyear'].values.reshape(-1,1))

print(data.head())

data.to_csv('data/final_data.csv', index=False)