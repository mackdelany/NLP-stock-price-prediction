import numpy as np
import pandas as pd
from pathlib import Path

data_folder = Path('data/interim')
text_file_path = data_folder / 'text_features.csv'
numbers_file_path = data_folder / 'number_data.csv'
labels_file_path =  data_folder / 'labels.csv'

text = pd.read_csv(text_file_path)
numbers = pd.read_csv(numbers_file_path)
labels = pd.read_csv(labels_file_path)

data = text.merge(labels, how='inner', left_on='Date', right_on='Date')\
    .merge(numbers, how='inner', left_on='Date', right_on='Date')

data = data.dropna()
data = data.drop('Date', axis=1)


print(data.head())