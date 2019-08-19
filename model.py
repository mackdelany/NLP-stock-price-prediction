import numpy as np
import pandas as pd
from pathlib import Path

data_folder = Path('data-interim')
text_file_path = data_folder / 'text_features.csv'
numbers_file_path = data_folder / 'numbers.csv'
labels_file_path =  data_folder / 'interim_labels.csv'

text = pd.read_csv(text_file_path)
labels = pd.read_csv(labels_file_path)


data = text.merge(export, how='inner', left_on='Date', right_on='Date')