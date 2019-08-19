import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

data_folder = Path('data/train/')
file_to_open = data_folder / 'RedditNews_train.csv'

news = pd.read_csv(file_to_open)

max_features = 200 # note this hyperparameter

# Drop NAs
news.dropna(inplace=True)

# Change text to lower case
news['News'] = news['News'].apply(lambda x: x.lower())

# Concat stories by day
news = news.groupby(['Date'])['News'].apply(lambda x: ', '.join(x)).reset_index()

# Import TfidfVectorizer
# Note max features as a hyperparameter!!
vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
corpus = news['News'].values
news_vectors = vectorizer.fit_transform(corpus).toarray()
text_features = pd.DataFrame(data=news_vectors,columns=vectorizer.get_feature_names())

# Concat vectors with dates
text_features = pd.concat([pd.Series(news.Date.unique()).to_frame(name='Date'), text_features], axis=1)

# Export csv
text_features.to_csv('data-interim/text_features.csv',index=False)