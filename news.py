import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

news = pd.read_csv('/data/train/RedditNews_train.csv')

# Drop NAs
news.dropna(inplace=True)

# Change text to lower case
news['News'] = news['News'].apply(lambda x: x.lower())

# Concat stories by day
news = news.groupby(['Date'])['News'].apply(lambda x: ', '.join(x)).reset_index()

# Import CountVectorizer
# Note max features as a hyperparameter!!
vectorizer = CountVectorizer(max_features=200)
corpus = news['News'].values
news_vectors = vectorizer.fit_transform(corpus).toarray()

# Concat vectors with dates
news = pd.concat([news.Date, pd.DataFrame(news_vectors)], axis=1)

# Export csv
news.to_csv('processed_news.csv')