import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == '__main__':
    news = pd.read_csv(Path('data/train/', 'RedditNews_train.csv'))

    max_features = 200 # note this hyperparameter

    # Drop NAs
    news.dropna(inplace=True, axis=0)

    # Change text to lower case
    news['News'] = news['News'].apply(lambda x: x.lower())

    # Concat stories by day
    news = news.groupby(['Date'])['News'].apply(lambda x: ', '.join(x)).reset_index()

    # Note max features as a hyperparameter!!
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)

    news_vectors = vectorizer.fit_transform(news['News'].value).toarray()

    text_features = pd.DataFrame(data=news_vectors, columns=vectorizer.get_feature_names())

    # Concat vectors with dates
    text_features = pd.concat([pd.Series(news.Date.unique()).to_frame(name='Date'), text_features], axis=1)

    os.makedirs('./data/interim', exist_ok=True)
    text_features.to_csv('data/interim/text_features.csv',index=False)
