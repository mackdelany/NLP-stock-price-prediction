import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def process_text_features(filepath, max_features=200, create_csv=False, return_frame=True, csv_directory='./data/interim', csv_path='data/interim/text_features.csv'):

    news = pd.read_csv(filepath).dropna(inplace=True, axis=0)
    news['News'] = news['News'].apply(lambda x: x.lower())
    news = news.groupby(['Date'])['News'].apply(lambda x: ', '.join(x)).reset_index()

    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    news_vectors = vectorizer.fit_transform(news['News'].value).toarray()
    text_features = pd.DataFrame(data=news_vectors, columns=vectorizer.get_feature_names())

    text_features = pd.concat([pd.Series(news.Date.unique()).to_frame(name='Date'), text_features], axis=1)

    if create_csv == True:
        os.makedirs(csv_directory, exist_ok=True)
        text_features.to_csv(csv_path,index=False)

    if return_frame == True:
        return text_features


if __name__ == '__main__':
    
    filepath = Path('data/train/', 'RedditNews_train.csv')
    process_text_features(filepath, create_csv=True, return_frame=False)

