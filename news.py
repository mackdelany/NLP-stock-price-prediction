import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer


def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lmtzr.lemmatize(word, tag))
            
    return " ".join(res_words)


def process_text_features(filepath, max_features=200, create_csv=True, return_frame=True, csv_directory='./data/interim', csv_path='data/interim/text_features.csv', dict_path='data/interim/text_features_dict.txt'):
    
    # Read data, drop NAs and transform to lower case
    news = pd.read_csv(filepath)
    news.dropna(inplace=True, axis=0)
    news['News'] = news['News'].apply(lambda x: x.lower())

    # Concatenate stories and group by date
    news = news.groupby(['Date'])['News'].apply(lambda x: ', '.join(x)).reset_index()

    # Lemmatize words
    lmtzr = WordNetLemmatizer()
    news['News'] = news['News'].apply(lemmatize_sentence)

    # Extract tfid Vectorizer features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    news_vectors = vectorizer.fit_transform(news['News'].values).toarray()
    text_features = pd.DataFrame(data=news_vectors, columns=vectorizer.get_feature_names())

    # Concat dates with tfid features
    text_features = pd.concat([pd.Series(news.Date.unique()).to_frame(name='Date'), text_features], axis=1)

    if create_csv == True:
        os.makedirs(csv_directory, exist_ok=True)
        text_features.to_csv(csv_path,index=False)

        with open(dict_path, 'w+') as f:
            text_features_dict = {'max_features':max_features}
            f.write(str(text_features_dict))

    if return_frame == True:
        return text_features


if __name__ == '__main__':
    
    filepath = Path('data/train/', 'RedditNews_train.csv')

    process_text_features(filepath, create_csv=True, return_frame=False)

