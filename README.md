# NLP-stock-price-prediction

This package applies natural language processing on real world news to predict whether stock markets will go up or down. 

## Data

The dataset was taken retrieved from the Kaggle competition [Daily News for Stock Market Prediction](https://www.kaggle.com/aaron7sun/stocknews/) on August 19th, 2019.

There are two sources of data:
- the r/worldnews Reddit - the top 25 headlines ranked by upvotes
- Yahoo Finance

## Model

The model predicts the daily movement of the Dow Jones Industrial Average (DJIA).  The task is formulated as a binary classification task:
- `0` when DJIA Adj Close value decreased
- `1` when DJIA Adj Close value rose or stayed as the same

The model was found to have ~95% accuracy when analyzing news from the same day. When forecasting tomorrows market movements, accuracy fall to under 60%

## Architecture

The majority of model accuracy can be attributed to NLP processing of r/worldnews. Lemmatizing was used on a feature set of 20,000 words, accuracy was seen to improve dramatically with word count.

Analysis of text sentiment and subjectivity also contributed positively. Interpolation and Principal Component Analysis were both found to be useful when cleaning and engineering the Yahoo Finance dataset.

## How to use

The model is executed as a series of bash scripts.

First, the two datasets are cleaned are processed:

```
python3 news.py
python3 number.py
```

Next, the two resultant cleaned datasets are merged:

```
python3 merge.py
```

Finally, the merged dataset is processed to train a logistic regression model:

```
python3 model.py
```
