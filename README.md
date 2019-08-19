# NLP-stock-price-prediction

This package uses natural language processing and real world news to predict whether stock markets will go up or down. 

Team members:
* [Catarina Ferreira](https://github.com/Naycat)
* [Edith Chorev](https://github.com/EdithChorev)
* [Adam Green](https://github.com/ADGEfficiency)
* [Mack Delany](https://github.com/mackdelany)

## Data

The dataset was taken from the Kaggle competition [Daily News for Stock Market Prediction](https://www.kaggle.com/aaron7sun/stocknews/).

The dataset is from two sources:
- the r/worldnews Reddit - the top 25 headlines ranked by upvotes
- Yahoo Finance
......................................



## Stock market prediction from news headlines

`Sun, J. (2016, August). Daily News for Stock Market Prediction, Version 1. Retrieved 2019-08-19 from https://www.kaggle.com/aaron7sun/stocknews`

## Target

The task is to predict the daily change in the Dow Jones Industrial Average (DJIA).  The task is formulated as a binary classification task:
- `0` when DJIA Adj Close value decreased
- `1` when DJIA Adj Close value rose or stayed as the same

## Metric

The metric is AUCROC (often called AUC) - area under the receiver operating characteristic curve.  We will use the `sklearn` implementation.  Note that `y_score` is the probability estimate of the positive class:

```python
from sklearn.metrics import roc_auc_score

metric = roc_auc_score(y_true, y_score)
```



The data is supplied as three csvs.  The raw data is in `./data/stocknews.zip`.  Run the commands below to unzip it into the correct place, and to generate the training set (you will need `pandas` and `numpy` to do this):

```bash
unzip data/stocknews.zip -d data
python data.py
```

This dataset is unclean & untrusted - be cautious!  Some data is duplicated across the three csvs.  I have randomly dropped data at a rate of 5% for all the csvs.

The data is split:
- training - 2008-06-08 to 2014-11-20
- test - 2014-12-03 to 2016-07-01

The test data is the same as the train except the two `label` columns are removed.  The test data will be generated only at test time, by running:

```python
python data.py --test 1
```

## Potential Approaches

There are a few approaches to take here - it is likely that some combination will work well.

An NLP approach - use the headlines to generate features (word counts etc):
- data cleaning (such as `.lower()`)
- a good starting point is a bag of words approach (use the `CountVectorizer` and/or `TfidfVectorizer` from `sklearn`)
- you can then move to an `n-gram` approach (looking at groups on `n` words)

Use previous values of the series + other series:
- think about how many previous values you will have at test time
- taking the log is useful to turn a multiplicative process (such as accumulating stock prices) into an additive process
- standardization/normalization
 
Use datetime features (day of week etc).
