import argparse
from collections import namedtuple
import os
import zipfile

import pandas as pd
import numpy as np


def process_csv(filename, train_end='2014-11-20', test_start='2014-12-03'):
    df = pd.read_csv('./data/raw/'+filename+'.csv', parse_dates=True)
    df = df.set_index('Date', drop=True)
    df = df.sort_index()

    train = df.loc[:train_end, :]
    test = df.loc[test_start:, :]

    try:
        test = test.drop('label', axis=1)
    except KeyError:
        pass

    try:
        test = test.drop('Label', axis=1)
    except KeyError:
        pass

    assert train.shape[0] > test.shape[0]
    return Data(filename, train, test)


def mask_training_data(data):
    print(data.name)
    print('missing vals before {}'.format(np.sum(np.sum(pd.isnull(data.train.values)))))
    arr = data.train.values
    mask = np.random.choice([0, 1], p=[0.95, 0.05], size=arr.size).astype(bool).reshape(arr.shape)

    arr[mask] = np.nan
    new_train = pd.DataFrame(arr, index=data.train.index, columns=data.train.columns)
    data = Data(data.name, new_train, data.test)
    print('missing vals after {}'.format(np.sum(np.sum(data.train.isnull()))))
    print('')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default=0, nargs='?')
    args = parser.parse_args()

    with zipfile.ZipFile('./data/stocknews.zip', 'r') as zip_ref:
        zip_ref.extractall('./data/raw')

    Data = namedtuple('data', ['name', 'train', 'test'])

    stocks = process_csv('DJIA_table')
    stocks.train.loc[:, 'label'] = (stocks.train.loc[:, 'Adj Close'].shift(1) > stocks.train.loc[:, 'Adj Close']).astype(int)

    comb = process_csv('Combined_News_DJIA')
    news = process_csv('RedditNews')

    dataset = [stocks, comb, news]

    for d in dataset:
        print(d.name)
        print(d.train.shape)
        print(d.train.index[0], d.train.index[-1])
        print(d.test.shape)
        print(d.test.index[0], d.test.index[-1])
        print('')

    np.random.seed(42)
    dataset = [mask_training_data(d) for d in dataset]

    os.makedirs('./data/train', exist_ok=True)

    for d in dataset:
        d.train.to_csv('./data/train/{}_train.csv'.format(d.name))

        if bool(int(args.test)):
            os.makedirs('./data/test', exist_ok=True)
            d.test.to_csv('./data/test/{}_test.csv'.format(d.name))
