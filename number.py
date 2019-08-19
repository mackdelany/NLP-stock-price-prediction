from pathlib import Path

import pandas as  pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

def fix_empty(df1, df2):
    f1_index = df1.index[df1.apply(np.isfinite)].to_list()
    f2_index = df2.index[df2.apply(np.isfinite)].to_list()

    idx=list(set(f1_index) & set(f2_index))
    # since linearly correlated I can fill values easy using linear regression
    slope, intercept,*z= stats.linregress(df1[idx], df2[idx])
    f2_index = df2.index[df2.apply(np.isnan)].to_list()

    df2[f2_index] = intercept + slope * df1[f2_index]


# load data
data_folder = Path('data/train')
text_file_path = data_folder / 'DJIA_table_train.csv'
DJ_df = pd.read_csv(text_file_path, parse_dates=True)

filename = "DJIA_table_train.csv"
mydir = mydir = "/home/edith/Documents/DSR/mincomp2/minicomp-news-stock-prices/data/NLP-stock-price-prediction/data/train/"
DJ_df = pd.read_csv(text_file_path, parse_dates=True)
mydir = './data/interim/'

# clean correlated values: open,high,close,lose and Adj_close
cols = ["Open","High","Low","Close","Adj Close"]

for col in range( len(cols) ):
    for col2 in range (col+1 , len(cols)):
        fix_empty(DJ_df[ cols[col] ],DJ_df[ cols[col2] ])
        fix_empty(DJ_df[ cols[col2] ],DJ_df[ cols[col] ])

# creat lag_vol column and replace nan values by taking larger vol shifts
shift=1     
DJ_df["Lag_Vol"] = DJ_df.Volume.shift(shift)
DJ_df.Lag_Vol.loc[0] = DJ_df.Lag_Vol.loc[1]
train_idx2 = DJ_df.index[DJ_df.Lag_Vol.apply(np.isnan)].to_list()

def lag_replace(shift, data, idx):
    d = data.Volume.shift(shift)
    data.loc[idx,"Lag_Vol"] = d[idx]
    return data.index[DJ_df.Lag_Vol.apply(np.isnan)]

while shift<5:
    shift += 1
    train_idx2 = lag_replace(shift, DJ_df, train_idx2)
    if len(train_idx2) == 0: break

if len(train_idx2) > 0:
    if train_idx2 == 0:
        DJ_df.Lag_Vol[train_idx2] = DJ_df.Lag_Vol[1]
    else:
        DJ_df.Lag_Vol[train_idx2] = DJ_df.Lag_Vol[train_idx2-1]

# prepare for random forest to replace nans in Volume column
train_idx = DJ_df.index[DJ_df.Volume.apply(np.isfinite)].to_list()

test_idx = DJ_df.index[DJ_df.Volume.apply(np.isnan)].to_list()

X_train = DJ_df.loc[train_idx,:]
y_train = X_train.Volume

X_train.drop(columns='Volume', inplace=True) 
X_train.drop(columns=["Date", "label"], inplace=True)

X_val = DJ_df.loc[test_idx,:]
X_val.drop(columns=["Date","label"],inplace=True)
y_val = X_val.Volume
X_val.drop(columns = 'Volume', inplace=True) 

# random forest 
regr = RandomForestRegressor(max_depth=3, random_state=42, n_estimators=100, verbose=1)
regr.fit(X_train, y_train)  
# predict nan values and replace 
yhat = regr.predict(X_val)
DJ_df.loc[test_idx, "Volume"] = yhat

# make labels
a = DJ_df["Adj Close"] - DJ_df["Adj Close"].shift(1)
a[a>=0] = 1
a[a<0] = 0
a[0] == 0
DJ_df.label = a

# PCA to reduce dimentionality 
pca = PCA(n_components=2)
pca.fit(DJ_df.loc[:,["Open","High","Low","Close","Volume","Adj Close","Lag_Vol"]])
b = pca.transform(DJ_df.loc[:, ["Open","High","Low","Close","Volume","Adj Close","Lag_Vol"]])
DJ_df["1st_PC"] = b[:,0]
DJ_df["2nd_PC"] = b[:,1]

DJ_df.drop('label',axis=1).to_csv(mydir+"clean_DJIA.csv", index=False)