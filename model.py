import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('data/final_data.csv')

y = data['Label'].values
X = data.drop('Label',axis=1).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0, shuffle=False)

lr = LogisticRegression(random_state=0)
lr.fit(X, y)

y_lr_pred_train = lr.predict(X_train)
y_lr_pred_test = lr.predict(X_test)

print('Logistic Regression')
print("Training Accuracy:",metrics.accuracy_score(y_train, y_lr_pred_train))
print("Test Accuracy:",metrics.accuracy_score(y_test, y_lr_pred_test))