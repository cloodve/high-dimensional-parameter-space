import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
import numpy as np
from copy import deepcopy

## Stitch together our own sample
# 200/1000 = 1/5, 20/1000 = 1/5
# Generate 200 variables - set seed for each
df = None
for seed, item in enumerate(range(200)):
    np.random.seed(seed)
    if seed < 50:
        loc = 10
    elif seed >= 50 and seed < 100:
        loc = -10
    else:
        loc = 0

    sample = np.random.normal(loc=loc, scale=25, size=1000)

    if seed == 0:
        df = pd.DataFrame({f'variable_{seed}':sample})
    else:
        df[f'variable_{seed}'] = sample

print(df)
km = KMeans(n_clusters=2, random_state=0).fit(df)
print(km.labels_)

Y = km.labels_
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.25)
lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
res = lr.predict(X_test)
print(cross_val_score(lr, X_test,  y_test, cv=3))

predicted_probs = lr.predict_proba(X_train)
X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
V = np.diagflat(np.product(predicted_probs, axis=1))
covLogit = np.linalg.inv(np.dot(np.dot(X_design.T, V), X_design))
print("Covariance matrix: ", covLogit)
print(np.diag(covLogit))