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
for seed, item in enumerate(range(1000)):
    np.random.seed(seed)
    sample = np.random.normal(loc=0, scale=25, size=200)
    if seed == 0:
        d = {}
        for var in range(200):
            d[f'variable_{var}'] = [sample[var]]
        # print(d)
        df = pd.DataFrame(d)
    else:
        d = {}
        for var in range(200):
            d[f'variable_{var}'] = [sample[var]]
        # print(d)
        new_df = pd.DataFrame(d)
        df = pd.concat([df, new_df]).reset_index(drop=True)

    

print(df)
km = KMeans(n_clusters=2, random_state=0).fit(df)
print(km.labels_)

Y = km.labels_
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.25)
lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
res = lr.predict(X_test)
print(cross_val_score(lr, X_test,  y_test, cv=3))

predicted_probs = lr.predict_proba(X_train)
print(predicted_probs[:3])
minimum, maximum = min(map(lambda x: x[0], predicted_probs)), max(map(lambda x: x[0], predicted_probs))
print('min/max', minimum, maximum)
X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
V = np.diagflat(np.product(predicted_probs, axis=1))
covLogit = np.linalg.inv(np.dot(np.dot(X_design.T, V), X_design))
print("Covariance matrix: ", covLogit)
print(np.diag(covLogit))