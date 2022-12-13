import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from copy import deepcopy

np.random.seed(53)


## Plan
# 1. Initialize covariance matrix with parameters given in paper, i.e. Var = 5
# 2. Prove matrix is positive semidefinite - https://www.geeksforgeeks.org/how-to-create-a-matrix-of-random-integers-in-python/
# 3. Create multivariate normal from scipy - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
# 4. Use a cluster analysis to find y values
# 5. Fit logistic regression from scipy to ensure it is MLE 
# 6. Hope that it's not perfect separation and show bias
# 7. Increase feature and sample size
# 8. Show bias
# 9. ChiSquare Tests?!?!?!?

ratio = 1/5
covariance_matrix_size = 100
sample_size = covariance_matrix_size / ratio

# matrix = np.random.random_sample((covariance_matrix_size, covariance_matrix_size))*25
matrix = np.random.uniform(low=2.5, high=7, size=(covariance_matrix_size, covariance_matrix_size))

def make_symetric(mat):
    temp = deepcopy(mat)

    for i in range(covariance_matrix_size):
        for j in range(covariance_matrix_size):
            if i == j: continue
            else:
                temp[j,i] = temp[i,j]

    return temp

sym_matrix = make_symetric(matrix)

def check_symmetry(matrix):
    for i in range(covariance_matrix_size):
        for j in range(covariance_matrix_size):
            if matrix[i][j] != matrix[j][i]: return False

    return True

def confirm_positive_semidefinite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)


## Create positive definite matrix
positive_definite_matrix = np.dot(sym_matrix, sym_matrix.transpose())
# print(positive_definite_matrix)

## Confirm covariance matrix properties
# print(positive_definite_matrix)
# print(positive_definite_matrix)
# print(check_symmetry(positive_definite_matrix))
# print(np.linalg.eigvals(positive_definite_matrix))
# print(confirm_positive_semidefinite(positive_definite_matrix))

## Create data samples
sample = np.random.multivariate_normal(mean=[10]*covariance_matrix_size, cov=positive_definite_matrix, size=int(sample_size))
headers=[f'column_{number}' for number in range(covariance_matrix_size)]
# sample2 = np.random.multivariate_normal(mean=[0]*covariance_matrix_size, cov=positive_definite_matrix, size=int(sample_size))
# headers2=[f'column_{number}' for number in range(covariance_matrix_size,2*covariance_matrix_size)]
df1 = pd.DataFrame(data=sample, columns=headers)
# df2 = pd.DataFrame(data=sample2, columns=headers2)
# df_final = pd.concat([df1, df2], axis=1)
# print(df)

# ## Cluster analysis
kmeans = KMeans(n_clusters=2, random_state=0).fit(df1)
# # print(kmeans.labels_)

# ## Fit logistic regression using MLE
print(df1)
log_reg = sm.Logit(kmeans.labels_, df1).fit()
print(log_reg.summary())
