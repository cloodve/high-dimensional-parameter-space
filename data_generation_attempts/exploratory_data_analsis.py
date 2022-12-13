import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
import numpy as np
from copy import deepcopy

pd.set_option("display.max_columns" , None)

# Grab data
kaggle_car_data = "C:\\Users\\ericd\\Desktop\\JohnsHopkins\\MATH625.725\\Project\\DataSets\\Cars\\archive\\car_data.csv"
fifa_data = "C:\\Users\\ericd\\Desktop\\JohnsHopkins\\MATH625.725\\Project\\DataSets\\Sports\\fifa18_clean.csv"
hosing_data_path = "C:\\Users\\ericd\\Desktop\\JohnsHopkins\\MATH625.725\\Project\\DataSets\\Housing\\home_data.txt"
store_formatted_data = ""
data = pd.read_csv(kaggle_car_data)

# Rename columns
data = data.rename(columns={'Unnamed: 0': 'index'})

def numerify_column(data, col):
    temp = deepcopy(data)
    temp = temp[~temp[col].str.contains('\?')].reset_index(drop=True)
    temp[col] = pd.to_numeric(temp[col])
    return temp


for column in ['price', 'peak_rpm', 'horsepower']:
    data = numerify_column(data, column)


# print(data.columns)
# print(data.dtypes)
# print(data.head())
X , y = data.loc[:,~data.columns.isin(['index', 'fuel_type'])], data.fuel_type


def convert_categorical_features(df):
    temp = deepcopy(df)
    dtypes = temp.dtypes.to_dict()
    for column, type in dtypes.items():
        if type == 'object':
            ret = pd.get_dummies(temp[column], prefix=column)
            temp = temp.drop(column, axis=1)
            temp = pd.concat([temp, ret], axis=1)
    return temp


X = convert_categorical_features(X)



## Now ready to build a model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
# res = lr.predict(X_test)
# print(cross_val_score(lr, X_test,  y_test, cv=3))

## Build statsmodels regression.
# Convert Y into dummy
new_y = np.where(y_train == 'gas', 1, 0)
log_reg = sm.Logit(new_y, X_train).fit()
print(log_reg.summary())



# Compute variance
# predicted_probs = lr.predict_proba(X_train)
# X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
# V = np.diagflat(np.product(predicted_probs, axis=1))
# covLogit = np.linalg.inv(np.dot(np.dot(X_design.T, V), X_design))
# print("Covariance matrix: ", covLogit)

# beta_norm_squared = beta.T.dot(beta).values[0,0]
# beta_scaled = beta * np.sqrt((n * signal_strength) / beta_norm_squared)






