import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import regression_quality as rq

df = pd.DataFrame()

# *******************
# Step 1: Examine Data
df.info()
df.describe().T()

# *******************
# Step 1.1: Plot variable distribution
for i in df.columns:
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x=i, kde=True)
    plt.show()

# *******************
# Step 1.2: View heatmap to identify colinear variables
plt.figure(figsize=(12, 8))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap=cmap)
plt.show()

# Step 2: Adjust data
df['MEDV_log'] = np.log(df['MEDV'])  # add log-scale

# *******************
# Step 3: Prepare X and Y
Y = df['MEDV_log']  # Dependent variable (we are trying to predict)
X = df.drop(columns={'MEDV', 'MEDV_log'})  # Independent variables (regressors)

# add the intercept term (where line intercepts y-axis) beta-sub-0 in y=B0 + B1X
X = sm.add_constant(X)

# *******************
# Step 4: Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

# *******************
# Step 5: Check co-linearity of features
vif = pd.DataFrame()
vif["feature"] = X_train.columns
vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(len(X_train.columns))]

# *******************
# Step 6: Create linear regression model with statsmodels.ols
# Provides more descriptive data, useful for exploratory analysis
ols_model = sm.OLS(Y_train, X_train).fit()
ols_model.summary()

# *******************
# Step 7: Check stats on quality of regression
rq.model_perf(ols_model, X_train, X_test, Y_train, Y_test)

# *******************
# Step 8: Cross-validation on different foldings of data
linearregression = LinearRegression()

cv_Score11 = cross_val_score(linearregression, X_train, Y_train, cv=10)
cv_Score12 = cross_val_score(linearregression, X_train, Y_train, cv=10, scoring='neg_mean_squared_error')

print("RSquared: %0.3f (+/- %0.3f)" % (cv_Score11.mean(), cv_Score11.std() * 2))
print("Mean Squared Error: %0.3f (+/- %0.3f)" % (-1 * cv_Score12.mean(), cv_Score12.std() * 2))

result = linearregression.fit(X_train, Y_train)

# *******************
# Step 9: Look at model coefficients
coef = ols_model.params


def feature_importance(reg, data):
    feat_importances = pd.Series(reg.feature_importances_, index=data.columns)