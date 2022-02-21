import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn import metrics

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    plot_confusion_matrix,
    precision_recall_curve,
    roc_curve,
    make_scorer,
)

import regression_quality as rq

df = pd.DataFrame()

# *******************
# Single Tree
Y1 = df['MEDV_log']
X1 = df.drop(columns={'MEDV', 'MEDV_log'})
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.30, random_state=1)
dt = DecisionTreeRegressor(min_samples_split=2)
dt.fit(X_train1, y_train1)
rq.model_perf(dt, X_train1, X_test1, y_train1, y_test1)

# Tree visualization
features = list(X1.columns)
plt.figure(figsize=(35, 25))
plot_tree(dt, max_depth=4, feature_names=features, filled=True, fontsize=12, node_ids=True, class_names=True)
plt.show()

# Visualize feature importance
importances = dt.feature_importances_
columns = X1.columns
importance_df = pd.DataFrame(importances, index=columns, columns=['Importance']).sort_values(by='Importance',
                                                                                             ascending=False)
plt.figure(figsize=(8, 4))
sns.barplot(importance_df.Importance, importance_df.index)
plt.show()

# *******************
# Random Forrest
rf = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_split=2)
rf.fit(X_train1, y_train1)
rf.score(X_test1, y_test1)
rq.model_perf(rf, X_train1, X_test1, y_train1, y_test1)


def hyper_parameter_tuning():
    parameters = {"splitter":["best","random"],
                  "max_depth" : [1,3,5],
                  "min_samples_leaf":[1,2,3,4,5],
                  "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                  "max_features":["auto","log2","sqrt",None],
                  "max_leaf_nodes":[None,10,20,30,40,50,60,70]}

    # Type of scoring used to compare parameter combinations - recall score for class 1
    scorer = metrics.make_scorer(recall_score, pos_label=1)

    # Run the grid search
    grid_obj = GridSearchCV(dt, parameters, scoring=scorer).fit(X_train, y_train)