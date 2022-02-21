from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

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

d_tree_tuned = DecisionTreeClassifier()


# Grid of hyperparameters to choose from
parameters = {"splitter":["best","random"],
              "max_depth" : [1,3,5],
              "min_samples_leaf":[1,2,3,4,5],
              "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
              "max_features":["auto","log2","sqrt",None],
              "max_leaf_nodes":[None,10,20,30,40,50,60,70]}

scorer = metrics.make_scorer(recall_score, pos_label=1)

grid_obj = GridSearchCV(d_tree_tuned, parameters).fit(X_train, y_train)

d_tree_tuned = grid_obj.best_estimator_