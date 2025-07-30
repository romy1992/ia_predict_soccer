from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Random Forest
param_random = {
    # 'classifier': [RandomForestClassifier()],
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced'],
    'max_features': [None, 'sqrt', 'log2']
}

# Support Vector Machine (SVM)
param_svc = {
    # 'classifier': [SVC(probability=True)],
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'class_weight': [None, 'balanced'],
    'shrinking': [True, False]
}

# Logistic Regression
param_lg = {
    # 'classifier': [LogisticRegression()],
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2', 'l1', 'elasticnet', 'none'],
    'solver': ['lbfgs', 'saga', 'liblinear'],
    'max_iter': [100, 200, 500],
    'class_weight': [None, 'balanced']
}

# Decision Tree
param_decision = {
    # 'classifier': [DecisionTreeClassifier()],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

# K-Nearest Neighbors
param_nei = {
    # 'classifier': [KNeighborsClassifier()],
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean', 'manhattan'],
    'p': [1, 2]
}

# Gradient Boosting (LightGBM)
param_gb = {
    # 'classifier': [LGBMClassifier()],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [-1, 5, 10],
    'num_leaves': [31, 50, 100],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'scale_pos_weight': [1, 2, 5, 10]  # Per bilanciare le classi
}

params_classifier_ensemble = [
    # AdaBoost:
    #
    # {
    #     'classifier': [AdaBoostClassifier()],
    #     'classifier__n_estimators': [50, 100, 200, 500],
    #     'classifier__learning_rate': [0.01, 0.1, 0.5, 1.0],
    #     'classifier__algorithm': ['SAMME.R', 'SAMME'],
    #     'classifier__random_state': [42],  # Per riproducibilità
    #     'classifier__estimator': [
    #         RandomForestClassifier(max_depth=3, class_weight='balanced', max_leaf_nodes=16),
    #         DecisionTreeClassifier(max_depth=5, class_weight='balanced')
    #     ]
    # },
    # Gradient Boosting (XGBoost)
    # {
    #     'classifier': [XGBClassifier()],
    #     'classifier__n_estimators': [50, 100, 200],
    #     'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
    #     'classifier__max_depth': [3, 5, 10],
    #     'classifier__subsample': [0.5, 0.7, 1.0],
    #     'classifier__colsample_bytree': [0.5, 0.7, 1.0],
    #     'classifier__scale_pos_weight': [1, 2, 5, 10]  # Per gestire classi sbilanciate
    # },
    # # GradientBoosting
    # {
    #     'classifier': [GradientBoostingClassifier()],
    #     'classifier__n_estimators': [50, 100, 200, 500],
    #     'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
    #     'classifier__max_depth': [3, 5, 10],
    #     'classifier__min_samples_split': [2, 5, 10],
    #     'classifier__min_samples_leaf': [1, 2, 4],
    #     'classifier__subsample': [0.5, 0.7, 1.0],
    #     'classifier__max_features': ['sqrt', 'log2', None],
    #     'classifier__loss': ['log_loss', 'exponential']
    # },
]


def create_gs(name):
    scorer = make_scorer(f1_score, average='weighted', zero_division=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    match name:
        case 'random':
            estimator = RandomForestClassifier()
            params = param_random
        case 'svc':
            estimator = SVC()
            params = param_svc
        case 'lg':
            estimator = LogisticRegression()
            params = param_lg
        case 'decision':
            estimator = DecisionTreeClassifier()
            params = param_decision
        case 'nei':
            estimator = KNeighborsClassifier()
            params = param_nei
        case 'gb':
            estimator = LGBMClassifier()
            params = param_gb
        case _:
            raise Exception

    return GridSearchCV(scoring=scorer, estimator=estimator, param_grid=params,
                        verbose=2, n_jobs=-1,
                        cv=cv, return_train_score=True)
