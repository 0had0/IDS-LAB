"""Centralized Learning Model"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from config import CentralizedModelParams
from utils import Model


def display_scores(scores):
    """Display Scores"""
    print(
        "Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(
            scores, np.mean(scores), np.std(scores)
        )
    )


def report_best_scores(results, n_top=3):
    """report the best score"""
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


PARAMS = {
    "colsample_bytree": 0.8123620356542087,
    "gamma": 0.4753571532049581,
    "learning_rate": 0.24959818254342153,
    "max_depth": 2,
    "n_estimators": 120,
    "subsample": 0.6624074561769746,
}


class CentralizedModel(Model):
    def __init__(self) -> None:
        """INIT Centrilized Model"""
        self.model = xgb.XGBClassifier(
            **PARAMS, objective="multi:softprob", random_state=75
        )

    def tune(self, x, y, params=CentralizedModelParams):
        """Tune Hyperparams"""
        self.search = RandomizedSearchCV(
            self.model,
            param_distributions=params,
            random_state=42,
            n_iter=50,
            cv=3,
            verbose=4,
            n_jobs=1,
            return_train_score=True,
        )
        self.search.fit(x, y, verbose=4)
        self.model = xgb.XGBClassifier(
            **self.search.cv_results_.params[0],
            objective="multi:softprob",
            random_state=75,
        )

    def train(self, x, y):
        """Train the Model"""
        X_train, X_val, y_train, y_val = train_test_split(
            x, y, test_size=0.1, random_state=75
        )
        # self.tune(X_val, y_val)
        self.model.fit(X_train, y_train)
