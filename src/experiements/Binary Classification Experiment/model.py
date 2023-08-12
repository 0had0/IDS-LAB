"""Model to classifier between normal and intrustion trafic without identification"""
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils import Model

BinaryClassifierParams = {
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


class BinaryClassifier(Model):
    """Model to classifier between normal and intrustion trafic without identification"""

    def __init__(self) -> None:
        self.model = DecisionTreeClassifier(random_state=75)

    def tune(self, x, y):
        """Tune Hyperparams"""
        self.search = RandomizedSearchCV(
            self.model,
            param_distributions=BinaryClassifierParams,
            random_state=75,
            n_iter=50,
            cv=3,
            verbose=4,
            n_jobs=1,
            return_train_score=True,
        )
        self.search.fit(x, y, verbose=4)
        self.model = DecisionTreeClassifier(
            **self.search.best_params_,
            random_state=75,
        )

    def train(self, x, y, tune_model=True):
        """Train the Model"""
        if tune_model:
            X_train, X_val, y_train, y_val = train_test_split(
                x, y, test_size=0.1, stratify=y, random_state=75
            )
            self.tune(X_val, y_val)
        else:
            X_train, y_train = x, y
        self.model.fit(X_train, y_train, verbose=4)
