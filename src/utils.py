from sklearn.metrics import confusion_matrix


class Model:
    """Architecture Model Template"""

    def train(self, x, y):
        """training phase, overwrite to add custom logic"""
        self.model.fit(x, y)

    def predict(self, x):
        """predict"""
        return self.model.predict(x)

    def evaluate(self, y_true, y_pred):
        """Evalute output"""
        return confusion_matrix(y_true, y_pred)
