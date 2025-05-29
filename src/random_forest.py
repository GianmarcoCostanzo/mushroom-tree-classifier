import numpy as np
from collections import Counter
from sklearn.utils import resample
from src.tree_predictor import TreePredictor

class RandomForestPredictor:
    def __init__(self, n_estimators=10, impurity="gini", max_depth=5, min_samples_split=10, bootstrap=True, random_state=42):
        self.n_estimators = n_estimators
        self.impurity = impurity
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.trees = []
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []

        for i in range(self.n_estimators):
            if self.bootstrap:
                X_sample, y_sample = resample(X, y, random_state=self.random_state + i)
            else:
                X_sample, y_sample = X, y

            tree = TreePredictor(
                impurity=self.impurity,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            print(f"🌲 Trained tree {i + 1}/{self.n_estimators}")

    def predict(self, X):
        all_preds = []
        for tree in self.trees:
            preds = tree.predict(X)
            all_preds.append(preds)

        all_preds = np.array(all_preds).T

        majority_votes = []
        for row in all_preds:
            vote = Counter(row).most_common(1)[0][0]
            majority_votes.append(vote)

        return np.array(majority_votes)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        return np.mean(y_pred != y_true)
