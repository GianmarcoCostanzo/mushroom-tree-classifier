import numpy as np
import pandas as pd
from src.node import Node
from src.utils import gini_impurity, entropy, misclassification_error
from src.utils import get_all_best_splits_numerical, get_all_best_splits_categorical

class TreePredictor:
    def __init__(self, impurity, max_depth=11, min_samples_split=2, max_splits=100):
        self.impurity = impurity
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_splits = max_splits
        self.num_splits = 0
        self.root = None

        if impurity == "gini":
            self.impurity_fn = gini_impurity
        elif impurity == "entropy":
            self.impurity_fn = entropy
        elif impurity in ["misclassification", "error"]:
            self.impurity_fn = misclassification_error
        else:
            raise ValueError(f"Unknown impurity: {impurity}")

    def fit(self, X, y):
        self.num_splits = 0
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth, available_features=None):
        if available_features is None:
            available_features = X.columns.tolist()

        if len(y) < self.min_samples_split or depth >= self.max_depth or y.nunique() == 1:
            prediction = y.mode().iloc[0]
            return Node(is_leaf=True, prediction=prediction)

        best_feature, best_criterion, best_score, left_idx, right_idx = self._choose_split(X, y, available_features)

        if best_feature is None or len(left_idx) == 0 or len(right_idx) == 0 or y.empty:
            prediction = y.mode().iloc[0] if not y.empty else None
            return Node(is_leaf=True, prediction=prediction)

        self.num_splits += 1

        left_node = self._build_tree(X.iloc[left_idx], y.iloc[left_idx], depth + 1, available_features)
        right_node = self._build_tree(X.iloc[right_idx], y.iloc[right_idx], depth + 1, available_features)

        return Node(
            is_leaf=False,
            test_function=best_criterion,
            feature_index=best_feature,
            left=left_node,
            right=right_node
        )

    def _choose_split(self, X, y, features):
        numerical_features = [f for f in features if np.issubdtype(X[f].dtype, np.number)]
        categorical_features = [f for f in features if not np.issubdtype(X[f].dtype, np.number)]

        df_numerical = get_all_best_splits_numerical(X, y, numerical_features, self.impurity_fn)
        df_categorical = get_all_best_splits_categorical(X, y, categorical_features, self.impurity_fn)

        df_all = pd.concat([df_numerical, df_categorical], ignore_index=True)
        if df_all.empty:
            return None, None, None, None, None

        best_row = df_all.loc[df_all['impurity'].idxmin()]
        best_feature = best_row['feature']
        best_threshold = best_row.get('best_threshold', np.nan)
        best_categories = best_row.get('best_categories', np.nan)

        if not pd.isna(best_threshold):
            test_function = lambda x, t=best_threshold: x[best_feature] < t
            left_mask = X[best_feature] < best_threshold
        else:
            test_function = lambda x, cat=best_categories: x[best_feature] in cat
            left_mask = X[best_feature].isin(best_categories)

        right_mask = ~left_mask

        return (
            best_feature,
            test_function,
            best_row['impurity'],
            left_mask[left_mask].index.tolist(),
            right_mask[right_mask].index.tolist()
        )

    def predict_single(self, x):
        node = self.root
        while not node.is_leaf:
            if node.test_function(x):
                node = node.left
            else:
                node = node.right
        return node.prediction

    def evaluate(self, X, y):
        predictions = X.apply(self.predict_single, axis=1)
        return (predictions != y).mean()

    def predict(self, X):
        return np.array([self.predict_single(x) for _, x in X.iterrows()])
