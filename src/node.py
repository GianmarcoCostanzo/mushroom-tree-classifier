class Node:
    def __init__(self, is_leaf=False, prediction=None, test_function=None, feature_index=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.test_function = test_function
        self.feature_index = feature_index
        self.left = left
        self.right = right

    def predict(self, x):
        if self.is_leaf:
            return self.prediction
        elif self.test_function is not None:
            if self.test_function(x):
                return self.left.predict(x)
            else:
                return self.right.predict(x)
        else:
            raise Exception("Invalid node: no test function and not a leaf.")
