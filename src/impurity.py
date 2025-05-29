import numpy as np

def gini_impurity(labels):
    if len(labels) == 0:
        return 0
    counts = labels.value_counts(normalize=True)
    return 1 - sum(counts ** 2)

def entropy(labels):
    if len(labels) == 0:
        return 0
    counts = labels.value_counts(normalize=True)
    return -sum(p * np.log2(p) for p in counts if p > 0)

def misclassification_error(labels):
    if len(labels) == 0:
        return 0
    counts = labels.value_counts(normalize=True)
    return 1 - counts.max()
