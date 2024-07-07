import numpy as np
from collections import Counter


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

# Function to calculate information gain
def information_gain(X, y, feature):
    parent_entropy = entropy(y)
    values, counts = np.unique(X[:, feature], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(y[X[:, feature] == v]) for i, v in enumerate(values)])
    return parent_entropy - weighted_entropy

def split_dataset(X, y, feature, value):
    left_idx = np.where(X[:, feature] == value)[0]
    right_idx = np.where(X[:, feature] != value)[0]
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Decision Tree Classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # stopping criteria
        if depth >= self.max_depth or num_labels == 1 or num_samples == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # find the best split
        best_feature, best_value = self._best_split(X, y, num_features)
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # grow the children that result from the split
        left_X, left_y, right_X, right_y = split_dataset(X, y, best_feature, best_value)
        left_child = self._grow_tree(left_X, left_y, depth + 1)
        right_child = self._grow_tree(right_X, right_y, depth + 1)
        return Node(feature=best_feature, threshold=best_value, left=left_child, right=right_child)

    def _best_split(self, X, y, num_features):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feature in range(num_features):
            gain = information_gain(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                split_idx = feature
        if split_idx is not None:
            split_threshold = Counter(X[:, split_idx]).most_common(1)[0][0]
        return split_idx, split_threshold

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] == node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
