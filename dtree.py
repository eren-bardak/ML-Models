import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score


class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] < self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction
        return self.prediction


def gini(y):
    """
    Return the gini impurity score for values in y
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    counts = np.unique(y, return_counts=True)[1]
    p = counts / len(y)
    return 1 - np.sum(p**2)


def find_best_split(X, y, loss, min_samples_leaf, max_features):
    best_loss = float('inf')
    best_col, best_split = -1, -1
    n, m = X.shape

    # If max_features is a float, multiply by the number of features and convert to int
    if isinstance(max_features, float):
        max_features = int(max_features * m)

    if max_features is not None:
        cols = np.random.choice(range(m), max_features, replace=False)
    else:
        cols = range(m)

    for col in cols:
        unique_vals = np.unique(X[:, col])
        # Improved split selection: Choose splits more dynamically
        splits = np.linspace(np.min(unique_vals), np.max(
            unique_vals), num=min(11, len(unique_vals)))

        for split in splits:
            left_indices = X[:, col] < split
            right_indices = X[:, col] >= split
            if np.sum(left_indices) < min_samples_leaf or np.sum(right_indices) < min_samples_leaf:
                continue

            left_loss = loss(y[left_indices])
            right_loss = loss(y[right_indices])
            total_loss = (left_loss * np.sum(left_indices) +
                          right_loss * np.sum(right_indices)) / n

            if total_loss < best_loss:
                best_loss = total_loss
                best_col, best_split = col, split

    return best_col, best_split


class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None, max_features=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss  # loss function; either np.var for regression or gini for classification
        self.max_features = max_features

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)

    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """

        if len(X) <= self.min_samples_leaf or len(np.unique(y)) == 1:
            return self.create_leaf(y)

        col, split = find_best_split(
            X, y, self.loss, self.min_samples_leaf, self.max_features)
        if col == -1:
            return self.create_leaf(y)

        left_indices = X[:, col] < split
        right_indices = X[:, col] >= split
        lchild = self.fit_(X[left_indices], y[left_indices])
        rchild = self.fit_(X[right_indices], y[right_indices])

        return DecisionNode(col, split, lchild, rchild)

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        return np.apply_along_axis(self.root.predict, 1, X_test)

    def get_leaf(self, x_test):
        node = self.root
        while isinstance(node, DecisionNode):
            if x_test[node.col] < node.split:
                node = node.lchild
            else:
                node = node.rchild
        return node


class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features=None):
        super().__init__(min_samples_leaf, loss=np.var, max_features=None)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        predictions = self.predict(X_test)
        return r2_score(y_test, predictions)

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features=None):
        super().__init__(min_samples_leaf, loss=gini, max_features=max_features)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        mode = stats.mode(y)[0]
        return LeafNode(y, mode)
