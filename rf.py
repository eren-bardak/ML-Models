import numpy as np
from sklearn.utils import resample
from dtree import *


class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.trees = []
        self.oob_indices = []

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            bootstrap_sample, oob_indices = self._bootstrap_sample(
                X, n_samples)
            tree = self.create_tree()  # Create specific tree type
            tree.fit(X[bootstrap_sample], y[bootstrap_sample])
            self.trees.append(tree)
            self.oob_indices.append(oob_indices)

        if self.oob_score:
            self.oob_score_ = self._compute_oob_score(X, y)

    def create_tree(self):
        # Method to be implemented in subclasses
        raise NotImplementedError

    def _bootstrap_sample(self, X, n_samples):
        bootstrap_indices = resample(
            np.arange(n_samples), n_samples=n_samples, replace=True)
        oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
        return bootstrap_indices, oob_indices

    def _compute_oob_score(self, X, y):

        oob_predictions = np.full((len(X), self.n_estimators), np.nan)
        oob_counts = np.zeros(len(X))

        for i, (tree, oob_indices) in enumerate(zip(self.trees, self.oob_indices)):
            if len(oob_indices) > 0:
                tree_predictions = tree.predict(X[oob_indices])
                oob_predictions[oob_indices, i] = tree_predictions
                oob_counts[oob_indices] += 1

        if self.score_metric == r2_score:
            mean_oob_predictions = np.nanmean(oob_predictions, axis=1)
            valid_predictions = ~np.isnan(mean_oob_predictions)
            final_predictions = mean_oob_predictions[valid_predictions]
        elif self.score_metric == accuracy_score:
            mode_oob_predictions = stats.mode(
                oob_predictions, axis=1, nan_policy='omit')[0].flatten()
            valid_predictions = ~np.isnan(mode_oob_predictions)
            final_predictions = mode_oob_predictions[valid_predictions]
        else:
            raise NotImplementedError("Unsupported score metric")

        # Calculate and return the OOB score
        return self.score_metric(y[valid_predictions], final_predictions)


class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3,
                 max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.score_metric = r2_score

    def create_tree(self):
        return RegressionTree621(self.min_samples_leaf, max_features=self.max_features)

    def predict(self, X_test):
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        predictions = [tree.predict(X_test) for tree in self.trees]
        return np.mean(predictions, axis=0)

    def score(self, X_test, y_test):
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        return r2_score(y_test, self.predict(X_test))

    def _compute_oob_score(self, X, y):
        oob_predictions = np.full((len(X), self.n_estimators), np.nan)
        oob_counts = np.zeros(len(X))

        # Accumulate predictions from each tree for their OOB samples
        for i, (tree, oob_indices) in enumerate(zip(self.trees, self.oob_indices)):
            if len(oob_indices) > 0:
                tree_predictions = tree.predict(X[oob_indices])
                oob_predictions[oob_indices, i] = tree_predictions
                oob_counts[oob_indices] += 1

        if self.score_metric == r2_score:
            mean_oob_predictions = np.nanmean(oob_predictions, axis=1)
            valid_predictions = ~np.isnan(mean_oob_predictions)
            final_predictions = mean_oob_predictions[valid_predictions]
        elif self.score_metric == accuracy_score:
            mode_oob_predictions = stats.mode(
                oob_predictions, axis=1, nan_policy='omit')[0].flatten()
            valid_predictions = ~np.isnan(mode_oob_predictions)
            final_predictions = mode_oob_predictions[valid_predictions]
        else:
            raise NotImplementedError("Unsupported score metric")

        return self.score_metric(y[valid_predictions], final_predictions)


class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3,
                 max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.score_metric = accuracy_score

    def create_tree(self):
        return ClassifierTree621(self.min_samples_leaf, max_features=self.max_features)

    def predict(self, X_test):
        predictions = [tree.predict(X_test) for tree in self.trees]
        mode_predictions = stats.mode(predictions, axis=0)[0]
        return mode_predictions.ravel()

    def score(self, X_test, y_test):
        return accuracy_score(y_test, self.predict(X_test))

    def _compute_oob_score(self, X, y):
        # Create an array to hold predictions for each tree
        oob_predictions = []

        for tree, oob_indices in zip(self.trees, self.oob_indices):
            # Initialize a full nan array for each tree
            tree_oob_predictions = np.full(len(X), np.nan)
            # Predict only for OOB samples
            if oob_indices.size > 0:
                tree_oob_predictions[oob_indices] = tree.predict(
                    X[oob_indices])
            # Add the predictions for this tree to the list
            oob_predictions.append(tree_oob_predictions)

        # Convert the list of arrays into a 2D array
        oob_predictions = np.vstack(oob_predictions).T

        # For classification, compute the mode (most common prediction) across trees
        mode_oob_predictions, _ = stats.mode(
            oob_predictions, axis=1, nan_policy='omit')

        # Flatten the array and create a mask for valid predictions
        mode_oob_predictions = mode_oob_predictions.ravel()
        valid_mask = ~np.isnan(mode_oob_predictions)

        # Compute the OOB score using the valid predictions only
        return self.score_metric(y[valid_mask], mode_oob_predictions[valid_mask])
