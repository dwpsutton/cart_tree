from cart_tree import *
import random


class RandomisedClassificationTree(ClassificationTree):
    def __init__(self, n_features, **kwargs):
        super(RandomisedClassificationTree, self).__init__(**kwargs)
        self.n_features = n_features

    def _data_parameters_consistent(self, X):
        checks = {
            "self.n_features > 0": self.n_features > 0,
            "self.n_features <= np.shape(X)[1]": self.n_features <= np.shape(X)[1]
        }
        for c in checks:
            if not checks[c]:
                raise ValueError("RandomisedClassificationTree input check failed: " + c)
        return None

    def _split(self, parent, X, indices, y):
        feature_set = set()
        while len(feature_set) < self.n_features:
            i = random.randrange(0, np.shape(X)[1], 1)
            if i not in feature_set:
                feature_set.add(i)
        chosen_features = list(feature_set)
        chosen_attribute, value = split_dataset(X[:, chosen_features], indices, y, verbose=self.verbose)
        attribute = chosen_features[chosen_attribute]
        verbose_print(self.verbose, 1,
                      '##' + str(parent.depth+1) + ' ' + str(attribute) + ' ' +
                      str(value) + ' ' + str(sum(X[:, attribute] <= value)) + ' ' +
                      str(sum(X[:, attribute] > value)) + ' ' + str(sum(y)/float(len(y)))
                      )
        if (sum(y) != 0. and sum(y) != len(y)) and \
                self._pre_split_checks(parent.depth, sum(X[:, attribute] <= value), sum(X[:, attribute] > value)):
            #
            left_rows = np.array(X[:, attribute] <= value)
            send_left = np.where(left_rows)[0]
            w1 = sum(y[send_left])
            w0 = len(send_left)-w1
            child_left = Node(parent, parent.depth+1, w0, w1)
            left_indices = remap_split_indices(left_rows, indices)
            self._split(child_left, X[send_left, :], left_indices, y[send_left])
            #
            right_rows = np.array(X[:, attribute] > value)
            send_right = np.where(right_rows)[0]
            w1 = sum(y[send_right])
            w0 = len(send_right)-w1
            child_right = Node(parent, parent.depth+1, w0, w1)
            right_indices = remap_split_indices(right_rows, indices)
            self._split(child_right, X[send_right, :], right_indices, y[send_right])
            #
            parent.make_splitter(attribute, value, child_left, child_right)
        else:
            parent.make_leaf()
        return None

    def fit(self, X, y):
        self._data_parameters_consistent(X)
        return super(RandomisedClassificationTree, self).fit(X, y)


class RandomForestClassifier(object):
    def __init__(self, n_trees=10, random_seed=None, bootstrap_fraction=1.0, **kwargs):
        self.n_trees = n_trees
        self.random_seed = random_seed
        self.bootstrap_fraction = bootstrap_fraction
        self._trees = []
        for i in range(self.n_trees):
            self._trees[i] = RandomisedClassificationTree(**kwargs)

    def _data_parameters_consistent(self, X):
        checks = {
            "self.n_trees > 0": self.n_trees > 0,
            "self.bootstrap_fraction > 0": self.bootstrap_fraction > 0,
            "self.bootstrap_fraction <= 1": self.bootstrap_fraction <= 1
        }
        for c in checks:
            if not checks[c]:
                raise ValueError("RandomForestClassifier input check failed: " + c)
        return None

    def fit(self, X, y):
        self._data_parameters_consistent(X)
        n_samples = np.shape(X)[0]
        random.seed(self.random_seed)
        for tree in self._trees:
            subsample_indices = np.random.choice(range(n_samples),
                                                 size=self.bootstrap_fraction * n_samples,
                                                 replace=True)
            tree.fit(X[subsample_indices, :], y[subsample_indices])
        return None

    def score(self, X):
        score = 0.0
        for tree in self._trees:
            score += tree.score(X)
        return score / self.n_trees
