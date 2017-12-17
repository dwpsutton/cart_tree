import numpy as np
from numba import jit


def verbose_print(verbosity, level, string_to_print):
    if level <= verbosity:
        print string_to_print


def gini_impurity(p0, p1):
    return 1.0 - p0**2 - p1**2


@jit
def split_attribute(datafield, indices, target):
    sum_right = float(sum(target[indices]))
    count_right = len(indices)
    sum_left = 0.
    count_left = 0
    #
    min_val = None
    min_impurity = 0.5
    for i in range(len(indices)):
        index = indices[i]
        count_left += 1
        count_right -= 1
        sum_left += target[index]
        sum_right -= target[index]
        #
        if count_left > 0:
            p0_left = 1.0 - sum_left/count_left
            p1_left = sum_left/count_left
        else:
            p0_left = 0.
            p1_left = 0.
        if count_right > 0:
            p0_right = 1.0 - sum_right/count_right
            p1_right = sum_right/count_right
        else:
            p0_right = 0.
            p1_right = 0.
        score_left = gini_impurity(p0_left, p1_left)
        score_right = gini_impurity(p0_right, p1_right)
        gini = score_left * count_left/(count_left+count_right) + score_right*count_right/(count_left+count_right)
        #
        value = datafield[index]
        #
        degenerate_value = False
        if i < len(indices)-1:
            degenerate_value = value == datafield[indices[i+1]]
        #
        if gini < min_impurity and not degenerate_value:
            min_impurity = gini
            min_val = value
    return min_val, min_impurity


def split_dataset(dataset, sorted_indices, target, verbose=0):
    best_impurity = 1.E20
    best_attribute = 0
    best_value = None
    for attribute in range(np.shape(dataset)[1]):
        value, impurity = split_attribute(dataset[:, attribute], sorted_indices[:, attribute], target)
        verbose_print(verbose, 2, str(attribute) + ' ' +
                      str(value) + ' ' + str(impurity) + ' ' +
                      str(sum(target[sorted_indices[:, attribute]]))
                      )
        if impurity < best_impurity:
            best_impurity = impurity
            best_value = value
            best_attribute = attribute
    return best_attribute, best_value


def presort_attributes(dataset):
    indices = np.zeros(np.shape(dataset), dtype=np.int32)
    for i in range(np.shape(dataset)[1]):
        indices[:, i] = np.argsort(dataset[:, i])
    return indices


def remap_split_indices(rows_to_keep, indices):
    number_to_keep = sum(rows_to_keep)
    mapper = {}
    ctr = 0
    for irow in range(len(rows_to_keep)):
        if rows_to_keep[irow]:
            mapper[irow] = ctr
            ctr += 1
    #
    remapped_indices = np.zeros([number_to_keep, np.shape(indices)[1]], dtype=np.int32)
    for jcol in range(np.shape(indices)[1]):
        ctr = 0
        for irow in range(np.shape(indices)[0]):
            old_row = indices[irow, jcol]
            if old_row in mapper:
                remapped_indices[ctr, jcol] = mapper[old_row]
                ctr += 1
    return remapped_indices


class Node:
    def __init__(self, parent, depth, w0, w1):
        self.parent = parent
        self.w0 = w0
        self.w1 = w1
        self.depth = depth
        self.attribute = None
        self.value = None
        self.child_left = None
        self.child_right = None
        self.is_leaf = None

    def make_splitter(self, attribute, value, child_left, child_right):
        self.attribute = attribute
        self.value = value
        self.child_left = child_left
        self.child_right = child_right
        self.is_leaf = False
        return None

    def make_leaf(self):
        self.child_right = None
        self.child_right = None
        self.attribute = None
        self.value = None
        self.is_leaf = True
        return None

    def score(self, x):
        if self.is_leaf:
            return float(self.w1) / (self.w0+self.w1)
        elif x[self.attribute] <= self.value:
            return self.child_left.score(x)
        else:
            return self.child_right.score(x)


class ClassificationTree:
    def __init__(self, max_depth=10, min_samples_leaf=1, verbose=0):
        self._root_node = None
        self.max_depth = max_depth
        self.min_sample_leaf = min_samples_leaf
        self.verbose = verbose

    def _pre_split_checks(self, depth, left_count, right_count):
        return (depth < self.max_depth and
                left_count >= self.min_sample_leaf and
                right_count >= self.min_sample_leaf)
    
    def _split(self, parent, X, indices, y):
        attribute, value = split_dataset(X, indices, y, verbose=self.verbose)
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
        indices = presort_attributes(X)
        self._root_node = Node(None, 0, len(y)-sum(y), sum(y))
        self._split(self._root_node, X, indices, y)
        return None

    def score(self, X):
        return self._root_node.score(X)





