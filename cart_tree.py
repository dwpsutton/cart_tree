import numpy as np
from numba import jit

def gini_index(groups):
    '''
        Calculates Gini impurity, 1 - sum_i(p^2) for classes i.
        Arguments:
            groups: list of data points in each group. Each group may have 0 or more rows.
    '''
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for target_val in [0,1]:
            p = [row[-1] for row in group].count(target_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


def gini_impurity(p0, p1):
    return 1.0 - p0**2 - p1**2

#@jit
def split_attribute(datafield,indices,target): # ADD MORE TESTS
    sum_right= float(sum(target))
    count_right= len(indices)
    sum_left= 0.
    count_left= 0
    #
    min_val= None
    min_impurity= 0.5
    for index in indices:
        count_left += 1
        count_right -= 1
        sum_left += target[index]
        sum_right -= target[index]
        #
        if count_left > 0:
            p0_left= 1.0 - sum_left/count_left
            p1_left= sum_left/count_left
        else:
            p0_left= 0.
            p1_left= 0.
        if count_right > 0:
            p0_right= 1.0 - sum_right/count_right
            p1_right= sum_right/count_right
        else:
            p0_right= 0.
            p1_right= 0.
        score_left= gini_impurity(p0_left, p1_left)
        score_right= gini_impurity(p0_right,p1_right)
        gini= score_left * count_left/(count_left+count_right) + score_right*count_right/(count_left+count_right)
        #
        value= datafield[index]
        if gini < min_impurity:
            min_impurity= gini
            min_val= value
    return min_val, min_impurity


def split_dataset(dataset,sorted_indices,target): # return attribute and value, TEST THIS
    best_impurity= 1.E20
    best_attribute= 0
    best_value= None
    for attribute in range(np.shape(dataset)[1]):
        value, impurity= split_attribute(dataset[:,attribute],sorted_indices[:,attribute],target)
        if impurity < best_impurity:
            best_impurity= impurity
            best_value= value
            best_attribute= attribute
    return best_attribute, best_value




'''
Pre sort each attribute, so you have original data and the sorted indices
Linear searching algo for each attribute: start at lowest index with each element in left group, then increase index and move items to right group (evaluating impurity)
Recursive splitting algorithm: takes as arguments the dataset, the targets, and the sorted indices for each attribute.  Must exit if leaf is pure or of minimum size
Build the decision tree from node class (which knows it's left and right children, parent, splitting attribute, splitting value
'''


