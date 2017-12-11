import numpy as np
#from numba import jit


def gini_impurity(p0, p1):
    return 1.0 - p0**2 - p1**2

#@jit
def split_attribute(datafield,indices,target): #TODO: ADD MORE EDGE CASE TESTS
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


def split_dataset(dataset,sorted_indices,target):
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


def presort_attributes(dataset):
    indices= np.zeros(np.shape(dataset),dtype=np.int32)
    for i in range(np.shape(dataset)[1]):
        indices[:,i]= np.argsort(dataset[:,i])
    return indices


'''
Pre sort each attribute, so you have original data and the sorted indices
Linear searching algo for each attribute: start at lowest index with each element in left group, then increase index and move items to right group (evaluating impurity)
Recursive splitting algorithm: takes as arguments the dataset, the targets, and the sorted indices for each attribute.  Must exit if leaf is pure or of minimum size
Build the decision tree from node class (which knows it's left and right children, parent, splitting attribute, splitting value
'''


