import numpy as np
#from numba import jit


def gini_impurity(p0, p1):
    return 1.0 - p0**2 - p1**2

#@jit
def split_attribute(datafield,indices,target): #TODO: ADD MORE EDGE CASE TESTS
    sum_right= float(sum(target[indices]))
    count_right= len(indices)
    sum_left= 0.
    count_left= 0
    #
    min_val= None
    min_impurity= 0.5
    for i in range(len(indices)):
        index= indices[i]
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
        #
        degenerate_value= False
        if i < len(indices)-1:
            degenerate_value= value == datafield[indices[i+1]]
        #
        if gini < min_impurity and not degenerate_value:
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


## NOTHING BENEATH HERE IS TESTED



class Node():
    def __init__(self,parent,depth,w0,w1):
        self.parent= parent
        self.w0= w0
        self.w1= w1
        self.depth= depth
        self.attribute= None
        self.value= None
        self.child_left= None
        self.child_right= None
        self.is_leaf= None
        return None

    def make_splitter(self,attribute,value,child_left,child_right):
        self.attribute= attribute
        self.value= value
        self.child_left= child_left
        self.child_right= child_right
        self.is_leaf= False
        return None

    def make_leaf(self):
        self.child_right= None
        self.child_right= None
        self.attribute= None
        self.value= None
        self.is_leaf= True
        return None

    def score(self,x):
        if self.is_leaf:
            return float(self.w1) / (self.w0+self.w1)
        elif x[self.attribute] <= self.value:
            return self.child_left.score(x)
        else:
            return self.child_right.score(x)




class ClassificationTree():
    def __init__(self,max_depth=10,min_samples_leaf=1):
        self._root_node= None
        self.max_depth= max_depth
        self.min_sample_leaf= min_samples_leaf
        return None
    
    def _pre_split_checks(self,depth,left_count,right_count):
        return depth < self.max_depth and left_count >= self.min_sample_leaf and right_count >= self.min_sample_leaf
    
    def _split(self,parent,X,indices,y):
        attribute,value= split_dataset(X,indices,y)
        if self._pre_split_checks(parent.depth,sum(X[:,attribute]<=value),sum(X[:,attribute]>value)):
            #
            send= np.where(X[:,attribute] <= value)[0]
            w1= sum( y[send] )
            w0= len(send) - w1
            child_left= Node(parent,parent.depth+1,w0,w1)
            self._split(child_left,X[send,:],indices[send,:],y[send]) # FIX THIS: indices[send,:] will not point to correct elements of X[send,:] (or y[send])...
            #
            send= np.where(X[:,attribute] > value)[0]
            w1= sum( y[send] )
            w0= len(send) - w1
            child_right= Node(parent,parent.depth+1,w0,w1)
            self._split(child_right,X[send,:],indices[send,:],y[send])
            #
            parent.make_splitter(child_left,child_right)
        else:
            parent.make_leaf()
        return None

    def fit(self,X,y):
        indices= presort_attributes(X)
        self._root_node= Node(None,0,len(y)-sum(y),sum(y))
        self._split(self._root_node,X,indices,y)
        return None

    def score(self,X):
        return self._root_node.score(X)





