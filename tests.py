import cart_tree
import numpy as np
import unittest

class TestAttributeSplitter(unittest.TestCase):
    def test_separable(self):
         a= np.array([ 0.67803705,  0.20739757,  0.31986182,  0.17318886,  0.33332778,
                      0.70697814,  0.15630182,  0.792228  ,  0.76916237,  0.8140787 ])
         ind= np.argsort( a )
         target= np.array( map(lambda x: int(x > 0.5),a) )
         self.assertEqual( cart_tree.split_attribute(a,ind,target), (0.33332778000000002, 0.0) )

    def test_single(self):
        a= np.array([0.67803705])
        ind= np.argsort( a )
        target= np.array( map(lambda x: int(x > 0.5),a) )
        self.assertEqual( cart_tree.split_attribute(a,ind,target), (0.67803705, 0.0) )


class TestDatasetSplitter(unittest.TestCase):
    def test_separable(self):
        X= np.zeros([10,2])
        X[:,0]= [2.77,1.72,3.67,3.96,2.999,7.498,9.00,7.445,10.12,6.64]
        X[:,1]= [1.78,1.17,2.81,2.62,2.21,3.16,3.34,0.48,3.23,3.32]
        target= np.array([0,0,0,0,0,1,1,1,1,1])
        indices= np.zeros(np.shape(X),dtype=np.int32)
        indices[:,0]= np.argsort(X[:,0])
        indices[:,1]= np.argsort(X[:,1])
        self.assertEqual( cart_tree.split_dataset(X,indices,target), (0,3.96) )
        
class TestDatasetSorter(unittest.TestCase):
    def test_unsorted(self):
        X= np.zeros([5,3])
        X[:,0]= [5,4,3,2,1]
        X[:,1]= [1,2,1,2,1]
        X[:,2]= [1,2,3,4,5]
        result= np.zeros([5,3],dtype=np.int32)
        result[:,0]= np.array([4,3,2,1,0],dtype= np.int32)
        result[:,1]= np.array([0, 2, 4, 1, 3],dtype= np.int32)
        result[:,2]= np.array([0,1,2,3,4],dtype= np.int32)
        indices= cart_tree.presort_attributes(X)
        for j in range(3):
            for i in range(5):
                self.assertEqual( indices[i,j], result[i,j] )
        
if __name__=='__main__':
    unittest.main()
