import cart_tree
import numpy as np
import unittest

class TestSplitting(unittest.TestCase):
    def testsplit_balanced(self):
        groups= [
                 [[1, 1], [1, 0]],
                 [[1, 1], [1, 0]]
                 ]
        self.assertEqual(cart_tree.gini_index(groups),0.5)

    def testsplit_pure(self):
        groups= [
                 [[1, 0], [1, 0]],
                 [[1, 1], [1, 1]]
                 ]
        self.assertEqual(cart_tree.gini_index(groups),0.0)

class TestAttributeSplitter(unittest.TestCase):
    def test_separable(self):
         a= np.array([ 0.67803705,  0.20739757,  0.31986182,  0.17318886,  0.33332778,
                      0.70697814,  0.15630182,  0.792228  ,  0.76916237,  0.8140787 ])
         ind= np.argsort( a )
         target= np.array( map(lambda x: int(x > 0.5),a) )
         self.assertEqual( cart_tree.split_attribute(a,ind,target), (0.33332778000000002, 0.0) )


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

if __name__=='__main__':
    unittest.main()
