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

    def test_binary(self):
        a= np.array([0,0,0,0,1,1,1,1])
        ind= np.argsort( a )
        target= [1,0,0,0,0,1,1,1]
        self.assertEqual( cart_tree.split_attribute(a,ind,target), (0, 0.375) )

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
                
class TestNode(unittest.TestCase):
    def test_scoring_separable(self):
        root= cart_tree.Node(None,0,5,5)
        left= cart_tree.Node(root,1,5,0)
        right= cart_tree.Node(root,1,0,5)
        root.make_splitter(0,0.5,left,right)
        left.make_leaf()
        right.make_leaf()
        self.assertEqual(root.score([0.5]),0.)
        self.assertEqual(root.score([0.1]),0.)
        self.assertEqual(root.score([0.9]),1.)
        
    def test_scoring_mix(self):
        root= cart_tree.Node(None,0,5,5)
        left1= cart_tree.Node(root,1,3,1)
        right1= cart_tree.Node(root,1,2,4)
        root.make_splitter(0,0.5,left1,right1)
        right1.make_leaf()
        left2= cart_tree.Node(left1,2,1,1)
        leftright= cart_tree.Node(left1,2,2,0)
        left1.make_splitter(0,0.25,left2,leftright)
        left2.make_leaf()
        leftright.make_leaf()
        self.assertEqual(root.score([0.5]),0.)
        self.assertEqual(root.score([0.1]),0.5)
        self.assertEqual(root.score([0.9]),2./3.)
        
    def test_scoring_2attributes(self):
        root= cart_tree.Node(None,0,5,5)
        left1= cart_tree.Node(root,1,3,1)
        right1= cart_tree.Node(root,1,2,4)
        root.make_splitter(0,0.5,left1,right1)
        right1.make_leaf()
        left2= cart_tree.Node(left1,2,1,1)
        leftright= cart_tree.Node(left1,2,2,0)
        left1.make_splitter(1,0.25,left2,leftright)
        left2.make_leaf()
        leftright.make_leaf()
        self.assertEqual(root.score([0.5,0.0]),0.5)
        self.assertEqual(root.score([0.1,1.0]),0.0)
        self.assertEqual(root.score([0.9,1.0]),2./3.)


class TestTree(unittest.TestCase):
    def test_quadrant_separable(self):
        test_data=np.array([ [ 0.03249823,  0.12690494],
                             [ 0.20112049,  0.39931997],
                             [ 0.11840662,  0.21909127],
                             [ 0.662697  ,  0.49125596],
                             [ 0.91746124,  0.83982612],
                             [ 0.32045904,  0.40729725],
                             [ 0.77466425,  0.32775041],
                             [ 0.78817532,  0.93786513],
                             [ 0.40143674,  0.95097253],
                             [ 0.61152213,  0.29923441],
                             [ 0.44111242,  0.12339367],
                             [ 0.66534288,  0.0702473 ],
                             [ 0.51026741,  0.67131538],
                             [ 0.87975577,  0.07774882],
                             [ 0.96308942,  0.16073926],
                             [ 0.30393258,  0.03719697],
                             [ 0.75398367,  0.40622869],
                             [ 0.04061277,  0.29120991],
                             [ 0.75661873,  0.62152135],
                             [ 0.99778455,  0.62935474]])
        test_target= np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0])
        tree= cart_tree.ClassificationTree(max_depth=2,min_samples_leaf=1)
        tree.fit(test_data,test_target)
        self.assertEqual(tree.score([0.25,0.25]),0.)
        self.assertEqual(tree.score([0.75,0.25]),1.)
        self.assertEqual(tree.score([0.75,0.25]),1.)
        self.assertEqual(tree.score([0.75,0.75]),0.)


if __name__=='__main__':
    unittest.main()
