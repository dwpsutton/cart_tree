import cart_tree
import numpy as np
import unittest


class TestAttributeSplitter(unittest.TestCase):
    def test_separable(self):
        a = np.array([0.67803705,  0.20739757,  0.31986182,  0.17318886,  0.33332778,
                      0.70697814,  0.15630182,  0.792228,  0.76916237,  0.8140787])
        ind = np.argsort(a)
        target = np.array(map(lambda x: int(x > 0.5), a))
        self.assertEqual(cart_tree.split_attribute(a, ind, target), (0.33332778000000002, 0.0))

    def test_single(self):
        a = np.array([0.67803705])
        ind = np.argsort(a)
        target = np.array(map(lambda x: int(x > 0.5), a))
        self.assertEqual(cart_tree.split_attribute(a, ind, target), (0.67803705, 0.0))

    def test_binary(self):
        a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        ind = np.argsort(a)
        target = np.array([1, 0, 0, 0, 0, 1, 1, 1])
        self.assertEqual(cart_tree.split_attribute(a, ind, target), (0, 0.375))

    def test_subset_of_indices(self):
        a = np.array([0.67803705, 0.20739757, 0.31986182, 0.17318886, 0.33332778,
                      0.70697814, 0.48, 0.792228, 0.76916237, 0.8140787])
        ind = np.array(filter(lambda x: x <= 4, np.argsort(a)))
        target = np.array(map(lambda x: int(x > 0.5), a))
        self.assertEqual(cart_tree.split_attribute(a, ind, target), (0.33332778, 0.0))


class TestDatasetSplitter(unittest.TestCase):
    def test_separable(self):
        X = np.zeros([10, 2])
        X[:, 0] = [2.77, 1.72, 3.67, 3.96, 2.999, 7.498, 9.00, 7.445, 10.12, 6.64]
        X[:, 1] = [1.78, 1.17, 2.81, 2.62, 2.21, 3.16, 3.34, 0.48, 3.23, 3.32]
        target = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        indices = np.zeros(np.shape(X), dtype=np.int32)
        indices[:, 0] = np.argsort(X[:, 0])
        indices[:, 1] = np.argsort(X[:, 1])
        self.assertEqual(cart_tree.split_dataset(X, indices, target), (0, 3.96))


class TestDatasetSorter(unittest.TestCase):
    def test_unsorted(self):
        X = np.zeros([5, 3])
        X[:, 0] = [5, 4, 3, 2, 1]
        X[:, 1] = [1, 2, 1, 2, 1]
        X[:, 2] = [1, 2, 3, 4, 5]
        result = np.zeros([5, 3], dtype=np.int32)
        result[:, 0] = np.array([4, 3, 2, 1, 0], dtype=np.int32)
        result[:, 1] = np.array([0, 2, 4, 1, 3], dtype=np.int32)
        result[:, 2] = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        indices = cart_tree.presort_attributes(X)
        for j in range(3):
            for i in range(5):
                self.assertEqual(indices[i, j], result[i, j])


class TestNode(unittest.TestCase):
    def test_scoring_separable(self):
        root = cart_tree.Node(None, 0, 5, 5)
        left = cart_tree.Node(root, 1, 5, 0)
        right = cart_tree.Node(root, 1, 0, 5)
        root.make_splitter(0, 0.5, left, right)
        left.make_leaf()
        right.make_leaf()
        self.assertEqual(root.score([0.5]), 0.)
        self.assertEqual(root.score([0.1]), 0.)
        self.assertEqual(root.score([0.9]), 1.)
        
    def test_scoring_mix(self):
        root = cart_tree.Node(None, 0, 5, 5)
        left1 = cart_tree.Node(root, 1, 3, 1)
        right1 = cart_tree.Node(root, 1, 2, 4)
        root.make_splitter(0, 0.5, left1, right1)
        right1.make_leaf()
        left2 = cart_tree.Node(left1, 2, 1, 1)
        leftright = cart_tree.Node(left1, 2, 2, 0)
        left1.make_splitter(0, 0.25, left2, leftright)
        left2.make_leaf()
        leftright.make_leaf()
        self.assertEqual(root.score([0.5]), 0.)
        self.assertEqual(root.score([0.1]), 0.5)
        self.assertEqual(root.score([0.9]), 2./3.)
        
    def test_scoring_2attributes(self):
        root = cart_tree.Node(None, 0, 5, 5)
        left1 = cart_tree.Node(root, 1, 3, 1)
        right1 = cart_tree.Node(root, 1, 2, 4)
        root.make_splitter(0, 0.5, left1, right1)
        right1.make_leaf()
        left2 = cart_tree.Node(left1, 2, 1, 1)
        leftright = cart_tree.Node(left1, 2, 2, 0)
        left1.make_splitter(1, 0.25, left2, leftright)
        left2.make_leaf()
        leftright.make_leaf()
        self.assertEqual(root.score([0.5, 0.0]), 0.5)
        self.assertEqual(root.score([0.1, 1.0]), 0.0)
        self.assertEqual(root.score([0.9, 1.0]), 2./3.)


class TestTree(unittest.TestCase):
    def test_no_data(self):
        tree = cart_tree.ClassificationTree()
        test_data = []
        test_target = []
        self.assertRaises(ValueError, tree.fit, test_data, test_target)

    def test_half_separable(self):
        test_data = np.array([[0.01564607,  0.47523396],
                             [0.85026578,  0.06799358],
                             [0.6094325,  0.19395484],
                             [0.0447059,  0.04507807],
                             [0.4125889,  0.98830637],
                             [0.19468337,  0.76298617],
                             [0.41811272,  0.05896198],
                             [0.77753677,  0.08361111],
                             [0.61621537,  0.7917574],
                             [0.52788686,  0.49538718],
                             [0.9558875,  0.94164169],
                             [0.30943519,  0.37436321],
                             [0.70378457,  0.16307878],
                             [0.66324182,  0.06721279],
                             [0.27245177,  0.25113635],
                             [0.146724,  0.0051949],
                             [0.46219164,  0.16581658]])
        test_target = np.array([1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
        tree = cart_tree.ClassificationTree(max_depth=10, min_samples_leaf=1)
        tree.fit(test_data, test_target)
        self.assertEqual(tree.score([0.25, 0.]), 1.)
        self.assertEqual(tree.score([0.75, 0.]), 0.)

    def test_half_separable2(self):
        test_target = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1])
        test_data = np.array([[0.99977421,  0.79289368],
                              [0.18882997,  0.55149463],
                              [0.70236439,  0.77434885],
                              [0.61500026,  0.0122244],
                              [0.35116481,  0.53293611],
                              [0.45900421,  0.87131334],
                              [0.34378177,  0.08678183],
                              [0.24001692,  0.48324197],
                              [0.81880015,  0.9954541],
                              [0.74728032,  0.59856393]])
        tree = cart_tree.ClassificationTree(max_depth=20, min_samples_leaf=1)
        tree.fit(test_data, test_target)
        self.assertEqual(tree.score([0.75, 0.75]), 1.)
        self.assertEqual(tree.score([0.25, 0.25]), 0.)
        self.assertEqual(tree.score([0.25, 0.75]), 1.)
        self.assertEqual(tree.score([0.75, 0.25]), 0.)

    def test_single_quadrant_separable(self):
        test_target = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
        test_data = np.array([[0.31331166,  0.83733991],
                             [0.42856292,  0.09216958],
                             [0.83379021,  0.58855574],
                             [0.39754771,  0.89547846],
                             [0.65344597,  0.51202825],
                             [0.69115553,  0.16143708],
                             [0.80320466,  0.54689887],
                             [0.20799181,  0.55479467],
                             [0.23448313,  0.80478437],
                             [0.45593921,  0.32604913],
                             [0.8,  0.2]])
        tree = cart_tree.ClassificationTree(max_depth=20, min_samples_leaf=1, verbose=2)
        tree.fit(test_data, test_target)
        self.assertEqual(tree.score([0.99, 0.99]), 1.)
        self.assertEqual(tree.score([0.25, 0.25]), 0.)
        self.assertEqual(tree.score([0.25, 0.75]), 0.)
        self.assertEqual(tree.score([0.75, 0.25]), 0.)

    def test_quadrant_separable(self):
        test_data = np.array([[0.01564607,  0.47523396],
                             [0.85026578,  0.06799358],
                             [0.6094325,  0.19395484],
                             [0.0447059,  0.04507807],
                             [0.4125889,  0.98830637],
                             [0.19468337,  0.76298617],
                             [0.41811272,  0.05896198],
                             [0.77753677,  0.08361111],
                             [0.61621537,  0.7917574],
                             [0.52788686,  0.49538718],
                             [0.9558875,  0.94164169],
                             [0.30943519,  0.37436321],
                             [0.70378457,  0.16307878],
                             [0.66324182,  0.06721279],
                             [0.27245177,  0.25113635],
                             [0.146724,  0.0051949],
                             [0.46219164,  0.16581658],
                             [0.46480223,  0.91925322],
                             [0.36407516,  0.50954243],
                             [0.42817742,  0.38744426],
                             [0.05312313,  0.04133153],
                             [0.30042736,  0.63205188],
                             [0.85089912,  0.90900282],
                             [0.18706373,  0.37055905],
                             [0.72537308,  0.25176723],
                             [0.76675455,  0.29276489],
                             [0.02950254,  0.97001854],
                             [0.82320439,  0.8722766],
                             [0.68152446,  0.33505206],
                             [0.90909761,  0.70129385],
                             [0.4720226,  0.78874071],
                             [0.69195048,  0.7879391],
                             [0.69988636,  0.66094043],
                             [0.87326872,  0.48736539],
                             [0.48143143,  0.65882207],
                             [0.07962102,  0.2499156],
                             [0.17730166,  0.72391327],
                             [0.7083899,  0.92588098],
                             [0.86903646,  0.09342719],
                             [0.41414822,  0.37218929],
                             [0.11518543,  0.52483104],
                             [0.56256031,  0.54309692],
                             [0.17677463,  0.78298492],
                             [0.56024137,  0.235775],
                             [0.12560022,  0.27063403],
                             [0.73197045,  0.80048748],
                             [0.19033346,  0.92975262],
                             [0.02137116,  0.37571995],
                             [0.550401,  0.28781182],
                             [0.12759252,  0.20153411],
                             [0.01727956,  0.12476569],
                             [0.83421287,  0.87644669],
                             [0.36250334,  0.1882965],
                             [0.38553584,  0.2495735],
                             [0.65728543,  0.51008615],
                             [0.79387965,  0.56480074],
                             [0.49978582,  0.59355409],
                             [0.10852298,  0.96111327],
                             [0.96826562,  0.84196345],
                             [0.28329722,  0.85578158],
                             [0.73586675,  0.69480946],
                             [0.27291296,  0.17785806],
                             [0.23550893,  0.63799361],
                             [0.43729547,  0.44940801],
                             [0.36459384,  0.06440611],
                             [0.93133307,  0.88138099],
                             [0.94991882,  0.98879652],
                             [0.94795101,  0.67565538],
                             [0.89186104,  0.67347093],
                             [0.46899736,  0.45440887],
                             [0.50687948,  0.43460291],
                             [0.61607592,  0.59150686],
                             [0.53851451,  0.44430214],
                             [0.05888026,  0.74044027],
                             [0.62473263,  0.33904088],
                             [0.36856443,  0.49667177],
                             [0.32878896,  0.98584409],
                             [0.96272571,  0.8541923],
                             [0.69246982,  0.63369316],
                             [0.79460068,  0.91756444],
                             [0.15763276,  0.34231972],
                             [0.17240017,  0.87373729],
                             [0.90232529,  0.84136069],
                             [0.70588187,  0.44191484],
                             [0.28337266,  0.72598487],
                             [0.67415721,  0.212771],
                             [0.13870175,  0.35985181],
                             [0.97358983,  0.32418814],
                             [0.9120423,  0.52882271],
                             [0.09588293,  0.05521627],
                             [0.57849555,  0.45660152],
                             [0.6121643,  0.08391775],
                             [0.0623358,  0.20536525],
                             [0.72531705,  0.36815045],
                             [0.54799431,  0.91802267],
                             [0.00789307,  0.7484025],
                             [0.76985369,  0.92943061],
                             [0.24165921,  0.19433648],
                             [0.70394644,  0.19267303],
                             [0.96347949,  0.93256121]])
        test_target = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0,
                               0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0,
                               1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                               0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1,
                               0, 1, 0, 1, 0, 0, 1, 0])
        tree = cart_tree.ClassificationTree(max_depth=10, min_samples_leaf=1, verbose=1)
        tree.fit(test_data, test_target)
        self.assertEqual(tree.score([0.25, 0.25]), 0.)
        self.assertEqual(tree.score([0.75, 0.25]), 1.)
        self.assertEqual(tree.score([0.75, 0.25]), 1.)
        self.assertEqual(tree.score([0.75, 0.75]), 0.)


if __name__ == '__main__':
    unittest.main()

