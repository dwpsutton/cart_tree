import numpy as np
import math

coeff1 = 1.98
coeff2 = 1.135
sqrt2 = np.sqrt(2.)
sqrtpi = np.sqrt(np.pi)


def q_function(arg):
    return 0.5-0.5*math.erf(arg/np.sqrt(2))


def approximate_q_function(arg):
    # Method of Karagiannidis & Lioumpas 2007
    x = arg / sqrt2
    numerator = (1.0 - np.exp(-coeff1 * x)) * np.exp(-x**2)
    denominator = coeff2 * sqrtpi * x
    erfc = numerator / denominator
    return erfc / 2.0


def approximate_erf(arg):
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    if arg < 0.:
        marg = -arg
        t = 1.0 / (1.0 + p*marg)
        approx = 1.0 - (a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5) * math.exp(-marg**2)
        return -approx
    else:
        marg = arg
        t = 1.0 / (1.0 + p*marg)
        approx = 1.0 - (a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5) * math.exp(-marg**2)
        return approx


def approximate_q_function_2(dev):
    return 0.5-0.5 * approximate_erf(dev / np.sqrt(2))


def approximate_binomial(num_successes, num_trials, probability):
    mean = probability * num_trials
    std = np.sqrt(num_trials * probability * (1 - probability))
    dev = (num_successes - mean) / std
    return approximate_q_function_2(dev)


class confidence_splitter(object):
    def __init__(self, threshold_false=0.0, threshold_true=0.0):
        self.C_false = threshold_false
        self.C_true = threshold_true

    def confidence_criterion(self, false_left, true_left, false_right, true_right):
        # calculate binomial confidence criterion. This computes the probability that the split has occurred by chance.
        # min( 1 - Xl0 . B(L_0 | L, p_0) , 1 - Xl1 . B(L_1 | L, p_1) )
        # * min( 1 - Xr0 . B(R_0 | R, p_0) , 1 - Xr1 . B(R_1 | R, p_1) )
        # calculate B(L_0 | L, p_0)
        # if this is above the criterion, return 1.0 - B(L_0 | L, p_0)
        num_left = false_left + true_left
        num_right = false_right + true_right
        parent_false_probability = (false_left + false_right) / float(num_left + num_right)
        parent_true_probability = (true_left + true_right) / float(num_left + num_right)
        p_left_false = 1.0 - approximate_binomial(false_left, num_left, parent_false_probability)
        p_left_true = 1.0 - approximate_binomial(true_left, num_left, parent_true_probability)
        p_right_false = 1.0 - approximate_binomial(false_right, num_right, parent_false_probability)
        p_right_true = 1.0 - approximate_binomial(true_right, num_right, parent_true_probability)
        if p_left_false < self.C_false:
            p_left_false = 0.0
        if p_right_false < self.C_false:
            p_right_false = 0.0
        if p_left_true < self.C_true:
            p_left_true = 0.0
        if p_right_true < self.C_true:
            p_right_true = 0.0
        return math.min(p_left_false, p_left_true) * math.min(p_right_false, p_right_true)
