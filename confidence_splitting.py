import numpy as np

coeff1= 1.98
coeff2= 1.135
sqrt2= np.sqrt(2.)
sqrtpi= np.sqrt(np.pi)

def approximate_q_function(arg):
    # Method of Karagiannidis & Lioumpas 2007
    x = arg / sqrt2
    numerator = (1.0 - np.exp(-coeff1 * x)) * np.exp(-x**2)
    denominator = coeff2 * sqrtpi * x
    erfc = numerator / denominator
    return erfc / 2.0


def approximate_binomial(num_successes, num_trials, probability):
    mean = probability * num_trials
    std = np.sqrt(num_trials * probability * (1 - probability))
    dev = (num_successes - mean) / std
    return approximate_q_function(dev)


def confidence_criterion():
    # calculate binomial confidence criterion.
    # min( 1 - Xl0 . B(L_0 | L, p_0) , 1 - Xl1 . B(L_1 | L, p_1) )
    # * min( 1 - Xr0 . B(R_0 | R, p_0) , 1 - Xr1 . B(R_1 | R, p_1) )
    return None