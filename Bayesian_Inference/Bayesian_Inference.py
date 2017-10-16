# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:39:10 2017

@author: a613274

Reference : https://www.datascience.com/blog/introduction-to-bayesian-inference-learn-data-science-tutorials

"""

import numpy as np
from scipy.misc import factorial
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 5) #(16,7)


def likelihood(theta, n, x):
    """
    likelihood function for a binomial distribution

    n: [int] the number of experiments
    x: [int] the number of successes
    theta: [float] the proposed probability of success
    """
    return (factorial(n) / (factorial(x) * factorial(n - x))) \
            * (theta ** x) * ((1 - theta) ** (n - x))

#the number of impressions for our facebook-yellow-dress campaign
n_impressions = 10.

#the number of clicks for our facebook-yellow-dress campaign
n_clicks = 7.
#observed click through rate
ctr = n_clicks / n_impressions
#0 to 1, all possible click through rates
possible_theta_values = map(lambda x: x/100., range(100))

#evaluate the likelihood function for possible click through rates
likelihoods = map(lambda theta: likelihood(theta, n_impressions, n_clicks)\
                                , possible_theta_values)

#pick the best theta
mle = possible_theta_values[np.argmax(likelihoods)]
#plot
f, ax = plt.subplots(1)
ax.plot(possible_theta_values, likelihoods)
ax.axvline(mle, linestyle = "--")
ax.set_xlabel("Theta")
ax.set_ylabel("Likelihood")
ax.grid()
ax.set_title("Likelihood of Theta for New Campaign")
plt.show()