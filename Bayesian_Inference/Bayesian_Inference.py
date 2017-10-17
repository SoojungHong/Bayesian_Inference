# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:39:10 2017

@author: a613274

Reference : https://www.datascience.com/blog/introduction-to-bayesian-inference-learn-data-science-tutorials

"""
#--------------------------------------
# Likelihood of Theta of sinlge run 
#--------------------------------------

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


#------------------------------------------------
# Overlay Likelihood with previous 100 campaign 
#------------------------------------------------
plt.rcParams['figure.figsize'] = (10, 5) # (16, 7)
import numpy as np
#import pandas as pd

true_a = 11.5
true_b = 48.5

#number of marketing campaigns
N = 100

#randomly generate "true" click through rate for each campaign
p = np.random.beta(true_a,true_b, size=N) #beta distribution

#randomly pick the number of impressions for each campaign
impressions = np.random.randint(1, 10000, size=N)

#sample number of clicks for each campaign
clicks = np.random.binomial(impressions, p).astype(float)
click_through_rates = clicks / impressions

#plot the histogram of previous click through rates with the evidence#of the new campaign
f, ax = plt.subplots(1)
ax.axvline(mle, linestyle = "--")
ax.plot(possible_theta_values, likelihoods)

zero_to_one = [j/100. for j in xrange(100)]
counts, bins = np.histogram(click_through_rates
                            , bins=zero_to_one)
counts = counts / 100.
ax.plot(bins[:-1],counts, alpha = .5)
line1, line2, line3 = ax.lines
ax.legend((line2, line3), ('Likelihood of Theta for New Campaign'
                           , 'Frequency of Theta Historically')
                          , loc = 'upper left')
ax.set_xlabel("Theta")
ax.grid()
ax.set_title("Evidence vs Historical Click Through Rates")
plt.show()

#--------------------------------------------------------------------------------------------------------------
# fit the beta distribution and compare the estimated prior distribution with previous click-through rates 
# to ensure the two are properly aligned:
#--------------------------------------------------------------------------------------------------------------
from scipy.stats import beta

#fit beta to previous CTRs
prior_parameters = beta.fit(click_through_rates
                            , floc = 0
                            , fscale = 1)
#extract a,b from fit
prior_a, prior_b = prior_parameters[0:2]

#define prior distribution sample from prior
prior_distribution = beta(prior_a, prior_b)

#get histogram of samples
prior_samples = prior_distribution.rvs(10000)

#get histogram of samples
fit_counts, bins = np.histogram(prior_samples, zero_to_one)

#normalize histogram
fit_counts = map(lambda x: float(x)/fit_counts.sum(), fit_counts)

#plot
f, ax = plt.subplots(1)
ax.plot(bins[:-1], fit_counts)

hist_ctr, bins = np.histogram(click_through_rates, zero_to_one)
hist_ctr = map(lambda x: float(x)/hist_ctr.sum(), hist_ctr)
ax.plot(bins[:-1], hist_ctr)
estimated_prior, previous_click_through_rates = ax.lines
ax.legend((estimated_prior, previous_click_through_rates)
          ,('Estimated Prior'
            , 'Previous Click Through Rates'))
ax.grid()
ax.set_title("Comparing Empirical Prior with Previous Click Through Rates")
plt.show()