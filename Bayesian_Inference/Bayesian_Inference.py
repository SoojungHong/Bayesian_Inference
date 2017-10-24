# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:39:10 2017

@author: a613274
@path : C:\Users\a613274\Bayesian_Inference
Reference : https://www.datascience.com/blog/introduction-to-bayesian-inference-learn-data-science-tutorials

"""
#-----------------------------------------------------------------
# Likelihood of various Theta of given data 
# theta is the probability that user will click the ad 
# P(x¦theta) is likelihood function if I set parameter theta  
#-----------------------------------------------------------------

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
    https://onlinecourses.science.psu.edu/stat504/node/27
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
possible_theta_values = map(lambda x: x/100., range(100)) # map(func, seq)
possible_theta_values

#evaluate the likelihood function for possible click through rates
likelihoods = map(lambda theta: likelihood(theta, n_impressions, n_clicks)\
                                , possible_theta_values)

#pick the best theta that maximum likelihood from likelihoods distribution
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


#-------------------------------------------------------------
# Overlay Likelihood with previous 100 campaign on the above 
#-------------------------------------------------------------
plt.rcParams['figure.figsize'] = (10, 5) # (16, 7)
import numpy as np
#import pandas as pd

true_a = 11.5
true_b = 48.5

#number of marketing campaigns
N = 100

#randomly generate "true" click through rate for each campaign
p = np.random.beta(true_a,true_b, size=N) #beta distribution
p

#randomly pick the number of impressions (number of exposure to users) for each campaign
impressions = np.random.randint(1, 10000, size=N)
impressions

#sample number of clicks for each campaign
clicks = np.random.binomial(impressions, p).astype(float)
clicks
click_through_rates = clicks / impressions
click_through_rates

#plot the histogram of previous click through rates with the evidence#of the new campaign
f, ax = plt.subplots(1)
ax.axvline(mle, linestyle = "--")
ax.plot(possible_theta_values, likelihoods)

zero_to_one = [j/100. for j in xrange(100)]

counts, bins = np.histogram(click_through_rates
                            , bins=zero_to_one) #If bins is an int, it defines the number of equal-width bins in the given range (10, by default). 
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
prior_parameters 

#extract a,b from fit
prior_a, prior_b = prior_parameters[0:2]
prior_a
prior_b

#define prior distribution sample from prior
prior_distribution = beta(prior_a, prior_b)
prior_distribution

#get histogram of samples
prior_samples = prior_distribution.rvs(10000) #rvs : produces a single value of a pseudorandom variable
prior_samples

#get histogram of samples
fit_counts, bins = np.histogram(prior_samples, zero_to_one)
fit_counts

#normalize histogram
fit_counts = map(lambda x: float(x)/fit_counts.sum(), fit_counts)
fit_counts

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

#---------------------------------------------------------------------
# Obtain from Posterior, we select our prior as a Beta (11.5, 48.5)
#---------------------------------------------------------------------
import pymc3 as pm
import numpy as np

#create our data:clicks = np.array([n_clicks])
#clicks represents our successes. We observed 7 clicks.
#randomly generate "true" click through rate for each campaign
#impressions = np.array([n_impressions])
#this represents the number of trials. There were 10 impressions.

with pm.Model() as model: #pm.Model creates a PyMC model object.. as model assigns it to the variable name 
#sets a context; all code in block "belongs" to the model object

    theta_prior = pm.Beta('prior', 11.5, 48.5) #Theta_prior represents a random variable for click-through rates. It will serve as our prior distribution for the parameter θ, the click-through rate of our facebook-yellow-dress campaign. 
    
    #our prior distribution, Beta (11.5, 48.5)
    observations = pm.Binomial('obs',n = impressions
                               , p = theta_prior
                               , observed = clicks)     #Sampling distribition of outcomes in the dataset.
    #our prior p_prior will be updated with data


    #The MAP estimate is the found by optimizing all of the parameters to maximize the posterior probability. We only need to optimize the transformed variable, since this fully determines the value of the variable you originally added to the model.  pm.find_MAP() returns only the variables being optimized, not the original variables. 
    start = pm.find_MAP()    #find good starting values for the sampling algorithm
    #Max Aposterior values, or values that are most likely

    #step = pm.NUTS(state=start)     #Choose a particular MCMC algorithm     #we'll choose NUTS, the No U-Turn Sampler (Hamiltonian)
    step = pm.NUTS()

    trace = pm.sample(5000
                      , step
                      , start=start
                      , progressbar=True)               #obtain samples
                      
    
