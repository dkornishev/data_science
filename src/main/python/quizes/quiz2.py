import math

import scipy.stats as stats
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# set the values of sample mean and sample standard deviation
x_bar, s = 160.9, 6

# set the value of sample size and degrees of freedom
n = 50
k = n - 1

# construct the confidence interval
np.round(stats.t.interval(0.95, df=k, loc=x_bar, scale=s / np.sqrt(n)), 2)

normal_pop = stats.norm.rvs(1, 0.2, size=10000)
# visualize the normal distribution
plt.hist(normal_pop, 200)
plt.title("Normal Distribution Population")
plt.xlabel("X~N(0,1)")
plt.ylabel("Count")
plt.show()

sample_means = []
n =35
# iterate the loop to draw multiple samples
for j in range(500):
    # draw a sample of size n
    sample = np.random.choice(normal_pop, size = n)
    # calculate the sample mean
    sample_mean = np.mean(sample)
    # append the sample mean to the sample_means list
    sample_means.append(sample_mean)
# plot the histogram of sample means
sns.displot(sample_means, kde = True)
plt.title('Distribution of Sample Means for n = ' + str(n))
plt.xlabel('sample mean')
plt.ylabel('count')
plt.show()

print(stats.norm.ppf(q=0.965, loc=1, scale=0.2/math.sqrt(35)))