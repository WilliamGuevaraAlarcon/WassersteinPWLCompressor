

import numpy as np
import matplotlib.pyplot as plt
from pwl_compressor_core.compressor import PWLcompressor
from pwl_compressor_core.pwl_distribution import PiecewiseLinearDistribution

# np.random.seed(1232121334)

#### PARAMETERS ####
mean = 10
CoV = 0.1
SampleSize = 10000
AccuracyList = [0.1,0.01,0.001,0.0001]

# lognormal parametrization
def LognormalParameters(mean = 1.0, CoV = 0.1):
    sigma2  = np.log(CoV*CoV+1)
    LNsigma = np.sqrt(sigma2)
    LNmu    = np.log(mean)-sigma2/2
    return [LNmu,LNsigma]

def ExpectedWassersteinError(sample):
    # calculates sqrt(2/pi)*\int_{-inf}^{+inf} sqrt( F_n(x)*(1-F_n(x)) ) dx
    # for F_n empirical distribution of sample
    sample.sort()
    ssize = len(sample)
    empdist = np.linspace(1/ssize,1-1/ssize,ssize-1)
    dx = np.diff(sample)
    integrand = np.sqrt( empdist*(1-empdist) )
    integral = np.sum(integrand*dx)
    integral *= np.sqrt(2/np.pi)
    integral /= np.sqrt(ssize)
    return integral

def WasserssteinDistance(pwldist,sample):
    sample.sort()
    ssize = len(sample)
    discretizerquantiles = np.linspace(0.5/ssize,1-0.5/ssize,ssize)
    discretizedpwl = pwldist.quantile(discretizerquantiles)
    WasserssteinDist = np.sum(np.abs(discretizedpwl-sample))/ssize
    return WasserssteinDist


[mu, sigma] = LognormalParameters(mean = mean, CoV = CoV) #(mean = 3.5, CoV = 0.8)
Sample = np.random.lognormal(mean = mu, sigma = sigma, size = (SampleSize,))



PWLlist = []
for Accuracy in AccuracyList:
    CompressedSample = PWLcompressor(Sample, Accuracy = Accuracy)
    pwl = PiecewiseLinearDistribution(CompressedSample.Result['PWLX'],CompressedSample.Result['PWLY'])
    PWLlist.append(pwl)


print('expected WasserssteinDistance true vs sample: '+str(ExpectedWassersteinError(Sample)))

for Accuracy,pwl in zip(AccuracyList,PWLlist):
    wdist = WasserssteinDistance(pwl,Sample)
    print('Accuracy: '+str(Accuracy) + ' Wasserstein distance pwl vs sample: '+str(wdist))


