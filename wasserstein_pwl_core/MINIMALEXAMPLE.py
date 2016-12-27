
## Minimal example:
from wasserstein_pwl_core.compressor import PWLcompressor
from Graphics.GenericIllustrationMethods import XLsimulations, LognormalParameters
import numpy as np

n = 100000

MEAN = 6.04405167667
VAR = 44.4314748556


[mu, sigma] = LognormalParameters(mean = MEAN, CoV = np.sqrt(VAR)/MEAN)

p = MEAN/VAR
r = MEAN**2/(VAR-MEAN)

np.random.seed(123)

#Sample = np.random.normal(MEAN, np.sqrt(VAR), n)
#Sample = np.random.negative_binomial(r, p, n)
#Sample = np.random.lognormal(mean = mu, sigma = sigma, size = (n,))
Sample = XLsimulations(SampleSize = n, PoissonLambda = 2, ParetoX0 = 10, ParetoAlpha = 2.5, Deductible = 12, Limit = 10, AggregateLimit = 30)
#Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
#PWLapprox =  PWLcompressor(Sample, Accuracy = 0.01, AtomDetectionMinimumSampleSize = 100, RelativeAtomDetectionThreshold = 0.05, Bisection = 'Original', PlotIntermediate = False, Verbose = True)
PWLapprox =  PWLcompressor(Sample, Accuracy = 0.01, AtomDetection = (1000, 0.015), Bisection = 'OLS', PlotIntermediate = False, Verbose = True)
#PWLapprox =  PWLcompressor(Sample, Accuracy = 0.01, AtomDetection= True, Bisection = 'OLS', PlotIntermediate = False, Verbose = True)
print(PWLapprox.Result['PWLX'])

print(PWLapprox.Result['PWLY'])


PWLapprox.plot()


# Sample = [0.8, 7.4, 12.9, 17.7]
# PWLapprox = PWLcompressor(Sample, Accuracy=0.5, EnforcedInterpolationQuantiles= [0.6])
#
# print(PWLapprox.SampleStats.LocalMean(0,1))
# print(PWLapprox.SampleStats.LocalMean(2,3))
#
# Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
# PWLapprox = PWLcompressor(Sample, Accuracy=0.01)
# print(PWLapprox.Result['PWLX'])
# print(PWLapprox.Result['PWLY'])
# PWLapprox.GiveSolIntervals('SOL')