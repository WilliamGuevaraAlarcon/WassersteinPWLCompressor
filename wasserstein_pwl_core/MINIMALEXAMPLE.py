#
## Minimal example:
from wasserstein_pwl_core.compressor import PWLcompressor
from Graphics.GenericIllustrationMethods import XLsimulations, LognormalParameters
from wasserstein_pwl_core.pwl_distribution import PiecewiseLinearDistribution, EmpiricalDistributionFromSample
import numpy as np
import pandas as pd

n = 100000

MEAN = 6.04405167667
VAR = 44.4314748556


[mu, sigma] = LognormalParameters(mean = MEAN, CoV = np.sqrt(VAR)/MEAN)

p = MEAN/VAR
r = MEAN**2/(VAR-MEAN)

np.random.seed(123)

#Data = pd.read_csv("/home/william/Downloads/Autoseg2014B/arq_casco_comp.csv", sep = ";", decimal = ",")
Data = pd.read_csv("/media/william/WILLIAM/Datos.csv", sep = ",", decimal = ".")
Data['ClaimAmountTotal'] = Data['ClaimAmountFire'] + Data['ClaimAmountOther'] + Data['ClaimAmountPartColl'] \
                           + Data['ClaimAmountRob'] + Data['ClaimAmountTotColl']

Data.filter = Data.query('ClaimAmountTotal>0')

#Sample = Data.filter['ClaimAmountTotal']
Sample = Data.filter['ClaimAmountPartColl']
print(Sample.size)
#Sample = Data.IS_MEDIA.dropna()
# Sample = Sample[~np.isnan(Sample)]
# Data



#Sample = np.random.normal(MEAN, np.sqrt(VAR), n)
#Sample = np.random.negative_binomial(r, p, n)
#Sample = np.random.lognormal(mean = mu, sigma = sigma, size = (n,))
#Sample = XLsimulations(SampleSize = n, PoissonLambda = 2, ParetoX0 = 10, ParetoAlpha = 2.5, Deductible = 12, Limit = 10, AggregateLimit = 30)
#Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 15.6, 16, 18.8]
#Sample = [1, 1.1, 1.2, 1.6, 4.3, 4.5, 4.6, 6, 6.1, 6.6,
#         7.1, 13, 13.4, 16, 18.8, 22, 30, 32, 39, 40]


PWLapprox =  PWLcompressor(Sample, AtomDetection = False, AccuracyMode="Relative", AccuracyParameter=0.1, CheckStrictWasserstein = True, PlotIntermediate = False, Verbose = True)
#PWLapprox =  PWLcompressor(Sample, RemoveNegativeJumps=True, AccuracyMode = "Absolute", AccuracyParameter = 1.5, AtomDetection= True, PlotIntermediate = False, Verbose = True)

print(PWLapprox.Result['PWLX'])

print(PWLapprox.Result['PWLY'])


#G = PiecewiseLinearDistribution(PWLapprox.Result['PWLX'], PWLapprox.Result['PWLY'])
F = EmpiricalDistributionFromSample(Sample)
# print(F.cdf(4.2999))
# print(G.cdf(4.2999))
# print(G.cdf(4.1923))
# print(G.cdf(4.4075))


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