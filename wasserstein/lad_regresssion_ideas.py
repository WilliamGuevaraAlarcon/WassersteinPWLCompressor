
import matplotlib.pyplot as plt
import numpy as np

from pwl_compressor_core.pwl_distribution import PiecewiseLinearDistribution, EmpiricalDistributionFromSample
from pwl_compressor_core.sample_characteristics import SampleCharacteristics
from scipy.integrate import quad


Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]

# pwl_sample = EmpiricalDistributionFromSample(Sample)
# pwl_sample.plot()


# delta = 6
# mean = pwl_sample.expected()
# pwl_reg = PiecewiseLinearDistribution([mean-delta,mean+delta],[0,1])

# plt.plot(pwl_sample.Xvalues, pwl_sample.Fvalues, linewidth=2.0, linestyle = '-', marker = 'o', color = 'g')
# plt.plot(pwl_reg.Xvalues, pwl_reg.Fvalues, linewidth=2.0, linestyle = '-', marker = 'o', color = 'y')

# def pwlregquantilefunction(prmean,prdelta):
#     def quantilefun(x):
#         return prmean + prdelta*(2*x-1)
#     return quantilefun

def absolutedeviation(InSample, DeltaList):
    print(InSample)
    sample = np.asarray(InSample)
    sample.sort()
    pwl_sample = EmpiricalDistributionFromSample(sample)
    SampleSize = len(sample)
    mean = pwl_sample.expected()
    discretizerquantiles = np.linspace(0.5/SampleSize,1-0.5/SampleSize,SampleSize)
    ADapprox = list()
    ADaccurate = list()
    for delta in DeltaList:
        def linapprox(x):
            return mean + delta*(2*x-1)
        discretizedlinapprox = linapprox(discretizerquantiles)
        ADapprox.append( np.sum(np.abs(discretizedlinapprox-sample))/SampleSize )
        def integrand(p):
            return np.abs(linapprox(p) -pwl_sample.quantile(p))
        ADaccurate.append( quad( integrand, 0, 1)[0] )

    return ADapprox, ADaccurate

# DeltaList = np.linspace(0,20,50)
DeltaList = np.linspace(8,12,50)
ADapprox, ADaccurate = absolutedeviation(Sample,DeltaList)

SampleStats = SampleCharacteristics(Sample)
RegressionDelta = SampleStats.CalculateRegressionDelta(0,len(Sample)-1)

plt.plot(DeltaList, ADapprox, linewidth=2.0, linestyle = '-', marker = 'o', color = 'r')
plt.plot(DeltaList, ADaccurate, linewidth=2.0, linestyle = '-', marker = 'o', color = 'y')

print(RegressionDelta)


plt.show()
