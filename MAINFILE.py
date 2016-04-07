
import numpy as np
import matplotlib.pyplot as plt
from pwl_compressor_core.compressor import PWLcompressor

np.random.seed(1232121334)

##Minimal example:
# from CompressorClass import PWLcompressor
#Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
# PWLapprox = PWLcompressor(Sample, MakePWLsmoother=True, Accuracy=0.01)

# lognormal parametrization
def LognormalParameters(mean = 1.0, CoV = 0.1):
    sigma2  = np.log(CoV*CoV+1)
    LNsigma = np.sqrt(sigma2)
    LNmu    = np.log(mean)-sigma2/2
    return [LNmu,LNsigma]

[mu, sigma] = LognormalParameters(mean = 10., CoV = 0.1) #(mean = 3.5, CoV = 0.8)
n = 1000000
Sample = np.random.lognormal(mean = mu, sigma = sigma, size = (n,))
# Sample = Sample - np.minimum(1,np.maximum(Sample-3,0)) # apply 1 xs 3 reinsurance layer to get jump in sample
#Accuracy = 0.1#0.4
QuantileList = [] #[0.25, 0.5, 0.75]

# run Compression with no Negative Increment removal, no smoothing, no Strict admissibility check
#CompressedSample = PWLcompressor(Sample, RemoveNegativeJumps = False, MakePWLsmoother = False, CheckStrictAdmissibility = False, Accuracy = Accuracy, EnforcedInterpolationQuantiles = QuantileList, Verbose=True)
#Step1 = CompressedSample.plot('r', ShowPlot = False)

# run Compression with Negative Increment removal, but no smoothing, no Strict admissibility check
#CompressedSample = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = False, CheckStrictAdmissibility = False, Accuracy = Accuracy, EnforcedInterpolationQuantiles = QuantileList, Verbose=True)
#Step2 = CompressedSample.plot('g', ShowPlot = False)

# run Compression with Negative Increment removal, smoothing, but no Strict admissibility check
CompressedSample = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False, Accuracy = 0.1, EnforcedInterpolationQuantiles = QuantileList, Verbose=True)
#Step3 = CompressedSample.plot('g', ShowPlot = False)
CompressedSample.GiveSolIntervals('SolNonStrict1.csv')

CompressedSample = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = True, Accuracy = 0.1, EnforcedInterpolationQuantiles = QuantileList, Verbose=True)
#Step3 = CompressedSample.plot('g', ShowPlot = False)
CompressedSample.GiveSolIntervals('SolStrict1.csv')

CompressedSample = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False, Accuracy = 0.001, EnforcedInterpolationQuantiles = QuantileList, Verbose=True)
#Step3 = CompressedSample.plot('g', ShowPlot = False)
CompressedSample.GiveSolIntervals('SolNonStrict001.csv')

CompressedSample = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = True, Accuracy = 0.001, EnforcedInterpolationQuantiles = QuantileList, Verbose=True)
#Step3 = CompressedSample.plot('g', ShowPlot = False)
CompressedSample.GiveSolIntervals('SolStrict001.csv')




print(CompressedSample.GivePWLPoints())
if False:

   # plt.plot(Step1['SampleX'],Step1['SampleY'],color='black')
   # plt.plot(Step1['PWLX'],Step1['PWLY'], linewidth=2.0, linestyle = '-', marker = 'o', color = 'r')
   # plt.plot(Step2['PWLX'],Step2['PWLY'], linewidth=2.0, linestyle = '-', marker = 'o', color = 'g')
    plt.plot(Step3['PWLX'],Step3['PWLY'], linewidth=2.0, linestyle = '-', marker = 'o', color = 'y')
    plt.show()


# np.savetxt('AfterSmoothing.csv',[Step3['PWLX'],Step3['PWLY']], delimiter = ";")

