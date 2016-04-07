__author__ = 'WGUEVARA'
from pwl_compressor_core.pwl_distribution import EmpiricalDistributionFromSample, PiecewiseLinearDistribution
from pwl_compressor_core.compressor import PWLcompressor

import numpy as np
import matplotlib.pyplot as plt

# Sample = np.asarray([ 24.  ,  26.75,  27.4 ,  27.45,  30.15,  30.5 ,  31.45,  32.7 , 32.7 ,  33.8 ]) #sorted!
#Sample = np.asarray([ 1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]) #sorted!
n = 100000
def LognormalParameters(mean = 1.0, CoV = 0.1):
    sigma2  = np.log(CoV*CoV+1)
    LNsigma = np.sqrt(sigma2)
    LNmu    = np.log(mean)-sigma2/2
    return [LNmu,LNsigma]

[mu, sigma] = LognormalParameters(mean = 10., CoV = 0.1) #(mean = 3.5, CoV = 0.8)


Sample = np.random.lognormal(mean = mu, sigma = sigma, size = (n,))
SampleDist = EmpiricalDistributionFromSample(Sample)
##Sample = XLsimulations(SampleSize = n, PoissonLambda = 2, ParetoX0 = 10, ParetoAlpha = 2.5, Deductible = 12, Limit = 10, AggregateLimit = 30)
#Sample = np.random.normal(loc = 10, scale = 10, size = (n,))


N = len(Sample)
k = 6
m1 = np.mean(Sample[:k])
m2 = np.mean(Sample[k:])

int1 = np.sum(Sample[:k])/N
int2 = np.sum(Sample[k:])/N
print(int1)
print(int2)

delta1 = 3.3
delta2 = 2.4
#PWLapprox = PiecewiseLinearDistribution(
#                [m1-delta1,m1+delta1,m2-delta2,m2+delta2],
#                [0,k/N,k/N,1])

Compres = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False, Accuracy = 0.1, Verbose=True)
PWLapprox=Compres.Result
print('PWL approx: X='+str(PWLapprox.Xvalues)+'. Y= '+str(PWLapprox.Fvalues))
print(abs((PWLapprox.Xvalues[0]+PWLapprox.Xvalues[1])/2*(PWLapprox.Fvalues[1]-PWLapprox.Fvalues[0])-int1))
print(abs((PWLapprox.Xvalues[2]+PWLapprox.Xvalues[3])/2*(PWLapprox.Fvalues[3]-PWLapprox.Fvalues[2])-int2))

plt.figure(figsize=(6,4), dpi=200) #figure size and resolution
plt.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=1.0, linestyle = '--', marker = '', color = 'k')
plt.plot(PWLapprox.Xvalues,PWLapprox.Fvalues,   linewidth=1.0, linestyle = '-', marker = '', color = 'k')


step = 0.001
alphaList = np.linspace(step,1-step,1/step-1)
SampleXTVaRs = np.zeros_like(alphaList)
PWLXTVaRs = np.zeros_like(alphaList)
for i in range(len(alphaList)):
    SampleXTVaRs[i] = SampleDist.xTVaR(alphaList[i],"upper")
    PWLXTVaRs[i] = PWLapprox.xTVaR(alphaList[i],"upper")

#np.savetxt('PWLVaR.txt', np.array(PWLXTVaRs), delimiter="\t")
#np.savetxt('SampleVaR.txt', np.array(SampleXTVaRs), delimiter="\t")



plt.figure(figsize=(6,4), dpi=200) #figure size and resolution
plt.plot(alphaList,SampleXTVaRs, linestyle = '--',color='k')
plt.plot(alphaList,PWLXTVaRs, linestyle = '-',color='k')
plt.plot(alphaList,SampleXTVaRs*(1+.17), linestyle = '-',color='r')
plt.plot(alphaList,SampleXTVaRs*(1-.17), linestyle = '-',color='r')
## plt.savefig('MeanInvarianceExampleXTVAR.png')
plt.show()



relError = np.abs(PWLXTVaRs-SampleXTVaRs)/SampleXTVaRs
plt.plot(alphaList,relError)
plt.show()


# alpha = 0.123
# print(SampleDist.xTVaR(alpha))
# print(PWLapprox.xTVaR(alpha))
