


import numpy as np
import matplotlib.pyplot as plt
from pwl_compressor_core.compressor import PWLcompressor
from pwl_compressor_core.pwl_distribution import PiecewiseLinearDistribution, EmpiricalDistributionFromSample
from Graphics.GenericIllustrationMethods import MyMark_inset, Characteristics, BoundPlot, SetAxisBoundLineWidth, PlotBrace, LognormalParameters, XLsimulations
from pwl_compressor_core.sample_characteristics import SampleCharacteristics
from matplotlib import rcParams
from matplotlib.transforms import Bbox, TransformedBbox, IdentityTransform
#from matplotlib.patches import Patch
#from matplotlib.path import Path
#from Graphics.ConvergenceRateClass import ConvergeRateCalculator



Sample = np.asarray([ 1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]) #sorted!
SampleLength = len(Sample)
SampleDist = EmpiricalDistributionFromSample(Sample)

#from Illustrations import BoundPlot


fig, ax = plt.subplots(1, figsize=(9.5,4.5), dpi=200)
CompressedSample = PWLcompressor(Sample, Accuracy = 1, EnforcedInterpolationQuantiles = [])
C1 = CompressedSample.plot('g', ShowPlot = False)
CompressedSample = PWLcompressor(Sample, Accuracy = 0.1, EnforcedInterpolationQuantiles = [])
C2 = CompressedSample.plot('g', ShowPlot = False)
# CompressedSample = PWLcompressor(Sample, Accuracy = 1, EnforcedInterpolationQuantiles = [0.4])
# C3 = CompressedSample.plot('g', ShowPlot = False)

ax.set_xlim([-2,21])
ax.set_ylim([-0.05,1.05])

ax.plot(C1['SampleX'],C1['SampleY'],color='black', linewidth=2.5, label='Sample distribution (Sample size = 10)')
ax.plot(C1['PWLX'],C1['PWLY'], linewidth=1.8, linestyle = '-', marker = 'o', markersize = 5, color = 'b', fillstyle = 'none', label='Some PWL distribution')
ax.plot(C2['PWLX'],C2['PWLY'], linewidth=1.8, linestyle = '-', marker = 'o', markersize = 5, color = 'r', fillstyle = 'none', label='Another PWL distribution')
# ax.plot(C3['PWLX'],C3['PWLY'], linewidth=1.1, linestyle = '-', marker = 'o', markersize = 3, color = 'g', fillstyle = 'none')
legend = ax.legend(loc='lower right', shadow=True)



# plt.show()
plt.savefig('PWLPossibilities.png', bbox_inches='tight', transparent=True)
plt.close()

del fig, ax, Sample, SampleDist, CompressedSample


##############################################################
######################### xTVaR exponential illustration########################
##############################################################
print("\n =========== xTVaR exponential illustration ============ \n")

fig, ax = plt.subplots(1, figsize=(8,3), dpi=200) #(6,3)

MinX = -0.1
MaxX = 30
#  K =4, x =(1, 4, 4,9), y = (0, 0.6,0.8,1),
MyMarkerSize = 9

theta = 10
alpha = 0.8

x = np.linspace(0,MaxX,10000)
y = 1-np.exp(-x/theta)

ax.plot(x,y)

VaR = - theta * np.log(1-alpha)
Mean = theta
TVaR = VaR+Mean
BarHeight = alpha

ax.plot([TVaR,TVaR],[0,.926],   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([VaR,VaR],[0,BarHeight],   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([Mean,Mean],[0,BarHeight],   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.text(TVaR-2.2, 0.7 , '$TVaR_{0.8}[X]$', verticalalignment='bottom', horizontalalignment='center', fontsize=14)
ax.text(Mean-1, 0.7, '$\mathbb{E}[X]$' , verticalalignment='bottom', horizontalalignment='center', fontsize=14)
ax.text(VaR-1.8, 0.4, '$VaR_{0.8}[X]$' , verticalalignment='bottom', horizontalalignment='center', fontsize=14)
ax.text(5, 0.9, '$(1-0.8)TVaR_{0.8}^{\Delta}[X]$' , verticalalignment='bottom', horizontalalignment='center', fontsize=14)
x1=np.ma.masked_less(x, Mean)
ax.fill_between(x,y1 = y, y2=1, where= x>=VaR, facecolor='k', alpha=0.5, linewidth = 0, color = 'k')
ax.fill_between(x1,y1 = 0.8, y2=1, where= x1<=VaR, facecolor='k', alpha=0.5, linewidth = 0, color = 'k')
ax.arrow(10, 0.95, -1.45, 0, head_width=0.05, head_length=0.1, fc='k', ec='k')

PlotBrace(ax, x1 = Mean, x2 = TVaR, y = 0.7, braceheight = 0.03, bracewidth = 0.5, text = '  $TVaR_{0.8}^{\Delta}[X]$', upper=False)

ax.set_xlim([MinX,MaxX])
ax.set_ylim([-0.05,1.15])

plt.savefig('xTVaRExpillustration.png')
plt.close()

del fig, ax, alpha, theta, x, y, VaR, Mean, TVaR, BarHeight, x1

##############################################################
######################### Step by step algorithm##############
##############################################################
print("\n Step by step algorithm \n")

np.random.seed(12121334)

fig, ax = plt.subplots(1, figsize=(8,6), dpi=200)

[mu, sigma] = LognormalParameters(mean = 10., CoV = 0.1) #(mean = 3.5, CoV = 0.8)
n = 1000
LNSample = np.random.lognormal(mean = mu, sigma = sigma, size = (n,))

Accuracy1 = 1 #0.5

CompressedSample = PWLcompressor(LNSample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = True, Accuracy = Accuracy1)
Step1 = CompressedSample.plot('g', ShowPlot = False)

plt.plot(Step1['SampleX'],Step1['SampleY'],color='black')
plt.plot(Step1['PWLX'],Step1['PWLY'], linewidth=2.0, linestyle = '-', marker = 'o', color = 'r')
plt.xlabel('$G\sim PWL(\mathbf{x},\mathbf{y}=(0,1))$')
plt.savefig('Step1Algorithm.png')
plt.close()

fig, ax = plt.subplots(1, figsize=(8,6), dpi=200)

Accuracy2 = 0.44

CompressedSample = PWLcompressor(LNSample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = True, Accuracy = Accuracy2)
Step2 = CompressedSample.plot('g', ShowPlot = False)

plt.plot(Step2['SampleX'],Step2['SampleY'],color='black')
plt.plot(Step2['PWLX'],Step2['PWLY'], linewidth=2.0, linestyle = '-', marker = 'o', color = 'r')
plt.xlabel('$G\sim PWL(\mathbf{x},\mathbf{y})=(0,0.916,1))$')
plt.savefig('Step2Algorithm.png')
plt.close()

fig, ax = plt.subplots(1, figsize=(8,6), dpi=200)

Accuracy3 = 0.3885

CompressedSample = PWLcompressor(LNSample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = True, Accuracy = Accuracy3)
Step3 = CompressedSample.plot('g', ShowPlot = False)

plt.plot(Step3['SampleX'],Step3['SampleY'],color='black')
plt.plot(Step3['PWLX'],Step3['PWLY'], linewidth=2.0, linestyle = '-', marker = 'o', color = 'r')
plt.xlabel('$G\sim PWL(\mathbf{x},\mathbf{y}=0,0.109,0.916,1))$')
plt.savefig('Step3Algorithm.png')
plt.close()

fig, ax = plt.subplots(1, figsize=(8,6), dpi=200)

Accuracy4 = 0.1

CompressedSample = PWLcompressor(LNSample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = True, Accuracy = Accuracy4)
Step4 = CompressedSample.plot('g', ShowPlot = False)

plt.plot(Step4['SampleX'],Step4['SampleY'],color='black')
plt.plot(Step4['PWLX'],Step4['PWLY'], linewidth=2.0, linestyle = '-', marker = 'o', color = 'r')
#plt.xlabel('G\sim PWL(\mathbf{x},\mathbf{y})=(0,0.008,0.109,0.195,0.846,0.916,0.985,0.998,1)')
plt.savefig('Step4Algorithm.png')
plt.close()

del fig, ax, mu, sigma, n, Accuracy1, Accuracy2, Accuracy3, Accuracy4, Step1, Step2, Step3, Step4, LNSample, CompressedSample


##############################################################
######################### Different distributions#############
##############################################################

n = 1000
Accuracy = 0.05


MEAN = 6.04405167667
VAR = 44.4314748556

np.random.seed(12121334)

#######

fig, ax = plt.subplots(1, figsize=(8,6), dpi=200)

FSSample = XLsimulations(SampleSize = n, PoissonLambda = 2, ParetoX0 = 10, ParetoAlpha = 2.5, Deductible = 12, Limit = 10, AggregateLimit = 30)

CompressedSample1 = PWLcompressor(FSSample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False,
                                 Accuracy = Accuracy, RelativeAtomDetectionThreshold = 0.001)
Dist1 = CompressedSample1.plot('g', ShowPlot = False)

plt.plot(Dist1['SampleX'],Dist1['SampleY'],color='black')
plt.plot(Dist1['PWLX'],Dist1['PWLY'], linewidth=1.0, linestyle = '-', marker = 'o', color = 'r',markersize=3)
plt.xlabel('$F\sim$ XL contract $(N\sim$ Poisson $(\lambda = 2),$ $Y\sim$ Pareto $(\\alpha=2.5,\\theta=10),d=12,u_Y=10,u_X=30);\quad \epsilon=0.05$')
plt.xlim([-1,25])

#plt.show()
plt.savefig('FullAlgorithmExampleXLContract.png')
plt.close()

####Negative Binomial

p=MEAN/VAR
r=MEAN**2/(VAR-MEAN)

fig, ax = plt.subplots(1, figsize=(8,6), dpi=200)

NegBinSample = np.random.negative_binomial(r, p, n)

CompressedSample2 = PWLcompressor(NegBinSample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False,
                                 Accuracy = Accuracy, RelativeAtomDetectionThreshold = 0.001)
Dist2 = CompressedSample2.plot('g', ShowPlot = False)

plt.plot(Dist2['SampleX'],Dist2['SampleY'],color='black')
plt.plot(Dist2['PWLX'],Dist2['PWLY'], linewidth=1.0, linestyle = '-', marker = 'o', color = 'r',markersize=3)
plt.xlabel('$F\sim$ Negative Binomial $(\\beta = 6.3513,r=0.9516);\quad \epsilon=0.05$')
plt.xlim([-1,35])

plt.savefig('FullAlgorithmExampleNegBIN.png')
plt.close()

##Normal

NormalSample = np.random.normal(MEAN, np.sqrt(VAR), n)

fig, ax = plt.subplots(1, figsize=(8,6), dpi=200)

CompressedSample3 = PWLcompressor(NormalSample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False,
                                 Accuracy = Accuracy, RelativeAtomDetectionThreshold = 0.001)
Dist3 = CompressedSample3.plot('g', ShowPlot = False)

plt.plot(Dist3['SampleX'],Dist3['SampleY'],color='black')
plt.plot(Dist3['PWLX'],Dist3['PWLY'], linewidth=1.0, linestyle = '-', marker = 'o', color = 'r',markersize=3)
plt.xlabel('$F\sim$ Normal $(\mu=6.0441,\sigma=6.6657);\quad \epsilon=0.05$')


plt.savefig('FullAlgorithmExampleNormal.png')
plt.close()

##Lognormal

[mu, sigma] = LognormalParameters(mean = MEAN, CoV = np.sqrt(VAR)/MEAN)

fig, ax = plt.subplots(1, figsize=(8,6), dpi=200)

LognormalSample = np.random.lognormal(mean = mu, sigma = sigma, size = n)

CompressedSample4 = PWLcompressor(LognormalSample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False,
                                 Accuracy = Accuracy, RelativeAtomDetectionThreshold = 0.001)
Dist4 = CompressedSample4.plot('g', ShowPlot = False)

plt.plot(Dist4['SampleX'],Dist4['SampleY'],color='black')
plt.plot(Dist4['PWLX'],Dist4['PWLY'], linewidth=1.0, linestyle = '-', marker = 'o', color = 'r',markersize=3)
plt.xlabel('$F\sim$ Lognormal $(\mu=1.4012,\sigma=0.8921);\quad \epsilon=0.05$')

#plt.show()
plt.savefig('FullAlgorithmExampleLN.png')
plt.close()

del fig, ax, mu, MEAN, VAR, sigma, n, Accuracy, Dist1, Dist2, Dist3, Dist4, FSSample, CompressedSample1, CompressedSample2, CompressedSample3, CompressedSample4