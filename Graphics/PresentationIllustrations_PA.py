

import numpy as np
import matplotlib.pyplot as plt
from pwl_compressor_core.compressor import PWLcompressor
from pwl_compressor_core.pwl_distribution import PiecewiseLinearDistribution, EmpiricalDistributionFromSample
#from Graphics.Illustrations import PlotBrace, LognormalParameters, XLsimulations
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from Graphics.GenericIllustrationMethods import MyMark_inset, Characteristics, BoundPlot, SetAxisBoundLineWidth, PlotBrace, LognormalParameters, XLsimulations

##############################################################
######################### PA: Step by step algorithm ##############
##############################################################
print("\n ================================")
print("\n =========== Example ============")
print("\n AlgorithmSteps.pdf ")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(8,3.5), dpi=200) # for text without two column setting
fig.subplots_adjust(wspace=0.1)
# fig.tight_layout()

np.random.seed(1232121334)
[mu, sigma] = LognormalParameters(mean = 10., CoV = 0.1) #(mean = 3.5, CoV = 0.8)
n = 1000
LNSample = np.random.lognormal(mean = mu, sigma = sigma, size = (n,))

Accuracies = [1,0.4,0.3,0.03]

CompressedSample = PWLcompressor(LNSample, Accuracy = Accuracies[0])
Step1 = CompressedSample.plot('g', ShowPlot = False)
CompressedSample = PWLcompressor(LNSample, Accuracy = Accuracies[1])
Step2 = CompressedSample.plot('g', ShowPlot = False)
CompressedSample = PWLcompressor(LNSample, Accuracy = Accuracies[2])
Step3 = CompressedSample.plot('g', ShowPlot = False)
CompressedSample = PWLcompressor(LNSample, Accuracy = Accuracies[3], EnforcedInterpolationQuantiles = [0.085, 0.922, 0.41, 0.75])
Step4 = CompressedSample.plot('g', ShowPlot = False)


print(Step3['PWLY'])

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim(7, 14.2)
    ax.set_ylim(-0.03, 1.03)
    ax.xaxis.set_ticks([ 8, 10, 12, 14])
    # SetAxisBoundLineWidth(axIns)
    # axIns.axhline(linewidth=40, color="g")

ax1.plot(Step1['PWLX'],Step1['PWLY'], linewidth=2.7, linestyle = '-', marker = 'o', color = 'r')
ax2.plot(Step2['PWLX'],Step2['PWLY'], linewidth=2.7, linestyle = '-', marker = 'o', color = 'r')
ax3.plot(Step3['PWLX'],Step3['PWLY'], linewidth=2.7, linestyle = '-', marker = 'o', color = 'r')
ax4.plot(Step4['PWLX'],Step4['PWLY'], linewidth=2.7, linestyle = '-', marker = 'o', color = 'r')

for ax in [ax1, ax2, ax3, ax4]:
    ax.plot(Step1['SampleX'],Step1['SampleY'], linewidth=1.0, linestyle = '-', marker = '', color = 'k')

ax1.set_title('Step 1')
ax2.set_title('Step 2')
ax3.set_title('Step 3')
ax4.set_title('Step 8')

# ax.xlabel('$G\sim PWL(\mathbf{x},\mathbf{y}=(0,1))$')
plt.savefig('AlgorithmSteps.png', bbox_inches = "tight")
plt.close()

del ax1, ax2, ax3, ax4, fig, mu, LNSample, Step1, Step2, Step3, Step4



##############################################################
######################### PA: Lognormal two steps ##############
##############################################################
print("\n ================================")
print("\n =========== Example ============")
print("\n LognormalTwoSteps.pdf ")



fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(9,3.5), dpi=200) # for text without two column setting
fig.subplots_adjust(wspace=0.1)
# fig.tight_layout()

np.random.seed(1232121334)
[mu, sigma] = LognormalParameters(mean = 10., CoV = 0.1) #(mean = 3.5, CoV = 0.8)
n = 1000000
LNSample = np.random.lognormal(mean = mu, sigma = sigma, size = (n,))

Accuracies = [0.1,0.005]

CompressedSample = PWLcompressor(LNSample, Accuracy = Accuracies[0])
Step1 = CompressedSample.plot('g', ShowPlot = False)
CompressedSample = PWLcompressor(LNSample, Accuracy = Accuracies[1])
Step2 = CompressedSample.plot('g', ShowPlot = False)
#CompressedSample = PWLcompressor(LNSample, Accuracy = Accuracies[3], EnforcedInterpolationQuantiles = [0.085, 0.922, 0.41, 0.75])
#Step4 = CompressedSample.plot('g', ShowPlot = False)


for ax in [ax1, ax2]:
    ax.set_xlim(6, 15.2)
    ax.set_ylim(-0.03, 1.03)
    ax.xaxis.set_ticks([ 8, 10, 12, 14])
    # SetAxisBoundLineWidth(axIns)
    # axIns.axhline(linewidth=40, color="g")

ax1.plot(Step1['PWLX'],Step1['PWLY'], linewidth=2.7, linestyle = '-', marker = '', color = 'r', label='piecewise linear')
ax2.plot(Step2['PWLX'],Step2['PWLY'], linewidth=2.7, linestyle = '-', marker = '', color = 'r', label='piecewise linear')

for ax in [ax1, ax2]:
    ax.plot(Step1['SampleX'],Step1['SampleY'], linewidth=1.5, linestyle = '-', marker = '', color = 'k', label='sample')

ax1.set_title( r'$\epsilon $ = 10%     ('+str(len(Step1['PWLX']))+' points)')
ax2.set_title(r'$\epsilon $ = 0.5%    ('+str(len(Step2['PWLX']))+' points)')

legend1 = ax1.legend(loc='lower right', shadow=False, prop={'size':12})
legend2 = ax2.legend(loc='lower right', shadow=False, prop={'size':12})

# ax.xlabel('$G\sim PWL(\mathbf{x},\mathbf{y}=(0,1))$')
plt.savefig('LognormalTwoSteps.png', bbox_inches = "tight", transparent=True)
plt.close()

del ax1, ax2, fig, mu, LNSample, Step1, Step2, Accuracies, n, sigma, CompressedSample


##########################################################################################
######################### PA: different distribution with same mean and std ##############
##########################################################################################
print("\n ================================")
print("\n =========== Example ============")
print("\n SameMeanDistributions.pdf ")


#target moments
TargetMean = 2
TargetStd  = 1

DistList = [0,0,0,0]

#dist 1: discrete
DistList[0] = PiecewiseLinearDistribution([0,0,1,1,2,2],[0,0.2,0.2,0.7,0.7,1])

#dist 2: normal
np.random.seed(12121334)
normdist = PWLcompressor(np.random.randn(10000))
DistList[1] = PiecewiseLinearDistribution(normdist.Result['PWLX'],normdist.Result['PWLY'])

#dist 3: pareto
ParetoAlpha = 2.5
ygrid = np.linspace(0.001,0.999,300)
DistList[2] = PiecewiseLinearDistribution((1-ygrid)**(-1/ParetoAlpha),ygrid)

#dist 3: XL
Alpha = 0.5
XLgridX = np.linspace(0,1,35)
XLx = np.concatenate([[0,0],XLgridX,[1,1],1+XLgridX,[2,2]])
atom=0.1
inc1 = 0.35
inc2 = 0.2
XLy = np.concatenate([[0,atom],
                      atom+inc1*XLgridX**(Alpha),
                      [atom+inc1,1-atom-inc2],
                      1-atom-inc2+inc2*XLgridX**(Alpha),
                      [1-atom,1]])

DistList[3] = PiecewiseLinearDistribution(XLx,XLy)


#make all moments equal
def MoveToTargetMoments(dist):
    dist = dist.shift(-dist.expected())
    dist = dist.scale(1/dist.stdev())
    dist = dist.scale(TargetStd)
    dist = dist.shift(TargetMean)
    return dist

DistList = [MoveToTargetMoments(dist) for dist in DistList]

for dist in DistList:
    print([dist.expected(),dist.stdev()])

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(8,2), dpi=300) # for text without two column setting
Axes = (ax1, ax2, ax3, ax4)
ColorList = ['k','g','b','r']

for ax, dist, color in zip(Axes, DistList, ColorList):
    ax.set_xlim(0,4)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_ticks([ 1,2,3])
    ax.plot(dist.Xvalues,dist.Fvalues, linewidth=1.3, linestyle = '-', color = color, fillstyle = 'none')


ax1.set_title('Discrete')
ax2.set_title('Normal')
ax3.set_title('Pareto')
ax4.set_title('Excess-of-Loss')

#fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.09, hspace=None)
#fig.subplots_adjust(wspace=0.09, top= -0.01)

#fig.tight_layout()
fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=11.0)

plt.savefig('SameMeanDistributions.png', transparent=True)
#plt.show()
plt.close()

del ax1, ax2, ax3, ax4, fig, ParetoAlpha, ygrid, DistList, Alpha, XLgridX, XLx, atom, inc1, inc2, XLy

####################################################################################
######################### FullAlgorithmExampleFreqSevNOPWL #########################
####################################################################################
print("\n ================================")
print("\n =========== Example ============ ")
print("\n FullAlgorithmExampleFreqSevNOPWL.png ")


np.random.seed(1232121334)

n = 1000
#parameters: see xAct pricing structure with xActID 6338199
FSSample = XLsimulations(SampleSize = n, PoissonLambda = 2, ParetoX0 = 10, ParetoAlpha = 2.5, Deductible = 12, Limit = 10, AggregateLimit = 30)
Accuracy = 0.001

CompressedSample = PWLcompressor(FSSample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False,
                                 Accuracy = Accuracy, RelativeAtomDetectionThreshold = 0.001)
Step3 = CompressedSample.plot('g', ShowPlot = False)


fig, ax = plt.subplots(1, figsize=(8,3), dpi=200) #(6,3)
zoom = 16

ax.set_xlim([-1,31])
ax.set_ylim([-0.02,1.02])

# ax.plot(Step3['SampleX'],Step3['SampleY'],color='black', linewidth=0.8)
# ax.plot(Step3['PWLX'],Step3['PWLY'], linewidth=1.3, linestyle = '--', marker = 'o', markersize = 5, color = 'k', fillstyle = 'none')

axIns = zoomed_inset_axes(ax, zoom, loc=4) # zoom = 6

axIns.set_xlim(11.25, 12.45)
axIns.set_ylim(0.810, 0.848)
axIns.set_xticklabels([])
axIns.set_xticks([])
axIns.set_yticklabels([])
axIns.set_yticks([])
SetAxisBoundLineWidth(axIns)

for thisAX in [ax,axIns]:
    thisAX.plot(Step3['SampleX'],Step3['SampleY'],color='black', linewidth=0.8)

MyMark_inset(ax, axIns, loc11=1, loc12=1, loc21=2, loc22=3, fc="none", ec="0.1", zorder = -10, linewidth=0.4)

# plt.show()
plt.savefig('FullAlgorithmExampleFreqSevNOPWL.png', transparent=True)

# for thisAX in [ax,axIns]:
#     thisAX.plot(Step3['PWLX'],Step3['PWLY'], linewidth=1.3, linestyle = '--', marker = 'o', markersize = 5, color = 'k', fillstyle = 'none')
#
# plt.savefig('FullAlgorithmExampleFreqSev.pdf')
# plt.close()


del fig, ax, axIns, FSSample, Accuracy, zoom, CompressedSample


####################################################################################
######################### FullAlgorithmExampleFreqSevNOPWL #########################
####################################################################################
print("\n ================================")
print("\n =========== Example ============ ")
print("\n XLStwoStepExample.png ")


np.random.seed(1232121334)

n = 100000
#parameters: see xAct pricing structure with xActID 6338199
XLSample = XLsimulations(SampleSize = n, PoissonLambda = 2, ParetoX0 = 10, ParetoAlpha = 2.5, Deductible = 12, Limit = 10, AggregateLimit = 30)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(9,3.5), dpi=200) # for text without two column setting
fig.subplots_adjust(wspace=0.1)
# fig.tight_layout()

Accuracies = [0.1,0.005]

CompressedSample = PWLcompressor(XLSample, Accuracy = Accuracies[0])
Step1 = CompressedSample.plot('g', ShowPlot = False)
CompressedSample = PWLcompressor(XLSample, Accuracy = Accuracies[1])
Step2 = CompressedSample.plot('g', ShowPlot = False)
#CompressedSample = PWLcompressor(LNSample, Accuracy = Accuracies[3], EnforcedInterpolationQuantiles = [0.085, 0.922, 0.41, 0.75])
#Step4 = CompressedSample.plot('g', ShowPlot = False)


for ax in [ax1, ax2]:
    ax.set_xlim(-1, 31)
    ax.set_ylim(-0.03, 1.03)
    #ax.xaxis.set_ticks([ 8, 10, 12, 14])
    # SetAxisBoundLineWidth(axIns)
    # axIns.axhline(linewidth=40, color="g")

ax1.plot(Step1['PWLX'],Step1['PWLY'], linewidth=2.7, linestyle = '-', marker = '', color = 'r', label='piecewise linear')
ax2.plot(Step2['PWLX'],Step2['PWLY'], linewidth=2.7, linestyle = '-', marker = '', color = 'r', label='piecewise linear')

for ax in [ax1, ax2]:
    ax.plot(Step1['SampleX'],Step1['SampleY'], linewidth=1.5, linestyle = '-', marker = '', color = 'k', label='sample')

ax1.set_title( r'$\epsilon = 10\%$     ('+str(len(Step1['PWLX']))+' points)')
ax2.set_title(r'$\epsilon = 0.5\%$     ('+str(len(Step2['PWLX']))+' points)')

legend1 = ax1.legend(loc='lower right', shadow=False)
legend2 = ax2.legend(loc='lower right', shadow=False)

# ax.xlabel('$G\sim PWL(\mathbf{x},\mathbf{y}=(0,1))$')
plt.savefig('XLStwoStepExample.png', bbox_inches = "tight", transparent=True)
plt.close()

del ax1, ax2, fig, XLSample, Step1, Step2, CompressedSample, Accuracies



##############################################################
######################### xTVaR exponential illustration########################
##############################################################
print("\n =========== xTVaR exponential illustration ============ \n")

fig, ax = plt.subplots(1, figsize=(9,3), dpi=300) #(6,3)

MinX = -0.1
MaxX = 30.6
#  K =4, x =(1, 4, 4,9), y = (0, 0.6,0.8,1),

theta = 10
alpha = 0.8

x = np.linspace(0,MaxX,10000)
y = 1-np.exp(-x/theta)

ax.plot(x,y)

VaR = - theta * np.log(1-alpha)
Mean = theta
TVaR = VaR+Mean
BarHeight = alpha

ax.plot([TVaR,TVaR],[0,.2],        linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([TVaR,TVaR],[.4,.926],        linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([VaR,VaR],  [0,.2],   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([VaR,VaR],  [.4,BarHeight],   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([Mean,Mean],[0,.2],   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([Mean,Mean],[.4,BarHeight],   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.text(TVaR, 0.22, '$\mathbb{E}[X|X > VaR_{80\%}(X)]$', verticalalignment='bottom', horizontalalignment='center', fontsize=16)
ax.text(Mean, 0.22, '$\mathbb{E}[X]$', verticalalignment='bottom', horizontalalignment='center', fontsize=16)
ax.text(VaR,  0.22, '$VaR_{80\%}(X)$' , verticalalignment='bottom', horizontalalignment='center', fontsize=16)
# ax.text(5, 0.9, '$(1-0.8)TVaR_{0.8}^{\Delta}[X]$' , verticalalignment='bottom', horizontalalignment='center', fontsize=14)
x1=np.ma.masked_less(x, Mean)
ax.fill_between(x,y1 = y, y2=1, where= x>=VaR, facecolor='k', alpha=0.5, linewidth = 0, color = 'k')
ax.fill_between(x1,y1 = 0.8, y2=1, where= x1<=VaR, facecolor='k', alpha=0.5, linewidth = 0, color = 'k')
# ax.arrow(10, 0.95, -1.45, 0, head_width=0.05, head_length=0.1, fc='k', ec='k')

PlotBrace(ax, x1 = Mean, x2 = TVaR, y = 1.07, braceheight = 0.03, bracewidth = 0.5, text = '$TVaR_{80\%}^{\Delta}[X]$', upper=True, fontsize=16)

ax.yaxis.set_ticks([ 0., 0.2, 0.4, 0.6, 0.8, 1.0])

ax.set_xlim([MinX,MaxX])
ax.set_ylim([-0.05,1.31])

plt.savefig('xTVaRExpillustration_PA.png', bbox_inches = "tight", transparent=True)
plt.close()

del fig, ax, alpha, theta, x, y, VaR, Mean, TVaR, BarHeight, x1


##############################################################
######################### Fifth plot #########################
##############################################################
print("\n PWLillustrationForPres.png ")



######### 'PWLillustration.pdf'

fig, ax = plt.subplots(1, figsize=(8,3), dpi=200) #(6,3)

MinX = -0.7 # -0.2
MaxX = 12.5 # 10.2
#  K =4, x =(1, 4, 4,9), y = (0, 0.6,0.8,1),
MyMarkerSize = 9
ax.plot([MinX,1,4],[0,0,0.6],   linewidth=1.0, linestyle = '-', marker = '', color = 'k')
ax.plot([4,9,MaxX],[0.8,1,1],   linewidth=1.0, linestyle = '-', marker = '',color = 'k')
ax.plot([1,4,4,9],[0,0.6,0.8,1],   linewidth=1.0, linestyle = '', marker = '.', markersize=MyMarkerSize, color = 'k')
ax.plot([4],[0.6],   linewidth=1.0, linestyle = '', marker = '.', markersize=MyMarkerSize*0.5, color = 'w')
# ax.scatter([4],[0.6],   color = 'k', facecolors='r', alpha = 1)
# ax.plot([4],[0.6],   linewidth=1.0, linestyle = '', marker = 'o', color = 'k', facecolors='none')
# PlotBrace(ax, x1 = m1, x2 = m1+delta1, y = 0.64, braceheight = 0.03, bracewidth = 0.5, text = '$\delta_1$')

# ax.yaxis.set_ticks([ 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xlim([MinX,MaxX])
ax.set_ylim([-0.05,1.05])

ax.text(  1, 0.0, '$(x_1,y_1)$ ', verticalalignment='bottom', horizontalalignment='right', fontsize=14)
ax.text(  4, 0.6, ' $(x_2,y_2)$', verticalalignment='top', horizontalalignment='left', fontsize=14)
ax.text(  4, 0.8, '$(x_3,y_3)$ ', verticalalignment='bottom', horizontalalignment='right', fontsize=14)
ax.text(  9, 1.0, ' $(x_4,y_4)$', verticalalignment='top', horizontalalignment='left', fontsize=14)


# ax.set_xlabel('$x$', fontsize=9)
ax.set_ylabel('$G(t)$', fontsize=14) #('$G(x)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)
# ax.set_ylabel('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=11) #('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)

plt.savefig('PWLillustrationForPres.png', bbox_inches = "tight", transparent=True)
# plt.show()
plt.close()

del fig, ax, MyMarkerSize


##############################################################
######################### Fifth plot #########################
##############################################################
print("\n PWLillustrationForPresNoAtom.png ")



######### 'PWLillustration.pdf'

fig, ax = plt.subplots(1, figsize=(8,3), dpi=300) #(6,3)

MinX = -0.7 # -0.2
MaxX = 12.5 # 10.2
#  K =4, x =(1, 4, 4,9), y = (0, 0.6,0.8,1),
MyMarkerSize = 9
pwlX = [1, 4, 6, 9]
pwlY = [0,0.7,0.9,1]
ax.plot(pwlX,pwlY,   linewidth=1.0, linestyle = '-', marker = '.', markersize=MyMarkerSize, color = 'k')


ax.plot([MinX,pwlX[0]],[0,0],   linewidth=1.0, linestyle = '-', marker = '', color = 'k')
ax.plot([pwlX[-1],MaxX],[1,1],   linewidth=1.0, linestyle = '-', marker = '',color = 'k')
# ax.plot([1,4,4,9],[0,0.6,0.8,1],   linewidth=1.0, linestyle = '', marker = '.', markersize=MyMarkerSize, color = 'k')
# ax.plot([4],[0.6],   linewidth=1.0, linestyle = '', marker = '.', markersize=MyMarkerSize*0.5, color = 'w')
# ax.scatter([4],[0.6],   color = 'k', facecolors='r', alpha = 1)
# ax.plot([4],[0.6],   linewidth=1.0, linestyle = '', marker = 'o', color = 'k', facecolors='none')
# PlotBrace(ax, x1 = m1, x2 = m1+delta1, y = 0.64, braceheight = 0.03, bracewidth = 0.5, text = '$\delta_1$')

# ax.yaxis.set_ticks([ 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xlim([MinX,MaxX])
ax.set_ylim([-0.02,1.02])

ax.text(  pwlX[0], pwlY[0], '$(x_1,y_1)$ ', verticalalignment='bottom', horizontalalignment='right', fontsize=14)
ax.text(  pwlX[1], pwlY[1], ' $(x_2,y_2)$', verticalalignment='top', horizontalalignment='left', fontsize=14)
ax.text(  pwlX[2], pwlY[2], '$(x_3,y_3)$ ', verticalalignment='top', horizontalalignment='left', fontsize=14)
ax.text(  pwlX[3], pwlY[3], ' $(x_4,y_4)$', verticalalignment='top', horizontalalignment='left', fontsize=14)


# ax.set_xlabel('$x$', fontsize=9)
ax.set_ylabel('$G(t)$', fontsize=14) #('$G(x)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)
# ax.set_ylabel('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=11) #('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)

plt.savefig('PWLillustrationForPresNoAtom.png', bbox_inches = "tight", transparent=True)
# plt.show()
plt.close()

del fig, ax, MyMarkerSize


##############################################################
######################### Store Full Sample ##################
##############################################################
print("\n StoreFullSample.png ")


Sample = np.asarray([ 1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]) #sorted!
SampleLength = len(Sample)
SampleDist = EmpiricalDistributionFromSample(Sample)

#from Illustrations import BoundPlot


fig, ax = plt.subplots(1, figsize=(7,3), dpi=300)
CompressedSample = PWLcompressor(Sample, Accuracy = 1, EnforcedInterpolationQuantiles = [])
C1 = CompressedSample.plot('g', ShowPlot = False)

ax.set_xlim([0,20])
ax.set_ylim([-0.02,1.02])
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

ax.plot(C1['SampleX'],C1['SampleY'],color='black', linewidth=3, label='Sample distribution (Sample size = 10)')
ax.plot(C1['SampleX'],C1['SampleY'],linewidth=2, linestyle = '--', marker = 'o', markersize = 7, color = 'r', label='Sample distribution (Sample size = 10)')
# ax.plot(C3['PWLX'],C3['PWLY'], linewidth=1.1, linestyle = '-', marker = 'o', markersize = 3, color = 'g', fillstyle = 'none')
# legend = ax.legend(loc='lower right', shadow=True)

# plt.show()
plt.savefig('StoreFullSample.png', bbox_inches='tight', transparent=True)
plt.close()

del fig, ax, Sample, C1, CompressedSample


##############################################################
######################### Store Fixed Quantiles ##############
##############################################################
print("\n StoreFixedQuantiles.png ")


Sample = np.asarray([ 1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]) #sorted!
SampleLength = len(Sample)
SampleDist = EmpiricalDistributionFromSample(Sample)

#from Illustrations import BoundPlot


fig, ax = plt.subplots(1, figsize=(7,3), dpi=300)
CompressedSample = PWLcompressor(Sample, Accuracy = 1, EnforcedInterpolationQuantiles = [])
C1 = CompressedSample.plot('g', ShowPlot = False)

ax.set_xlim([0,20])
ax.set_ylim([-0.05,1.05])
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

ind = [x for x in range(-1,20,4)]
ind[0]=0

ax.plot(C1['SampleX'],C1['SampleY'],color='black', linewidth=3, label='Sample distribution (Sample size = 10)')
ax.plot(C1['SampleX'][ind],C1['SampleY'][ind],linewidth=2, linestyle = '--', marker = 'o', markersize = 7, color = 'r', label='Sample distribution (Sample size = 10)')
# ax.plot(C3['PWLX'],C3['PWLY'], linewidth=1.1, linestyle = '-', marker = 'o', markersize = 3, color = 'g', fillstyle = 'none')
# legend = ax.legend(loc='lower right', shadow=True)

# plt.show()
plt.savefig('StoreFixedQuantiles.png', bbox_inches='tight', transparent=True)
plt.close()

del fig, ax, Sample, C1, CompressedSample