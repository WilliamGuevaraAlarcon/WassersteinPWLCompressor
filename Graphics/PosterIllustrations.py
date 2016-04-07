__author__ = 'William Guevara'
import numpy as np
import matplotlib.pyplot as plt
from pwl_compressor_core.compressor import PWLcompressor

from Graphics.GenericIllustrationMethods import MyMark_inset, Characteristics, BoundPlot, SetAxisBoundLineWidth, PlotBrace, LognormalParameters, XLsimulations
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

##############################################################
######################### First plot #########################
##############################################################


print("\n =========== Figure 1 ============ \n")
print("\n FullAlgorithmExampleFreqSevTwoPWL.pdf ")


np.random.seed(1232121334)

n = 1000
#parameters: see xAct pricing structure with xActID 6338199
FSSample = XLsimulations(SampleSize = n, PoissonLambda = 2, ParetoX0 = 10, ParetoAlpha = 2.5, Deductible = 12, Limit = 10, AggregateLimit = 30)
Accuracy = 0.005
Accuracy2 = 0.7

CompressedSample = PWLcompressor(FSSample, RemoveNegativeJumps = True, MakePWLsmoother = False, CheckStrictAdmissibility = True,
                                 Accuracy = Accuracy, RelativeAtomDetectionThreshold = 0.001)
CompressedSample2 = PWLcompressor(FSSample, RemoveNegativeJumps = True, MakePWLsmoother = False, CheckStrictAdmissibility = True,
                                 Accuracy = Accuracy2, RelativeAtomDetectionThreshold = 0.001)
Step3 = CompressedSample.plot('g', ShowPlot = False)
Step4 = CompressedSample2.plot('g', ShowPlot = False)


fig, ax = plt.subplots(1, figsize=(7,3), dpi=200) #(6,3)
zoom = 16

ax.set_xlim([-1,31])
ax.set_ylim([-0.02,1.02])

axIns = zoomed_inset_axes(ax, zoom, loc=4) # zoom = 6

axIns.set_xlim(11.25, 12.45)
axIns.set_ylim(0.810, 0.848)
axIns.set_xticklabels([])
axIns.set_xticks([])
axIns.set_yticklabels([])
axIns.set_yticks([])
SetAxisBoundLineWidth(axIns)

for thisAX in [ax,axIns]:
    thisAX.plot(Step3['SampleX'],Step3['SampleY'],color='black', linewidth=1.0, label='Empirical distribution')

MyMark_inset(ax, axIns, loc11=1, loc12=1, loc21=2, loc22=3, fc="none", ec="0.4", zorder = -10, linewidth=0.4)

# plt.show()
#plt.savefig('FullAlgorithmExampleFreqSevNOPWL.png')

for thisAX in [ax,axIns]:
    thisAX.plot(Step3['PWLX'],Step3['PWLY'], linewidth=1.0, linestyle = '--', marker = '.', markersize = 5, color = 'blue', fillstyle = 'none', label='Some PWL distribution')
    thisAX.plot(Step4['PWLX'],Step4['PWLY'], linewidth=1.0, linestyle = '--', marker = '.', markersize = 5, color = 'red', fillstyle = 'none', label='Another PWL distribution')

legend = ax.legend(loc=0, fontsize = 8)

plt.savefig('FullAlgorithmExampleFreqSevTwoPWL.pdf', transparent=True)
plt.close()


del fig, ax, axIns, FSSample, Accuracy, zoom, CompressedSample

##############################################################
######################### Second plot #########################
##############################################################
print("\n =========== Figure 2 ============ \n")
print("\n PWLillustrationTwoParametrizations.pdf ")

######### 'PWLillustration.pdf'

fig, ax = plt.subplots(1, figsize=(8,3), dpi=300) #(6,3)

MinX = -1 # -0.2
MaxX = 12.5 # 10.2
#  K =4, x =(1, 4, 4,9), y = (0, 0.6,0.8,1),
MyMarkerSize = 9
pwlX = [1, 4, 6, 9]
pwlY = [0,0.7,0.9,1]
ax.plot(pwlX,pwlY,   linewidth=1.0, linestyle = '-', marker = '.', markersize=MyMarkerSize, color = 'k')


ax.plot([MinX,MaxX],[0,0],   linewidth=1.0, linestyle = '-', marker = '', color = 'k')
ax.plot([pwlX[-1],MaxX],[1,1],   linewidth=1.0, linestyle = '-', marker = '',color = 'k')
ax.plot([0,0],[0,1],   linewidth=1.0, linestyle = '-', marker = '',color = 'k')

ax.plot(0,pwlY[0], marker = '_',color = 'k')
ax.plot(0,pwlY[1], marker = '_',color = 'k')
ax.plot(0,pwlY[2], marker = '_',color = 'k')
ax.plot(0,pwlY[3], marker = '_',color = 'k')


# ax.plot([1,4,4,9],[0,0.6,0.8,1],   linewidth=1.0, linestyle = '', marker = '.', markersize=MyMarkerSize, color = 'k')
# ax.plot([4],[0.6],   linewidth=1.0, linestyle = '', marker = '.', markersize=MyMarkerSize*0.5, color = 'w')
# ax.scatter([4],[0.6],   color = 'k', facecolors='r', alpha = 1)
# ax.plot([4],[0.6],   linewidth=1.0, linestyle = '', marker = 'o', color = 'k', facecolors='none')
# PlotBrace(ax, x1 = m1, x2 = m1+delta1, y = 0.64, braceheight = 0.03, bracewidth = 0.5, text = '$\delta_1$')

# ax.yaxis.set_ticks([ 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xlim([MinX,MaxX])
ax.set_ylim([-0.2,1.05])

ax.text(  pwlX[0], pwlY[0], '$(x_1,y_1)$', verticalalignment='bottom', horizontalalignment='center', fontsize=14, color="blue")
ax.text(  pwlX[1], pwlY[1], '$(x_2,y_2)$', verticalalignment='top', horizontalalignment='left', fontsize=14, color="blue")
ax.text(  pwlX[2], pwlY[2], '$(x_3,y_3)$', verticalalignment='top', horizontalalignment='left', fontsize=14, color="blue")
ax.text(  pwlX[3], pwlY[3], '$(x_4,y_4)$', verticalalignment='top', horizontalalignment='left', fontsize=14, color="blue")

ax.text(  (pwlX[0]+pwlX[1])/2, 0, '$\mu_1$', verticalalignment='bottom', horizontalalignment='center', fontsize=14, color="red")
ax.text(  (pwlX[1]+pwlX[2])/2, 0, '$\mu_2$', verticalalignment='bottom', horizontalalignment='center', fontsize=14, color="red")
ax.text(  (pwlX[2]+pwlX[3])/2, 0, '$\mu_3$', verticalalignment='bottom', horizontalalignment='center', fontsize=14, color="red")

ax.text(  -0.5, pwlY[0], '$z_1$', verticalalignment='bottom', horizontalalignment='left', fontsize=14, color="red")
ax.text(  -0.5, pwlY[1], '$z_2$', verticalalignment='center', horizontalalignment='left', fontsize=14, color="red")
ax.text(  -0.5, pwlY[2], '$z_3$', verticalalignment='center', horizontalalignment='left', fontsize=14, color="red")
ax.text(  -0.5, pwlY[3], '$z_4$', verticalalignment='center', horizontalalignment='left', fontsize=14, color="red")

PlotBrace(ax, x1 = pwlX[0], x2 = pwlX[1], y = -0.05, braceheight = 0.03, bracewidth = 0.5, text = '$2\delta_1$', upper=False, fontsize=14, color="red")
PlotBrace(ax, x1 = pwlX[1], x2 = pwlX[2], y = -0.05, braceheight = 0.03, bracewidth = 0.5, text = '$2\delta_2$', upper=False, fontsize=14, color="red")
PlotBrace(ax, x1 = pwlX[2], x2 = pwlX[3], y = -0.05, braceheight = 0.03, bracewidth = 0.5, text = '$2\delta_3$', upper=False, fontsize=14, color="red")

# ax.set_xlabel('$x$', fontsize=9)
ax.set_ylabel('$G(t)$', fontsize=14) #('$G(x)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)
# ax.set_ylabel('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=11) #('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)

plt.savefig('PWLillustrationTwoParametrizations.pdf', bbox_inches = "tight", transparent=True)
# plt.show()
plt.close()

del fig, ax, MyMarkerSize

##############################################################
######################### Third plot########################
##############################################################
print("\n =========== Figure 3 ============ \n")
print("\n xTVaRExpillustration_Poster.pdf ")

fig, ax = plt.subplots(1, figsize=(9,3), dpi=300) #(6,3)

MinX = -0.1
MaxX = 35
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

ax.plot([TVaR,TVaR],[0.13,.3],        linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([TVaR,TVaR],[.45,.926],        linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([VaR,VaR],  [0.13,.3],   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([VaR,VaR],  [.45,BarHeight],   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([Mean,Mean],[0.13,.3],   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([Mean,Mean],[.45,0.63],   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.text(TVaR, 0.32, '$TVaR_{0.8}[X]=\mathbb{E}[X|X > VaR_{0.8}(X)]$', verticalalignment='bottom', horizontalalignment='center', fontsize=14)
ax.text(Mean, 0.32, '$\mathbb{E}[X]$', verticalalignment='bottom', horizontalalignment='center', fontsize=14)
ax.text(VaR,  0.32, '$VaR_{0.8}(X)$' , verticalalignment='bottom', horizontalalignment='center', fontsize=14)
# ax.text(5, 0.9, '$(1-0.8)xTVaR_{0.8}[X]$' , verticalalignment='bottom', horizontalalignment='center', fontsize=14)
x1=np.ma.masked_less(x, Mean)
#ax.fill_between(x,y1 = y, y2=1, where= x>=VaR, facecolor='k', alpha=0.5, linewidth = 0, color = 'k')
#ax.fill_between(x1,y1 = 0.8, y2=1, where= x1<=VaR, facecolor='k', alpha=0.5, linewidth = 0, color = 'k')
# ax.arrow(10, 0.95, -1.45, 0, head_width=0.05, head_length=0.1, fc='k', ec='k')

PlotBrace(ax, x1 = Mean, x2 = TVaR, y = 0.1, braceheight = 0.03, bracewidth = 0.5, text = '$TVaR_{0.8}^\Delta[X]$', upper=False, fontsize=14)

ax.yaxis.set_ticks([ 0., 0.2, 0.4, 0.6, 0.8, 1.0])

ax.set_xlim([MinX,MaxX])
ax.set_ylim([-0.05,1.05])

plt.savefig('xTVaRExpillustration_Poster.pdf', bbox_inches = "tight", transparent=True)
plt.close()

del fig, ax, alpha, theta, x, y, VaR, Mean, TVaR, BarHeight, x1

##############################################################
######################### Fourth plot ##############
##############################################################

print("\n =========== Figure 4 ============")
print("\n AlgorithmSteps2x2.png ")

#fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True, figsize=(8,3.5), dpi=200) # for text without two column setting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True, figsize=(6,5), dpi=200) # for text without two column setting
fig.subplots_adjust(wspace=0.5)
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


ax2.annotate('simple', xy=(0, 0.5), xycoords='data',
                 xytext=(100, 60), textcoords='offset points',
                 size=20,
                 # bbox=dict(boxstyle="round", fc="0.8"),
                 arrowprops=dict(arrowstyle="simple",
                                 fc="0.6", ec="none",
                                 connectionstyle="arc3,rad=0.3"),
                 )

ax1.set_title('Step 1')
ax2.set_title('Step 2')
ax3.set_title('Step 3')
ax4.set_title('Final step')

# ax.xlabel('$G\sim PWL(\mathbf{x},\mathbf{y}=(0,1))$')
plt.savefig('AlgorithmSteps2x2.png', bbox_inches = "tight")
plt.close()

del ax1, ax2, ax3, ax4, fig, mu, LNSample, Step1, Step2, Step3, Step4