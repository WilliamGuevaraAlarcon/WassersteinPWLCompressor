
import numpy as np
import matplotlib.pyplot as plt
from wasserstein_pwl_core.compressor import PWLcompressor
from wasserstein_pwl_core.pwl_distribution import PiecewiseLinearDistribution, EmpiricalDistributionFromSample
from wasserstein_pwl_core.sample_characteristics import SampleCharacteristics
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from matplotlib.patches import Patch
#from matplotlib.path import Path
from Graphics.GenericIllustrationMethods import MyMark_inset, Characteristics, BoundPlot, SetAxisBoundLineWidth, PlotBrace, LognormalParameters, XLsimulations


##############################################################
######################### Fifth plot #########################
##############################################################
print("\n =========== Example 5 ============ \n")
print("\n PWLillustration.pdf ")



######### 'PWLillustration.pdf'

fig, ax = plt.subplots(1, figsize=(8,3), dpi=200) #(6,3)

MinX = -1.2 # -0.2
MaxX = 12.2 # 10.2
#  K =4, x =(1, 4, 4,9), y = (0, 0.6,0.8,1),
MyMarkerSize = 9
ax.plot([MinX,1,4],[0,0,0.6],   linewidth=1.0, linestyle = '-', marker = '', color = 'k')
ax.plot([4,6,9,MaxX],[0.8,0.8,1,1],   linewidth=1.0, linestyle = '-', marker = '',color = 'k')
ax.plot([1,4,4,6,9],[0,0.6,0.8,0.8,1],   linewidth=1.0, linestyle = '', marker = '.', markersize=MyMarkerSize, color = 'k')
ax.plot([4],[0.6],   linewidth=1.0, linestyle = '', marker = '.', markersize=MyMarkerSize*0.5, color = 'w')
# ax.scatter([4],[0.6],   color = 'k', facecolors='r', alpha = 1)
# ax.plot([4],[0.6],   linewidth=1.0, linestyle = '', marker = 'o', color = 'k', facecolors='none')
# PlotBrace(ax, x1 = m1, x2 = m1+delta1, y = 0.64, braceheight = 0.03, bracewidth = 0.5, text = '$\delta_1$')

# ax.yaxis.set_ticks([ 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xlim([MinX,MaxX])
ax.set_ylim([-0.05,1.05])

# ax.set_xlabel('$x$', fontsize=9)
ax.set_ylabel('$G(t)$', fontsize=14) #('$G(x)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)
# ax.set_ylabel('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=11) #('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)

plt.savefig('PWLillustration.pdf')
# plt.show()
plt.close()

del fig, ax, MyMarkerSize

#############################################################
#########################  Wasserstein distance########################
##############################################################
print("\n =========== Wasserstein distance exact ============ \n")


Sample = np.asarray([ 1, 1.6, 4.3, 4.6, 6, 7.1, 13, 15.6, 16, 18.8]) #sorted!
SampleDist = EmpiricalDistributionFromSample(Sample)
PWLAprox = PiecewiseLinearDistribution([0.8, 7.4, 13.95, 17.75],[0, 0.6, 0.6, 1])

fig, ax = plt.subplots(1, figsize=(6,3), dpi=200)

ax.set_xlim([0,20])
ax.set_ylim([0.0,1.0])

x = np.linspace(0,20,1000)
y1 = SampleDist.cdf(x)
y2 = PWLAprox.cdf(x)

ax.plot(SampleDist.Xvalues, SampleDist.Fvalues, linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot(x ,y2, linewidth=1.0, linestyle = '--', marker = '', color = 'k')

plt.fill_between(x, y1, y2, color='grey', alpha='0.5')

# for i in range(n):
#     if Sample[i] > PWLAprox.quantile((i+1/2)/n):
#         x = np.array([PWLAprox.quantile((i+1/2)/n), Sample[i]])
#     else:
#         x = np.array([Sample[i], PWLAprox.quantile((i+1/2)/n)])
#     y = np.array([(i+1/2)/n, (i+1/2)/n])
#     ax.plot(x, y, color = "black", linestyle = "-")
#ax.fill_between()
#
#  legend = ax.legend(loc='lower right', shadow=True)

plt.savefig('WassersteinExact.pdf')
plt.close()

#############################################################
#########################  Wasserstein distance########################
##############################################################
print("\n =========== Wasserstein distance discretized ============ \n")


Sample = np.asarray([ 1, 1.6, 4.3, 4.6, 6, 7.1, 13, 15.6, 16, 18.8]) #sorted!
SampleDist = EmpiricalDistributionFromSample(Sample)
PWLAprox = PiecewiseLinearDistribution([0.8, 7.4, 13.95, 17.75],[0, 0.6, 0.6, 1])

fig, ax = plt.subplots(1, figsize=(6,3), dpi=200)

ax.set_xlim([0,20])
ax.set_ylim([0.0,1.0])

x = np.linspace(0,20,1000)
y1 = SampleDist.cdf(x)
y2 = PWLAprox.cdf(x)

ax.plot(SampleDist.Xvalues, SampleDist.Fvalues, linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot(x ,y2, linewidth=1.0, linestyle = '--', marker = '', color = 'k')

n = Sample.size

for i in range(n):
    if Sample[i] > PWLAprox.quantile((i+1/2)/n):
        x = np.array([PWLAprox.quantile((i+1/2)/n), Sample[i]])
    else:
        x = np.array([Sample[i], PWLAprox.quantile((i+1/2)/n)])
    y = np.array([(i+1/2)/n, (i+1/2)/n])
    ax.plot(x, y, color = "black", linestyle = "-")

plt.savefig('WassersteinDiscretized.pdf')
plt.close()

##############################################################
#########################  #########################
##############################################################
print("\n =========== Theorem 3.2 ============ \n")
print("\n WassersteinDistance.pdf ")

######### Non cross

Sample = np.asarray([1.0, 1.1, 1.2, 1.6, 4.3, 4.5, 4.6, 6, 6.1, 6.6,
          7.1, 13, 13.4, 16, 18.8, 22, 30, 32, 39, 40])#sorted!
SampleDist = EmpiricalDistributionFromSample(Sample)

fig, ax = plt.subplots(1, figsize=(8,3), dpi=200) #(6,3)

MinX = 17
MaxX = 21

#  K =4, x =(1, 4, 4,9), y = (0, 0.6,0.8,1),
MyMarkerSize = 9
#ax.plot([3.583535354, 6.136868687],[0.2, 0.35], linewidth=1.0, linestyle = '-', marker = '', color = 'k')
#ax.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=1.0, linestyle = '--', marker = '', color = 'k')
Approx = PWLcompressor(Sample, AccuracyMode="Absolute", AccuracyParameter= 0.65)

#ax.plot([10.589999999999989, 22.690000000000005],
#        [0.55000000000000004, 0.80000000000000004],  linewidth=1.0, linestyle = '-', marker = '', color = 'k')

ax.plot([18.5, 20.5],
        [0.67, 0.77],  linewidth=1.0, linestyle = '-', marker = '', color = 'k')

ax.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=1.0, linestyle = '--', marker = '', color = 'k')

# ax.scatter([4],[0.6],   color = 'k', facecolors='r', alpha = 1)
# ax.plot([4],[0.6],   linewidth=1.0, linestyle = '', marker = 'o', color = 'k', facecolors='none')
# PlotBrace(ax, x1 = m1, x2 = m1+delta1, y = 0.64, braceheight = 0.03, bracewidth = 0.5, text = '$\delta_1$')




# ax.yaxis.set_ticks([ 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xlim([MinX,MaxX])
ax.set_ylim([0.67, 0.76])

ax.set_xticks([17.85, 18.8, 19.06, 20.27])
ax.set_yticks([0.7, 0.725, 0.75])
ax.set_yticklabels([r'$\frac{i-1}{n}$', r'$\frac{i-1/2}{n}$', r'$\frac{i}{n}$'])
ax.set_xticklabels(['', r'$X_{(i)}$', r'$G^{\leftarrow}\left(\frac{i-1/2}{n}\right)$', ''], size =  'x-small')
ax.plot([19.06,19.06],[0.69, 0.76] ,   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
PlotBrace(ax, x1 = 17.85, x2 = 18.8, y = 0.7, braceheight = 0.005, bracewidth = 0.5, text = r'$\delta^*_s-|X_{(i)}-G^{\leftarrow}\left(\frac{i-1/2}{n}\right)|$', upper=False, fontsize=8)
#PlotBrace(ax, x1 = 17.85, x2 = 19.06, y = 0.69, braceheight = 0.03, bracewidth = 0.5, text = r'$\delta^*_s$', upper=False, fontsize=10)

# ax.set_xlabel('$x$', fontsize=9)
ax.set_ylabel('$G(t)$', fontsize=14) #('$G(x)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)
# ax.set_ylabel('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=11) #('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)

plt.savefig('ExactWasserstein_1.pdf')
# plt.show()
plt.close()

del fig, ax, MyMarkerSize


##########################################
##########################################
##########################################
Approx = PWLcompressor(Sample, AccuracyMode="Absolute", AccuracyParameter= 0.65)

fig, ax = plt.subplots(1, figsize=(8,3), dpi=200) #(6,3)

MinX = 17
MaxX = 21

#  K =4, x =(1, 4, 4,9), y = (0, 0.6,0.8,1),
MyMarkerSize = 9
#ax.plot([3.583535354, 6.136868687],[0.2, 0.35], linewidth=1.0, linestyle = '-', marker = '', color = 'k')
#ax.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=1.0, linestyle = '--', marker = '', color = 'k')

ax.plot(Approx.Result['PWLX'], Approx.Result['PWLY'], linewidth=1.0, linestyle = '-', marker = '', color = 'k')
ax.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=1.0, linestyle = '--', marker = '', color = 'k')

# ax.scatter([4],[0.6],   color = 'k', facecolors='r', alpha = 1)
# ax.plot([4],[0.6],   linewidth=1.0, linestyle = '', marker = 'o', color = 'k', facecolors='none')
    # PlotBrace(ax, x1 = m1, x2 = m1+delta1, y = 0.64, braceheight = 0.03, bracewidth = 0.5, text = '$\delta_1$')




# ax.yaxis.set_ticks([ 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xlim([MinX,MaxX])
ax.set_ylim([0.67, 0.76])

ax.set_xticks([17.85, 18.8, 19.06, 20.27])
ax.set_yticks([0.7, 0.725, 0.75])
ax.set_yticklabels([r'$\frac{i-1}{n}$', r'$\frac{i-1/2}{n}$', r'$\frac{i}{n}$'])
ax.set_xticklabels(['', r'$X_{(i)}$', r'$G^{\leftarrow}\left(\frac{i-1/2}{n}\right)$', ''], size =  'x-small')
ax.plot([19.06,19.06],[0.69, 0.76] ,   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.annotate(r'$\delta^*_s-|X_{(i)}-G^{\leftarrow}\left(\frac{i-1/2}{n}\right)|$', xy=(.33, .3), xytext=(.33, .2), xycoords = "axes fraction", ha="center", va="top",
            arrowprops = dict(arrowstyle = '-[, widthB=5.5, lengthB = 0.5', lw =1 ))
ax.annotate(r'$\delta^*_s-|X_{(i)}-G^{\leftarrow}\left(\frac{i-1/2}{n}\right)|$', xy=(.33, .3), xytext=(.33, .2), xycoords = "axes fraction", ha="center", va="top",
            arrowprops = dict(arrowstyle = '-[, widthB=5.5, lengthB = 0.5', lw =1 ))

#PlotBrace(ax, x1 = 17.85, x2 = 18.8, y = 0.7, braceheight = 0.01, bracewidth = 0.5, text = r'$\delta^*_s-|X_{(i)}-G^{\leftarrow}\left(\frac{i-1/2}{n}\right)|$', upper=False, fontsize=8)
#PlotBrace(ax, x1 = 17.85, x2 = 19.06, y = 0.69, braceheight = 0.03, bracewidth = 0.5, text = r'$\delta^*_s$', upper=False, fontsize=10)

# ax.set_xlabel('$x$', fontsize=9)
ax.set_ylabel('$G(t)$', fontsize=14) #('$G(x)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)
# ax.set_ylabel('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=11) #('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)

plt.savefig('ExactWasserstein_2.pdf')
# plt.show()
plt.close()

del fig, ax, MyMarkerSize

#############################################################
#########################  Delta parameter ########################
##############################################################
print("\n =========== Delta parameter ============ \n")


Sample = np.asarray([ 1, 1.6, 4.3, 4.6, 6, 7.1, 13, 15.6, 16, 18.8]) #sorted!
SampleDist = EmpiricalDistributionFromSample(Sample)

PWLAprox = PiecewiseLinearDistribution([0.38, 7.82, 12.05, 19.65],[0, 0.6, 0.6, 1])#delta_2 = 3.8


fig, ax = plt.subplots(1, figsize=(6,3), dpi=200)

ax.set_xlim([0,20])
ax.set_ylim([0.0,1.0])

x = np.linspace(0,20,1000)
y = PWLAprox.cdf(x)

ax.plot(SampleDist.Xvalues, SampleDist.Fvalues, linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot(x ,y, linewidth=1.0, linestyle = '--', marker = '', color = 'k')


#ax.plot(CompressedSample['PWLX'],CompressedSample['PWLY'], linewidth=1.0, linestyle = '-', marker = '', color = 'k', label='Some PWL distribution')
#ax.fill_between(, y1 = SampleDist.Fvalues, y2 = CompressedSample['PWLY'], facecolor='k', alpha=0.5, linewidth = 0, color = 'k')
#plt.fill_between(x, y, y3, color='grey', alpha='0.5')
#
n = Sample.size

for i in range(n):
    if Sample[i] > PWLAprox.quantile((i+1/2)/n):
        x1 = np.array([PWLAprox.quantile((i+1/2)/n), Sample[i]])
    else:
        x1 = np.array([Sample[i], PWLAprox.quantile((i+1/2)/n)])

    # z = np.array([(i+1/2)/n, (i+1/2)/n])
    # ax.plot(x1, z, color = "red", linestyle = "-")
    # ax.plot(x2, z, color = "blue", linestyle="-")
    # ax.plot(x3, z, color="k", linestyle="-")

    print("low", x1[1]-x1[0])

legend = ax.legend(loc='lower right', shadow=True)

plt.savefig('DeltaParameter.pdf')
plt.close()

#############################################################
#########################  Delta 1 ########################
##############################################################
print("\n =========== Delta1 ============ \n")

Sample = np.asarray([ 1, 1.6, 4.3, 4.6, 6, 7.1, 13, 15.6, 16, 18.8]) #sorted!
SampleChar = SampleCharacteristics(Sample)
SampleLength = SampleChar.SampleSize

CutPoint = 5

Delta1 = np.linspace(-2.2, 6, 1000)
Delta2 = np.linspace(-0.6, 5, 1000)

Mult1 = SampleChar.Multiplier(0, CutPoint)
Mult2 = SampleChar.Multiplier(CutPoint+1, SampleLength-1)
W1 =   [SampleChar.WD(0, CutPoint, Mult1, d) for d in Delta1]
W2 =   [SampleChar.WD(CutPoint+1, SampleLength-1, Mult2, d) for d in Delta2]

 #S2 = SampleChar.FindBestSolutionLine(CutPoint+1,SampleLength-1)

fig, ax = plt.subplots(1, figsize=(6,4), dpi=200)

ax.set_xlim([0,6])
ax.set_ylim([0, 12])

ax.plot(Delta1 ,W1, linewidth=1.0, linestyle = '-', marker = '', color = 'k')

#ax.plot([3.72,3.72],[0, 18] ,   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
#ax.plot([3.8, 3.8] ,[0, 18] ,   linewidth=1.0, linestyle = '-', marker = '', color = 'k')

#ax.text(3.72, 0.0 , '$\delta_1^W$', verticalalignment = 'bottom', horizontalalignment = 'center', fontsize = 10)

ax.set_xticks([0, 1, 2, 3, 3.72, 4, 5, 6])
ax.set_xticklabels(['0', '1', '2', '3', '$\delta_1^W$', '4', '5', '6'])

ax.set_xlabel('$\delta_s$', fontsize=14)
ax.set_ylabel('$\omega_1(\delta_1)$', fontsize=14)

#plt.show()
plt.savefig('Delta1.pdf')
plt.close()

 #S2 = SampleChar.FindBestSolutionLine(CutPoint+1,SampleLength-1)
fig, ax = plt.subplots(1, figsize=(6,4), dpi=200)

ax.set_xlim([0, 5])
ax.set_ylim([0, 7])

ax.plot(Delta2 ,W2, linewidth=1.0, linestyle = '-', marker = '', color = 'k')

#ax.text(3.8, 0.0 , '$\delta_2^W$', verticalalignment = 'bottom', horizontalalignment = 'center', fontsize = 10)

ax.set_xticks([0, 1, 2, 3, 3.8, 4, 5])
ax.set_xticklabels(['0', '1', '2', '3', '$\delta_2^W$', '4', '5',])

ax.set_xlabel('$\delta_2$', fontsize=14)
ax.set_ylabel('$\omega_2(\delta_2)$', fontsize=14)

#plt.show()
plt.savefig('Delta2.pdf')
plt.close()

##############################################################
######################## First Plot ##########################
##############################################################
print("\n =========== Example ============ \n")
print("\n NegativeIncrement.pdf ")

Sample = np.asarray([ 1, 1.6, 4.3, 4.6, 6, 7.1, 13, 15.6, 16, 18.8]) #sorted!
SampleChar = SampleCharacteristics(Sample)
SampleLength = SampleChar.SampleSize

CutPoint = 2
S1 = SampleChar.FindBestSolutionLine(0,CutPoint)
S2 = SampleChar.FindBestSolutionLine(CutPoint+1,SampleLength-1)

CharS1 = Characteristics(S1)
CharS2 = Characteristics(S2)

# midpoint = (min(CharS1['UpReg'],CharS2['LoMax'])+max(CharS1['UpMin'],CharS2['LoReg']))/2
midpoint = (CharS1['UpSel']+CharS2['LoSel'])/2

S1.SetDelta(S1.Calculate_DeltaFromEndX(midpoint))
S2.SetDelta(S2.Calculate_DeltaFromStartX(midpoint))

CharS1 = Characteristics(S1)
CharS2 = Characteristics(S2)


print("[CharS1['UpSel'],CharS2['LoSel']]")
print([CharS1['UpSel'],CharS2['LoSel']])
print('[S1.Delta_Selected, S2.Delta_Selected, midpoint]')
print([S1.Delta_Selected, S2.Delta_Selected, midpoint])
print('[S1.Mean, S1.Delta_Regression]')
print([S1.Mean, S1.Delta_Regression])
print('[S2.Mean,S2.Delta_Regression]')
print([S2.Mean, S2.Delta_Regression])

FValues = [0,(CutPoint+1)/SampleLength,(CutPoint+1)/SampleLength,1]
FValues1 = FValues[0:2]
FValues2 = FValues[2:4]

# fig = plt.figure(figsize=(4,5), dpi=200) #figure size and resolution  # before: figsize=(6,3)
# ax1 = plt.subplot(2, 1, 1)
# ax2 = plt.subplot(2, 1, 2)
#fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(4,4), dpi=200)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6,2), dpi=200) # for text without two column setting

PWLX_Values = [S1.Segment_Line_Start_X, S1.Segment_Line_End_X, S2.Segment_Line_Start_X,S2.Segment_Line_End_X]


zoom = 1.9
ax1ins = zoomed_inset_axes(ax1, zoom, loc=4) # zoom = 6
ax2ins = zoomed_inset_axes(ax2, zoom, loc=4) # zoom = 6

#ax1ins.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=1.0, linestyle = '--', marker = '', color = 'k')

for axIns in [ax1ins,ax2ins]:
     axIns.set_xlim(1.2, 6.5)
     axIns.set_ylim(0.18, 0.42)
     axIns.set_xticklabels([])
     axIns.set_xticks([])
     axIns.set_yticklabels([])
     axIns.set_yticks([])
     SetAxisBoundLineWidth(axIns)
     #axIns.axhline(linewidth=40, color="g")

for ax in [ax1, ax1ins]:
    ax.plot([CharS1['LoReg'],CharS1['UpReg']], FValues1, linewidth=1.0, linestyle = '-', marker = '', color = 'k')
    ax.plot([CharS2['LoReg'],CharS2['UpReg']], FValues2, linewidth=1.0, linestyle = '-', marker = '', color = 'k')

for ax in [ax2, ax2ins]:
    ax.plot(PWLX_Values, FValues, linewidth=1.0, linestyle = '-', marker = '', color = 'k')

for ax in [ax1, ax2, ax1ins, ax2ins]:
    ax.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=1.0, linestyle = ':', marker = '', color = 'k')



ax1.set_title('Using $\delta_1=\delta_1^{W}$ and $\delta_2=\delta_2^{W}$', fontsize=9)
ax2.set_title('After adjusting $\delta_1$ and $\delta_2$', fontsize=9)

ax1.tick_params(axis='both', which='major', labelsize=9)
ax2.tick_params(axis='both', which='major', labelsize=9)

#plt.savefig('MeanInvarianceExampleDist.pdf')
ax1.set_xlim([-1,22])
ax2.set_xlim([-1,22])

#plt.xticks(visible=False)
#plt.yticks(visible=False)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
#mark_inset(ax1, ax1ins, loc1=1, loc2=4, fc="none", ec="0.4", zorder = -10, linewidth=0.4)
for (ax,axIns) in [(ax1,ax1ins),(ax2,ax2ins)]:
     MyMark_inset(ax, axIns, loc11=2, loc12=1, loc21=3, loc22=4, fc="none", ec="0.4", zorder = -10, linewidth=0.4)


#ax1.set_title('$\,$') #empty space to get the suptitle spacing right

#t = fig.suptitle('Fixing negative increments \n by resetting $\delta$', fontsize=14)

# plt.subplots_adjust(top=0.86)



plt.savefig('NegativeIncrement.pdf')
# plt.show()
plt.close()


del fig, ax1, ax2, CharS1, CharS2, midpoint, S1, S2, CutPoint, PWLX_Values

