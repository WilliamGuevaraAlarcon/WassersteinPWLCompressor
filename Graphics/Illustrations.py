
import numpy as np
import matplotlib.pyplot as plt
from pwl_compressor_core.compressor import PWLcompressor
from pwl_compressor_core.pwl_distribution import PiecewiseLinearDistribution, EmpiricalDistributionFromSample
from pwl_compressor_core.sample_characteristics import SampleCharacteristics
from matplotlib import rcParams

#from matplotlib.patches import Patch
#from matplotlib.path import Path
from Graphics.ConvergenceRateClass import ConvergeRateCalculator
from Graphics.GenericIllustrationMethods import MyMark_inset, Characteristics, BoundPlot, SetAxisBoundLineWidth, PlotBrace, LognormalParameters, XLsimulations

# from Graphics.PlotBraceTool import PlotBrace

#from matplotlib import patches
#http://matplotlib.org/1.3.1/mpl_toolkits/axes_grid/users/overview.html
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset, BboxPatch, BboxConnector


rcParams.update({'figure.autolayout': True})
rcParams['font.family'] = 'serif'

# IncludeConvergenceErrorGraph = False


##############################################################
######################## Generic Functions ###################
##############################################################

Sample = np.asarray([ 1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]) #sorted!
SampleLength = len(Sample)
SampleDist = EmpiricalDistributionFromSample(Sample)



##############################################################
######################## First Plot ##########################
##############################################################
print("\n =========== Example 1 ============ \n")
print("\n NegativeIncrementDeltaFix.pdf ")

Accuracy = 0.2
#Accuracy = 0.08 #not compatible
SampleChar = SampleCharacteristics(Sample, Accuracy = Accuracy)


CutPoint = 2
S1 = SampleChar.FindBestSolutionLine(0,CutPoint)
S2 = SampleChar.FindBestSolutionLine(CutPoint+1,SampleLength-1)

CharS1 = Characteristics(S1)
CharS2 = Characteristics(S2)

# midpoint = (min(CharS1['UpReg'],CharS2['LoMax'])+max(CharS1['UpMin'],CharS2['LoReg']))/2
midpoint = (max(CharS1['UpReg'],CharS2['LoMin'])+min(CharS1['UpMax'],CharS2['LoReg']))/2

S1.SetDelta(S1.Calculate_DeltaFromEndX(midpoint))
S2.SetDelta(S2.Calculate_DeltaFromStartX(midpoint))

CharS1 = Characteristics(S1)
CharS2 = Characteristics(S2)


print("[CharS1['UpReg'],CharS1['UpMin'],CharS1['UpSel'],CharS2['LoReg'],CharS2['LoMax'],CharS2['LoSel']]")
print([CharS1['UpReg'],CharS1['UpMin'],CharS1['UpSel'],CharS2['LoReg'],CharS2['LoMax'],CharS2['LoSel']])
print('[S1.Delta_Regression, S1.Delta_Selected, S2.Delta_Regression, S2.Delta_Selected, midpoint]')
print([S1.Delta_Regression, S1.Delta_Selected, S2.Delta_Regression, S2.Delta_Selected, midpoint])
print('[S1.Mean, S1.Delta_LowerBound, S1.Delta_UpperBound, S1.Delta_Regression]')
print([S1.Mean, S1.Delta_LowerBound, S1.Delta_UpperBound, S1.Delta_Regression])
print('[S2.Mean, S2.Delta_LowerBound, S2.Delta_UpperBound, S2.Delta_Regression]')
print([S2.Mean, S2.Delta_LowerBound, S2.Delta_UpperBound, S2.Delta_Regression])

FValues = [0,(CutPoint+1)/SampleLength,(CutPoint+1)/SampleLength,1]
FValues1 = FValues[0:2]
FValues2 = FValues[2:4]

# fig = plt.figure(figsize=(4,5), dpi=200) #figure size and resolution  # before: figsize=(6,3)
# ax1 = plt.subplot(2, 1, 1)
# ax2 = plt.subplot(2, 1, 2)
#fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(4,4), dpi=200)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6,2), dpi=200) # for text without two column setting



zoom = 1.9
ax1ins = zoomed_inset_axes(ax1, zoom, loc=4) # zoom = 6
ax2ins = zoomed_inset_axes(ax2, zoom, loc=4) # zoom = 6
#loc =         BEST, UR, UL, LL, LR, R, CL, CR, LC, UC, C = list(range(11))

#ax1ins.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=1.0, linestyle = '--', marker = '', color = 'k')

for axIns in [ax1ins,ax2ins]:
    axIns.set_xlim(1.2, 6.5)
    axIns.set_ylim(0.18, 0.42)
    axIns.set_xticklabels([])
    axIns.set_xticks([])
    axIns.set_yticklabels([])
    axIns.set_yticks([])
    SetAxisBoundLineWidth(axIns)
    # axIns.axhline(linewidth=40, color="g")

for ax in [ax1, ax1ins, ax2, ax2ins]:
    BoundPlot(ax, [CharS1['LoMin'],CharS1['UpMax']], FValues1)
    BoundPlot(ax, [CharS1['LoMax'],CharS1['UpMin']], FValues1)
    BoundPlot(ax, [CharS2['LoMin'],CharS2['UpMax']], FValues2)
    BoundPlot(ax, [CharS2['LoMax'],CharS2['UpMin']], FValues2)
    ax.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=0.7, linestyle = '--', marker = '', color = 'k', dashes = (3,2))

for ax in [ax1, ax1ins]:
    ax.plot([CharS1['LoReg'],CharS1['UpReg']], FValues1, linewidth=1.0, linestyle = '-', marker = '', color = 'k')
    ax.plot([CharS2['LoReg'],CharS2['UpReg']], FValues2, linewidth=1.0, linestyle = '-', marker = '', color = 'k')

PWLX_Values = [S1.Segment_Line_Start_X, S1.Segment_Line_End_X, S2.Segment_Line_Start_X,S2.Segment_Line_End_X]
for ax in [ax2, ax2ins]:
    ax.plot(PWLX_Values, FValues, linewidth=1.0, linestyle = '-', marker = '', color = 'k')

ax1.set_title('Using $\delta_1=\delta_1^{reg}$ and $\delta_2=\delta_2^{reg}$', fontsize=9)
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


# ax1.set_title('$\,$') #empty space to get the suptitle spacing right

# t = fig.suptitle('Fixing negative increments \n by resetting $\delta$', fontsize=14)

# plt.subplots_adjust(top=0.86)



plt.savefig('NegativeIncrementDeltaFix.pdf')
# plt.show()
plt.close()


del fig, ax1, ax2, CharS1, CharS2, midpoint, S1, S2, CutPoint, ax1ins, ax2ins, PWLX_Values, Accuracy




##############################################################
######################### Second plot #########################
##############################################################
print("\n =========== Example 2 ============ \n")
print(" NegativeIncrementBisection.pdf ")

# Accuracy = 0.2
Accuracy = 0.08 #0.08 #not compatible
SampleChar = SampleCharacteristics(Sample, Accuracy = Accuracy)

CutPoint = 2
FValues = [0,(CutPoint+1)/SampleLength,(CutPoint+1)/SampleLength,1]
FValues1 = FValues[0:2]
FValues2 = FValues[2:4]
S1 = SampleChar.FindBestSolutionLine(0,CutPoint)
S2 = SampleChar.FindBestSolutionLine(CutPoint+1,SampleLength-1)

CharS1 = Characteristics(S1)
CharS2 = Characteristics(S2)

CutPoint1 = 2
CutPoint2 = 5
SA = SampleChar.FindBestSolutionLine(0,CutPoint1)
SB = SampleChar.FindBestSolutionLine(CutPoint1+1,CutPoint2)
SC = SampleChar.FindBestSolutionLine(CutPoint2+1,SampleLength-1)

print("[CharS1['UpReg'], CharS1['UpMin'], CharS1['UpSel'], CharS2['LoReg'], CharS2['LoMax'], CharS2['LoSel']]")
print([CharS1['UpReg'], CharS1['UpMin'], CharS1['UpSel'], CharS2['LoReg'], CharS2['LoMax'], CharS2['LoSel']])
print('[S1.Delta_Regression, S1.Delta_Selected, S2.Delta_Regression, S2.Delta_Selected]')
print([S1.Delta_Regression, S1.Delta_Selected, S2.Delta_Regression, S2.Delta_Selected])
print('[S1.Mean, S1.Delta_LowerBound, S1.Delta_UpperBound, S1.Delta_Regression, S1.BestBisectionPoint]')
print([S1.Mean, S1.Delta_LowerBound, S1.Delta_UpperBound, S1.Delta_Regression, S1.BestBisectionPoint])
print('[S2.Mean, S2.Delta_LowerBound, S2.Delta_UpperBound, S2.Delta_Regression, S2.BestBisectionPoint]')
print([S2.Mean, S2.Delta_LowerBound, S2.Delta_UpperBound, S2.Delta_Regression, S2.BestBisectionPoint])
print('[SA.Mean, SA.Delta_LowerBound, SA.Delta_UpperBound, SA.Delta_Regression, SA.BestBisectionPoint]')
print([SA.Mean, SA.Delta_LowerBound, SA.Delta_UpperBound, SA.Delta_Regression, SA.BestBisectionPoint])
print('[SB.Mean, SB.Delta_LowerBound, SB.Delta_UpperBound, SB.Delta_Regression, SB.BestBisectionPoint]')
print([SB.Mean, SB.Delta_LowerBound, SB.Delta_UpperBound, SB.Delta_Regression, SB.BestBisectionPoint])
print('[SC.Mean, SC.Delta_LowerBound, SC.Delta_UpperBound, SC.Delta_Regression, SC.BestBisectionPoint]')
print([SC.Mean, SC.Delta_LowerBound, SC.Delta_UpperBound, SC.Delta_Regression, SC.BestBisectionPoint])
P1 = SA.Mean + SA.Delta_Regression
P2 = SB.Mean - SB.Delta_Regression
Midpoint = (P1+P2)/2
print('[P1, P2, Midpoint]')
print([P1, P2, Midpoint])
del P1, P2, Midpoint

# [CharS1['UpReg'], CharS1['UpMin'], CharS1['UpSel'], CharS2['LoReg'], CharS2['LoMax'], CharS2['LoSel']]
# [4.5, 3.5528000000000008, 4.5, 2.8346938775510164, 3.1258285714285741, 2.8346938775510164]
# [S1.Delta_Regression, S1.Delta_Selected, S2.Delta_Regression, S2.Delta_Selected]
# [2.2000000000000002, 2.2000000000000002, 8.436734693877554, 8.436734693877554]
# [S1.Mean, S1.Delta_LowerBound, S1.Delta_UpperBound, S1.Delta_Regression, S1.BestBisectionPoint]
# [2.3000000000000003, 1.2528000000000006, 2.8596000000000008, 2.2000000000000002, 1]
# [S2.Mean, S2.Delta_LowerBound, S2.Delta_UpperBound, S2.Delta_Regression, S2.BestBisectionPoint]
# [11.27142857142857, 8.1455999999999964, 9.4397999999999964, 8.436734693877554, 5]
# [SA.Mean, SA.Delta_LowerBound, SA.Delta_UpperBound, SA.Delta_Regression, SA.BestBisectionPoint]
# [2.3000000000000003, 1.2528000000000006, 2.8596000000000008, 2.2000000000000002, 1]
# [SB.Mean, SB.Delta_LowerBound, SB.Delta_UpperBound, SB.Delta_Regression, SB.BestBisectionPoint]
# [5.9000000000000012, 0.0, 4.6884000000000032, 1.6666666666666501, 3]
# [SC.Mean, SC.Delta_LowerBound, SC.Delta_UpperBound, SC.Delta_Regression, SC.BestBisectionPoint]
# [15.299999999999999, 3.5765333333333338, 5.4623999999999988, 3.75, 8]
# [P1, P2, Midpoint]
# [4.5, 4.2333333333333512, 4.366666666666676]

FValues = [0,(CutPoint1+1)/SampleLength,(CutPoint1+1)/SampleLength,(CutPoint2+1)/SampleLength,(CutPoint2+1)/SampleLength,1]
FValuesA = FValues[0:2]
FValuesB = FValues[2:4]
FValuesC = FValues[4:6]

CharSA = Characteristics(SA)
CharSB = Characteristics(SB)
CharSC = Characteristics(SC)

#fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(4,4), dpi=200)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6,2), dpi=200) # for text without two column setting


zoom = 1.9
ax1ins = zoomed_inset_axes(ax1, zoom, loc=4) # zoom = 6
ax2ins = zoomed_inset_axes(ax2, zoom, loc=4) # zoom = 6
#loc =         BEST, UR, UL, LL, LR, R, CL, CR, LC, UC, C = list(range(11))

#ax1ins.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=1.0, linestyle = '--', marker = '', color = 'k')

for axIns in [ax1ins,ax2ins]:
    axIns.set_xlim(0.9, 6.2)
    axIns.set_ylim(0.18, 0.42)
    axIns.set_xticklabels([])
    axIns.set_xticks([])
    axIns.set_yticklabels([])
    axIns.set_yticks([])
    SetAxisBoundLineWidth(axIns)

for ax in [ax1, ax1ins, ax2, ax2ins]:
    ax.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=0.7, linestyle = '--', marker = '', color = 'k', dashes = (3,2))

for ax in [ax1, ax1ins]:
    BoundPlot(ax, [CharS1['LoMin'],CharS1['UpMax']], FValues1)
    BoundPlot(ax, [CharS1['LoMax'],CharS1['UpMin']], FValues1)
    BoundPlot(ax, [CharS2['LoMin'],CharS2['UpMax']], FValues2)
    BoundPlot(ax, [CharS2['LoMax'],CharS2['UpMin']], FValues2)#
    ax.plot([CharS1['LoReg'],CharS1['UpReg']], FValues1, linewidth=1.0, linestyle = '-', marker = '', color = 'k')
    ax.plot([CharS2['LoReg'],CharS2['UpReg']], FValues2, linewidth=1.0, linestyle = '-', marker = '', color = 'k')

for ax in [ax2, ax2ins]:
    BoundPlot(ax, [CharSA['LoMin'],CharSA['UpMax']], FValuesA)
    BoundPlot(ax, [CharSA['LoMax'],CharSA['UpMin']], FValuesA)
    BoundPlot(ax, [CharSB['LoMin'],CharSB['UpMax']], FValuesB)
    BoundPlot(ax, [CharSB['LoMax'],CharSB['UpMin']], FValuesB)#
    BoundPlot(ax, [CharSC['LoMin'],CharSC['UpMax']], FValuesC)
    BoundPlot(ax, [CharSC['LoMax'],CharSC['UpMin']], FValuesC)
    ax.plot([CharSA['LoReg'],CharSA['UpReg']], FValuesA, linewidth=1.0, linestyle = '-', marker = '', color = 'k')
    ax.plot([CharSB['LoReg'],CharSB['UpReg']], FValuesB, linewidth=1.0, linestyle = '-', marker = '', color = 'k')
    ax.plot([CharSC['LoReg'],CharSC['UpReg']], FValuesC, linewidth=1.0, linestyle = '-', marker = '', color = 'k')



ax1.set_title('Incompatible segments', fontsize=9) #with two column layout: set_ylabel
ax2.set_title('After bisection', fontsize=9)



#plt.savefig('MeanInvarianceExampleDist.pdf')
ax1.set_xlim([-1,22])
ax2.set_xlim([-1,22])

ax1.tick_params(axis='both', which='major', labelsize=9)
ax2.tick_params(axis='both', which='major', labelsize=9)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
#mark_inset(ax1, ax1ins, loc1=1, loc2=4, fc="none", ec="0.4", zorder = -10, linewidth=0.4)
for (ax,axIns) in [(ax1,ax1ins),(ax2,ax2ins)]:
    MyMark_inset(ax, axIns, loc11=2, loc12=1, loc21=3, loc22=4, fc="none", ec="0.4", zorder = -10, linewidth=0.4)


plt.savefig('NegativeIncrementBisection.pdf')
#plt.show()
plt.close()

del fig, ax1, ax2, CharS1, CharS2, S1, S2, CutPoint, ax1ins, ax2ins, SA, SB, SC, CharSA, CharSB, CharSC, Accuracy, CutPoint1, CutPoint2, FValuesA, FValuesB, FValuesC




##############################################################
######################### Third plot #########################
##############################################################
print("\n =========== Example 3 ============ \n")
print("\n SmoothingIllustration.pdf ")

Accuracy = 0.25 #0.15
SampleChar = SampleCharacteristics(Sample, Accuracy = Accuracy)


CutPoint = 5 #4
S1 = SampleChar.FindBestSolutionLine(0,CutPoint)
S2 = SampleChar.FindBestSolutionLine(CutPoint+1,SampleLength-1)

CharS1 = Characteristics(S1)
CharS2 = Characteristics(S2)

FValues = [0,(CutPoint+1)/SampleLength,(CutPoint+1)/SampleLength,1]
FValues1 = FValues[0:2]
FValues2 = FValues[2:4]

midpoint = (max(CharS1['UpReg'],CharS2['LoMin'])+min(CharS1['UpMax'],CharS2['LoReg']))/2
print(midpoint)
S1.SetDelta(S1.Calculate_DeltaFromEndX(midpoint))
S2.SetDelta(S2.Calculate_DeltaFromStartX(midpoint))

CharS1 = Characteristics(S1)
CharS2 = Characteristics(S2)


print("[CharS1['UpReg'],CharS1['UpMax'],CharS1['UpSel'],CharS2['LoReg'],CharS2['LoMin'],CharS2['LoSel']]")
print([CharS1['UpReg'],CharS1['UpMax'],CharS1['UpSel'],CharS2['LoReg'],CharS2['LoMin'],CharS2['LoSel']])
print('[S1.Delta_Regression, S1.Delta_Selected, S2.Delta_Regression, S2.Delta_Selected]')
print([S1.Delta_Regression, S1.Delta_Selected, S2.Delta_Regression, S2.Delta_Selected])
print('[S1.Mean, S1.Delta_LowerBound, S1.Delta_UpperBound, S1.Delta_Regression]')
print([S1.Mean, S1.Delta_LowerBound, S1.Delta_UpperBound, S1.Delta_Regression])
print('[S2.Mean, S2.Delta_LowerBound, S2.Delta_UpperBound, S2.Delta_Regression]')
print([S2.Mean, S2.Delta_LowerBound, S2.Delta_UpperBound, S2.Delta_Regression])
print('[FValues = [0,(CutPoint+1)/SampleLength,(CutPoint+1)/SampleLength,1]')
print(FValues)

#fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(4,4), dpi=200)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(6,2), dpi=200) # for text without two column setting


zoom = 1.5
ax1ins = zoomed_inset_axes(ax1, zoom, loc=4) # zoom = 6
ax2ins = zoomed_inset_axes(ax2, zoom, loc=4) # zoom = 6
#loc =         BEST, UR, UL, LL, LR, R, CL, CR, LC, UC, C = list(range(11))

#ax1ins.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=1.0, linestyle = '--', marker = '', color = 'k')

for axIns in [ax1ins,ax2ins]:
    axIns.set_xlim(5, 15)
    axIns.set_ylim(0.52, 0.68)
    axIns.set_xticklabels([])
    axIns.set_xticks([])
    axIns.set_yticklabels([])
    axIns.set_yticks([])
    SetAxisBoundLineWidth(axIns)


for ax in [ax1, ax1ins, ax2, ax2ins]:
    BoundPlot(ax, [CharS1['LoMin'],CharS1['UpMax']], FValues1)
    BoundPlot(ax, [CharS1['LoMax'],CharS1['UpMin']], FValues1)
    BoundPlot(ax, [CharS2['LoMin'],CharS2['UpMax']], FValues2)
    BoundPlot(ax, [CharS2['LoMax'],CharS2['UpMin']], FValues2)
    ax.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=0.7, linestyle = '--', marker = '', color = 'k', dashes = (3,2))

for ax in [ax1, ax1ins]:
    ax.plot([CharS2['LoReg'],CharS2['UpReg']], FValues2, linewidth=1.0, linestyle = '-', marker = '', color = 'k')
    ax.plot([CharS1['LoReg'],CharS1['UpReg']], FValues1, linewidth=1.0, linestyle = '-', marker = '', color = 'k')

PWLX_Values2 = [S1.Segment_Line_Start_X, S1.Segment_Line_End_X,S2.Segment_Line_Start_X, S2.Segment_Line_End_X]
for ax in [ax2, ax2ins]:
    ax.plot(PWLX_Values2, FValues, linewidth=1.0, linestyle = '-', marker = '', color = 'k')

ax1.set_title('Using $\delta_1=\delta_1^{reg}$ and $\delta_2=\delta_2^{reg}$', fontsize=9)
ax2.set_title('After smoothing', fontsize=9)

#plt.savefig('MeanInvarianceExampleDist.pdf')
ax1.set_xlim([-1,22])
ax2.set_xlim([-1,22])

# ax1.set_title('$\,$') #empty space to get the suptitle spacing right

# t = fig.suptitle('Fixing negative increments \n by resetting $\delta$', fontsize=14)

# plt.subplots_adjust(top=0.86)

ax1.tick_params(axis='both', which='major', labelsize=9)
ax2.tick_params(axis='both', which='major', labelsize=9)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
#mark_inset(ax1, ax1ins, loc1=1, loc2=4, fc="none", ec="0.4", zorder = -10, linewidth=0.4)
for (ax,axIns) in [(ax1,ax1ins),(ax2,ax2ins)]:
    MyMark_inset(ax, axIns, loc11=1, loc12=1, loc21=3, loc22=3, fc="none", ec="0.4", zorder = -10, linewidth=0.4)


plt.savefig('SmoothingIllustration.pdf')
#plt.show()
plt.close()

del fig, ax1, ax2, CharS1, CharS2, S1, S2, CutPoint, ax1ins, ax2ins, Accuracy, FValues, FValues1, FValues2, midpoint




##############################################################
######################### Fourth plot #########################
##############################################################
print("\n =========== Example 4 ============ \n")
print("\n MinMaxDeltaParametrizationIllustration.pdf ")

#Accuracy = 0.1
#Accuracy = 0.08 #not compatible
SampleChar = SampleCharacteristics(Sample, Accuracy = 0.25)
SampleChar2 = SampleCharacteristics(Sample, Accuracy = 0.1)

k = 6
N = len(Sample)
S1 = SampleChar.FindBestSolutionLine(0,k-1)
S2 = SampleChar2.FindBestSolutionLine(0,k-1)

CharS1 = Characteristics(S1)
CharS2 = Characteristics(S2)

FValues = [0,k/SampleLength,k/SampleLength,1]
FValues1 = FValues[0:2]


print("[CharS1['UpReg'],CharS1['UpMax'],CharS1['UpSel']]")
print([CharS1['UpReg'],CharS1['UpMax'],CharS1['UpSel']])
print('[S1.Delta_Regression, S1.Delta_Selected]')
print([S1.Delta_Regression, S1.Delta_Selected])

m1 = np.mean(Sample[:k])
DeltaMax = S1.Delta_UpperBound
DeltaMin = S1.Delta_LowerBound
DeltaMax2 = S2.Delta_UpperBound
DeltaMin2 = S2.Delta_LowerBound
PWLSegmentX1 = [m1-DeltaMax,m1+DeltaMax]
PWLSegmentX2 = [m1-DeltaMin,m1+DeltaMin]
PWLSegmentX3 = [m1-DeltaMax2,m1+DeltaMax2]
PWLSegmentX4 = [m1-DeltaMin2,m1+DeltaMin2]
print('[m1, DeltaMin, DeltaMax, PWLSegmentX1, PWLSegmentX2]')
print([m1, DeltaMin, DeltaMax, PWLSegmentX1, PWLSegmentX2])

PWLSegmentY = [0,k/N]

fig, ax = plt.subplots(1, figsize=(7,3), dpi=200) #(6,3)


# fig = plt.figure(figsize=(6,3), dpi=200) #figure size and resolution  # before: figsize=(6,3)
ax.plot(SampleDist.Xvalues[:2*k],SampleDist.Fvalues[:2*k], linewidth=1.0, linestyle = '--', marker = '', color = 'k')
ax.plot(PWLSegmentX1,PWLSegmentY,   linewidth=1.0, linestyle = '-', marker = '', color = 'k')
ax.plot(PWLSegmentX2,PWLSegmentY,   linewidth=1.0, linestyle = '-', marker = '', color = 'k')
ax.plot(PWLSegmentX3,PWLSegmentY,   linewidth=1.0, linestyle = '-', marker = '', color = 'red')
ax.plot(PWLSegmentX4,PWLSegmentY,   linewidth=1.0, linestyle = '-', marker = '', color = 'red')
ax.plot([m1,m1],PWLSegmentY,   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
PlotBrace(ax, x1 = m1, x2 = m1+DeltaMax, y = 0.64, braceheight = 0.03, bracewidth = 0.5, text = '  $\delta_1^{max}$')
PlotBrace(ax, x1 = m1, x2 = m1+DeltaMin, y = 0.70, braceheight = 0.03, bracewidth = 0.2, text = '  $\delta_1^{min}$')

ax.text(m1, 0.0 , '$\mu_1$', verticalalignment='top', horizontalalignment='center', fontsize=14)

ax.yaxis.set_ticks([ 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xlim([-2.1,10.7]) #ax.set_xlim([-2.1,10.7])
ax.set_ylim([-0.08,0.83])

plt.savefig('MinMaxDeltaParametrizationIllustration.pdf')
# plt.show()

del fig, ax, CharS1, m1, DeltaMax, DeltaMin, PWLSegmentX1, PWLSegmentX2





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
ax.plot([4,9,MaxX],[0.8,1,1],   linewidth=1.0, linestyle = '-', marker = '',color = 'k')
ax.plot([1,4,4,9],[0,0.6,0.8,1],   linewidth=1.0, linestyle = '', marker = '.', markersize=MyMarkerSize, color = 'k')
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





##############################################################
######################### Sixth plot #########################
##############################################################
print("\n =========== Example 6 ============ \n")
print("\n xTVaRillustration.pdf ")

fig, ax = plt.subplots(1, figsize=(8,3), dpi=200) #(6,3)

MinX = -0.5
MaxX = 6.5 #6
#  K =4, x =(1, 4, 4,9), y = (0, 0.6,0.8,1),
MyMarkerSize = 9
ax.plot([MinX,1,4],[0,0,0.6],   linewidth=1.0, linestyle = '-', marker = '', color = 'k')
ax.plot([4,9,10],[0.8,1,1],   linewidth=1.0, linestyle = '-', marker = '',color = 'k')
ax.plot([1,4,4,9],[0,0.6,0.8,1],   linewidth=1.0, linestyle = '', marker = '.', markersize=MyMarkerSize, color = 'k')
ax.plot([4],[0.6],   linewidth=1.0, linestyle = '', marker = '.', markersize=MyMarkerSize*0.5, color = 'w')

# ax.text(m1       , -0.01 , '$\mu_1$', verticalalignment='top', horizontalalignment='center', fontsize=14)
# ax.text(m1-delta1, -0.01 , '$\mu_1-\delta_1$', verticalalignment='top', horizontalalignment='center', fontsize=14)
# ax.text(m1+delta1, -0.01 , '$\mu_1+\delta_1$', verticalalignment='top', horizontalalignment='center', fontsize=14)

TVaR = 1.5
Mean = 3.6
BarHeight = 0.82

ax.plot([TVaR,TVaR],[0,BarHeight],   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([Mean,Mean],[0,BarHeight],   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.text(TVaR, BarHeight , '$-TVaR_{0.2}[G]$', verticalalignment='bottom', horizontalalignment='center', fontsize=14)
ax.text(Mean, BarHeight , '$\mathbb{E}[G]$' , verticalalignment='bottom', horizontalalignment='center', fontsize=14)
ax.fill_between(x=[0,1,2],y1 = [0,0,0.2], y2=[0.2,0.2,0.2], facecolor='k', alpha=0.5, linewidth = 0, color = 'k')

PlotBrace(ax, x1 = TVaR, x2 = Mean, y = BarHeight + 0.16, braceheight = 0.03, bracewidth = 0.5, text = '  $TVaR_{0.2}^{\Delta}(G)$')

# ax.yaxis.set_ticks([ 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xlim([MinX,MaxX])
ax.set_ylim([-0.05,1.15])

# ax.set_xlabel('$x$', fontsize=9)
# ax.set_ylabel('$G(x)$', fontsize=14) #('$G(x)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)
# ax.set_ylabel('$G(x)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=11) #('$G(x)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)

plt.savefig('xTVaRillustration.pdf')
# plt.show()
plt.close()


del fig, ax, MyMarkerSize, TVaR, Mean, BarHeight, MinX




##############################################################
######################### Seventh plot #######################
##############################################################
print("\n =========== Example 7 ============ \n")
print("\n MeanInvarianceExampleDist.pdf ")

k = 6
m1 = np.mean(Sample[:k])
m2 = np.mean(Sample[k:])

int1 = np.sum(Sample[:k])/SampleLength
int2 = np.sum(Sample[k:])/SampleLength
print(int1)
print(int2)


delta1 = 3.3
delta2 = 2.4
PWLapprox = PiecewiseLinearDistribution(
                [m1-delta1,m1+delta1,m2-delta2,m2+delta2],
                [0,k/SampleLength,k/SampleLength,1])
print('PWL approx: X='+str(PWLapprox.Xvalues)+'. Y= '+str(PWLapprox.Fvalues))
print(abs((PWLapprox.Xvalues[0]+PWLapprox.Xvalues[1])/2*(PWLapprox.Fvalues[1]-PWLapprox.Fvalues[0])-int1))
print(abs((PWLapprox.Xvalues[2]+PWLapprox.Xvalues[3])/2*(PWLapprox.Fvalues[3]-PWLapprox.Fvalues[2])-int2))

plt.figure(figsize=(6,3), dpi=200) #figure size and resolution  # before: figsize=(6,3)
plt.plot(SampleDist.Xvalues,SampleDist.Fvalues, linewidth=1.0, linestyle = '--', marker = '', color = 'k')
plt.plot(PWLapprox.Xvalues,PWLapprox.Fvalues,   linewidth=1.0, linestyle = '-', marker = '', color = 'k')
plt.savefig('MeanInvarianceExampleDist.pdf')
# plt.show()
plt.close()

del k, m1, m2, int1, int2, delta1, delta2, PWLapprox


##############################################################
######################### Eight plot #########################
##############################################################
print("\n =========== Example 8 ============ \n")
print("\n MuDeltaParametrizationIllustration.pdf ")

k = 6
m1 = np.mean(Sample[:k])

delta1 = 3.3
PWLSegmentX = [m1-delta1,m1+delta1]
PWLSegmentY = [0,k/N]
print('PWL approx: X='+str(PWLSegmentX)+'. Y= '+str(PWLSegmentY))

fig, ax = plt.subplots(1, figsize=(6,3), dpi=200)
print('[m1, delta1, m1-delta1, m1+delta1]')
print([m1, delta1, m1-delta1, m1+delta1])

# fig = plt.figure(figsize=(6,3), dpi=200) #figure size and resolution  # before: figsize=(6,3)
# ax.plot(SampleDist.Xvalues[:2*k],SampleDist.Fvalues[:2*k], linewidth=1.0, linestyle = '--', marker = '', color = 'k')
ax.plot(PWLSegmentX,PWLSegmentY,   linewidth=1.0, linestyle = '-', marker = '', color = 'k')
ax.plot([m1,m1],PWLSegmentY,   linewidth=1.0, linestyle = ':', marker = '', color = 'k')
ax.plot([m1-delta1,m1-delta1],PWLSegmentY,   linewidth=0.7, linestyle = ':', marker = '', color = 'k')
ax.plot([m1+delta1,m1+delta1],PWLSegmentY,   linewidth=0.7, linestyle = ':', marker = '', color = 'k')
PlotBrace(ax, x1 = m1, x2 = m1+delta1, y = 0.64, braceheight = 0.03, bracewidth = 0.5, text = '$\delta_1$')

ax.text(m1       , -0.01 , '$\mu_1$', verticalalignment='top', horizontalalignment='center', fontsize=14)
ax.text(m1-delta1, -0.01 , '$\mu_1-\delta_1$', verticalalignment='top', horizontalalignment='center', fontsize=14)
ax.text(m1+delta1, -0.01 , '$\mu_1+\delta_1$', verticalalignment='top', horizontalalignment='center', fontsize=14)

ax.yaxis.set_ticks([ 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_xlim([0.2,7.9])
ax.set_ylim([-0.09,0.75])

plt.savefig('MuDeltaParametrizationIllustration.pdf')
# plt.show()
plt.close()

del fig, ax, k, m1, delta1


##############################################################
######################### Ninth plot #########################
##############################################################
print("\n =========== Example 9 ============ \n")
print("\n FullAlgorithmExampleFreqSev.pdf ")


np.random.seed(1232121334)

n = 1000
#parameters: see xAct pricing structure with xActID 6338199
FSSample = XLsimulations(SampleSize = n, PoissonLambda = 2, ParetoX0 = 10, ParetoAlpha = 2.5, Deductible = 12, Limit = 10, AggregateLimit = 30)
Accuracy = 0.001

CompressedSample = PWLcompressor(FSSample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False,
                                 Accuracy = Accuracy, RelativeAtomDetectionThreshold = 0.001)
Step3 = CompressedSample.plot('g', ShowPlot = False)


fig, ax = plt.subplots(1, figsize=(7,3), dpi=200) #(6,3)
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

MyMark_inset(ax, axIns, loc11=1, loc12=1, loc21=2, loc22=3, fc="none", ec="0.4", zorder = -10, linewidth=0.4)

# plt.show()
plt.savefig('FullAlgorithmExampleFreqSevNOPWL.png')

for thisAX in [ax,axIns]:
    thisAX.plot(Step3['PWLX'],Step3['PWLY'], linewidth=1.3, linestyle = '--', marker = 'o', markersize = 5, color = 'k', fillstyle = 'none')

plt.savefig('FullAlgorithmExampleFreqSev.pdf')
plt.close()


del fig, ax, axIns, FSSample, Accuracy, zoom, CompressedSample



##############################################################
######################### Tenth plot #########################
##############################################################
print("\n =========== Example 10=========== \n")
print("\n ErrorConvergenceMaximum.pdf ")
# print("\n ErrorConvergenceMean.pdf ")

def PlotConvergenceInfo(DictWithInfoList, color = 'k', Variable='Maximum', Name=None):
    # Xpoints = list(range(1,len(self.TreeCollection)+1))# [1,2,3,4,...,n]    np.linspace(1,len(self.TreeCollection),num=len(self.TreeCollection))
    Xpoints = [x['Iteration'] for x in DictWithInfoList]
    Ypoints = [x[Variable] for x in DictWithInfoList]
    print(Variable)
    print(Ypoints)

    plt.figure(figsize=(6,3), dpi=200)
    #plt.axes.Axes(set_yscale="log")
    plt.semilogy(Xpoints,Ypoints, linewidth=1.0, linestyle = '-', marker = 'o', color = color)
    # plt.axis([-0.5, len(Xpoints)-0.5, min(Ypoints)/2, max(Ypoints)*2])
    plt.axis([-0.5, len(Xpoints)-0.5, min(Ypoints)/2, 0.11])

    plt.savefig(Name)
    plt.close()
    #    plt.show()  # plt.show(block=False) # plt.hold(True)

RecalculateConvergenceErrorGraph = True
if RecalculateConvergenceErrorGraph:
    np.random.seed(12121334)

    n = 1000000
    [mu, sigma] = LognormalParameters(mean = 10, CoV = 0.1)
    Sample = np.random.lognormal(mean = mu, sigma = sigma, size = (n,))

    Convergence = ConvergeRateCalculator(Sample, NrOfIterations = 15)
    Convergence.print()
    DictWithInfo = Convergence.TreeCollection
else:
    #output as of 10.6.2015, 14:32
    DictWithInfo = [{'Length': 1, 'Iteration': 0, 'Maximum': 0.034781385836612493, 'Mean': 0.034781385836612493},
                    {'Length': 2, 'Iteration': 1, 'Maximum': 0.020672084831889675, 'Mean': 0.012161826397513396},
                    {'Length': 4, 'Iteration': 2, 'Maximum': 0.0089248727136497334, 'Mean': 0.0031071628876866699},
                    {'Length': 8, 'Iteration': 3, 'Maximum': 0.0047821128371561212, 'Mean': 0.00080491450622862456},
                    {'Length': 16, 'Iteration': 4, 'Maximum': 0.0025747359468483336, 'Mean': 0.00021248819915866221},
                    {'Length': 32, 'Iteration': 5, 'Maximum': 0.0013372083334321724, 'Mean': 5.4476999775167857e-05},
                    {'Length': 64, 'Iteration': 6, 'Maximum': 0.00079414784539554235, 'Mean': 1.5597543815910072e-05},
                    {'Length': 127, 'Iteration': 7, 'Maximum': 0.00038262004867415452, 'Mean': 3.9157798461538413e-06},
                    {'Length': 248, 'Iteration': 8, 'Maximum': 0.00025467460705021585, 'Mean': 1.2694457828483434e-06},
                    {'Length': 470, 'Iteration': 9, 'Maximum': 0.00012085511497015614, 'Mean': 3.2174656498323416e-07},
                    {'Length': 888, 'Iteration': 10, 'Maximum': 8.8650914647581889e-05, 'Mean': 1.1946548561958203e-07},
                    {'Length': 1611, 'Iteration': 11, 'Maximum': 4.0121366588056844e-05, 'Mean': 3.1587490977872294e-08},
                    {'Length': 2736, 'Iteration': 12, 'Maximum': 2.1453533250813053e-05, 'Mean': 1.6443070028419378e-08},
                    {'Length': 4435, 'Iteration': 13, 'Maximum': 9.3323167800069769e-06, 'Mean': 5.2121177384611498e-09},
                    {'Length': 6800, 'Iteration': 14, 'Maximum': 3.6562569856380379e-06, 'Mean': 1.8908330110434078e-09}]

PlotConvergenceInfo(DictWithInfo, Variable='Maximum', Name="ErrorConvergenceMaximum.pdf")
# PlotConvergenceInfo(Convergence.TreeCollection, Variable='Mean', Name="ErrorConvergenceMean.pdf")

del PlotConvergenceInfo

##############################################################
######################### Eleventh example ###################
##############################################################
print("\n =========== Example 11=========== \n")

Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
SC = SampleCharacteristics(Sample, Accuracy = 100)
Sol = SC.FindBestSolutionLine(0,9)
NonEpsilonPart = (Sol.AdmissibilityInequality_Left + Sol.AdmissibilityInequality_Right)/2
IntegralDiffWithRegressionDelta = np.abs(Sol.Delta_Regression*Sol.AdmissibilityInequality_Middle - NonEpsilonPart)
IntegralDiffWithRegressionDelta /= len(Sample)
print('IntegralDiffWithRegressionDelta')
print(IntegralDiffWithRegressionDelta) #print('Iteration '+str(i)+' : '+ ', '.join(["%.3f" % y for y in Basis]))
print('Sol.MaximumDifference')
print(Sol.MaximumDifference)

