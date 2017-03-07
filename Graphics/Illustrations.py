
import numpy as np
import matplotlib.pyplot as plt
from wasserstein_pwl_core.compressor import PWLcompressor
from wasserstein_pwl_core.pwl_distribution import PiecewiseLinearDistribution, EmpiricalDistributionFromSample
from wasserstein_pwl_core.sample_characteristics import SampleCharacteristics
from matplotlib import rcParams

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
print("\n =========== Wasserstein distance ============ \n")


Sample = np.asarray([ 1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]) #sorted!
SampleDist = EmpiricalDistributionFromSample(Sample)
PWLAprox = PiecewiseLinearDistribution([0.8, 7.4, 12.9, 17.7],[0, 0.6, 0.6, 1])

fig, ax = plt.subplots(1, figsize=(6,3), dpi=200)

ax.set_xlim([0,20])
ax.set_ylim([0.0,1.0])

x = np.linspace(0,20,1000)
y1 = SampleDist.cdf(x)
y2 = PWLAprox.cdf(x)

ax.plot(SampleDist.Xvalues, SampleDist.Fvalues, linewidth=1.0, linestyle = '--', marker = '', color = 'k')
ax.plot(x ,y2, linewidth=1.0, linestyle = '-', marker = '', color = 'k')

#ax.plot(CompressedSample['PWLX'],CompressedSample['PWLY'], linewidth=1.0, linestyle = '-', marker = '', color = 'k', label='Some PWL distribution')
#ax.fill_between(, y1 = SampleDist.Fvalues, y2 = CompressedSample['PWLY'], facecolor='k', alpha=0.5, linewidth = 0, color = 'k')
plt.fill_between(x, y1, y2, color='grey', alpha='0.5')
#ax.fill_between()
#
#  legend = ax.legend(loc='lower right', shadow=True)

plt.savefig('Wasserstein.pdf')
plt.close()



##############################################################
#########################  #########################
##############################################################
print("\n =========== Theorem 3.2 ============ \n")
print("\n WassersteinDistance.pdf ")

######### 'WassersteinDistance.pdf'

Sample = np.asarray([1.0, 1.1, 1.2, 1.6, 4.3, 4.5, 4.6, 6, 6.1, 6.6,
          7.1, 13, 13.4, 16, 18.8, 22, 30, 32, 39, 40])#sorted!
SampleDist = EmpiricalDistributionFromSample(Sample)

Approx = PWLcompressor(Sample, Accuracy= 0.65)

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
PlotBrace(ax, x1 = 17.85, x2 = 18.8, y = 0.7, braceheight = 0.005, bracewidth = 0.5, text = r'$\delta^*_s-|X_{(i)}-G^{\leftarrow}\left(\frac{i-1/2}{n}\right)|$', upper=False, fontsize=8)
#PlotBrace(ax, x1 = 17.85, x2 = 19.06, y = 0.69, braceheight = 0.03, bracewidth = 0.5, text = r'$\delta^*_s$', upper=False, fontsize=10)

# ax.set_xlabel('$x$', fontsize=9)
ax.set_ylabel('$G(t)$', fontsize=14) #('$G(x)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)
# ax.set_ylabel('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=11) #('$G(t)$ with $G=PWL((1,4,4,9),(0,0.6,0.8,1))$', fontsize=9)

plt.savefig('WassersteinDistance.pdf')
# plt.show()
plt.close()

del fig, ax, MyMarkerSize

