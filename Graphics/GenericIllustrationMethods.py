
import numpy as np
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, BboxPatch, BboxConnector
from matplotlib.transforms import Bbox, TransformedBbox, IdentityTransform

def MyMark_inset(parent_axes, inset_axes, loc11, loc12, loc21, loc22, **kwargs):
   rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

   pp = BboxPatch(rect, **kwargs)
   parent_axes.add_patch(pp)

   p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc11, loc2=loc12, **kwargs)
   inset_axes.add_patch(p1)
   p1.set_clip_on(False)
   p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc21, loc2=loc22, **kwargs)
   inset_axes.add_patch(p2)
   p2.set_clip_on(False)

   return pp, p1, p2


def BoundPlot(thisax, x, y):
    thisax.plot(x, y, linewidth=.5, linestyle = '--', marker = '', color = 'k', dashes = (1,3))


def Characteristics(Sol):
    LoReg = Sol.Calculate_StartXFromDelta(Sol.Delta_Regression)
    LoSel = Sol.Calculate_StartXFromDelta(Sol.Delta_Selected)
    UpReg = Sol.Calculate_EndXFromDelta(Sol.Delta_Regression)
    UpSel = Sol.Calculate_EndXFromDelta(Sol.Delta_Selected)
    return dict(LoReg = LoReg, UpReg = UpReg, LoSel = LoSel, UpSel = UpSel)

def SetAxisBoundLineWidth(ThisAx, linewidth = 0.7):
    for side in ['top','bottom','left','right']:
        ThisAx.spines[side].set_linewidth(linewidth)

def PlotBrace(ax, x1 = 0.0, x2 = 0.0, y = 0.0, braceheight = 0.0, bracewidth = 0.0, text = 'bla', textdistance = 1.0, upper=True, fontsize=14, color = "black"):
    m = (x1+x2)/2
    if upper:
        e1 = patches.Arc((x1+bracewidth, y-braceheight), 2*bracewidth, 2*braceheight,angle=0, linewidth=1, fill=False, theta1 = 90, theta2 = 180, color = color)
        e2 = patches.Arc((x2-bracewidth, y-braceheight), 2*bracewidth, 2*braceheight,angle=0, linewidth=1, fill=False, theta1 = 0, theta2 = 90, color = color)
        e3 = patches.Arc((m -bracewidth, y+braceheight), 2*bracewidth, 2*braceheight,angle=0, linewidth=1, fill=False, theta1 = 270, theta2 = 0, color = color)
        e4 = patches.Arc((m +bracewidth, y+braceheight), 2*bracewidth, 2*braceheight,angle=0, linewidth=1, fill=False, theta1 = 180, theta2 = 270, color = color)
    else:
        e1 = patches.Arc((x1+bracewidth, y+braceheight), 2*bracewidth, 2*braceheight,angle=0, linewidth=1, fill=False, theta1 = 180, theta2 = 270, color = color)
        e2 = patches.Arc((x2 -bracewidth, y+braceheight), 2*bracewidth, 2*braceheight,angle=0, linewidth=1, fill=False, theta1 = 270, theta2 = 0, color = color)
        e3 = patches.Arc((m -bracewidth, y-braceheight), 2*bracewidth, 2*braceheight,angle=0, linewidth=1, fill=False, theta1 = 0, theta2 = 90, color = color)
        e4 = patches.Arc((m +bracewidth, y-braceheight), 2*bracewidth, 2*braceheight,angle=0, linewidth=1, fill=False, theta1 = 90, theta2 = 180, color = color)

    for patch in [e1,e2,e3,e4]:
        ax.add_patch(patch)
    ax.plot([x1+bracewidth,m-bracewidth], [y,y], linewidth=1.0, linestyle = '-', marker = '', color = color)
    ax.plot([m+bracewidth,x2-bracewidth], [y,y], linewidth=1.0, linestyle = '-', marker = '', color = color)
    if upper:
        ax.text(m, y+braceheight*textdistance, text,
        verticalalignment='bottom', horizontalalignment='center', fontsize=fontsize, color = color)
    else:
        ax.text(m, y-braceheight*textdistance, text,
        verticalalignment='top', horizontalalignment='center', fontsize=fontsize, color = color)

def LognormalParameters(mean = 1.0, CoV = 0.1):
    sigma2  = np.log(CoV*CoV+1)
    LNsigma = np.sqrt(sigma2)
    LNmu    = np.log(mean)-sigma2/2
    return [LNmu,LNsigma]

def XLsimulations(SampleSize = 10,
                    PoissonLambda = 1.0,
                    ParetoX0 = 10,
                    ParetoAlpha = 1.5,
                    Deductible = 12,
                    Limit = 10,
                    AggregateLimit = 40,
                    Verbose = False):
      #generate poisson numbers
      N = np.random.poisson(lam=PoissonLambda, size=SampleSize)
      totalN = np.sum(N)
      #generate single losses
      Losses = np.random.uniform(size=totalN)
      Losses = ParetoX0*np.power(1-Losses,-1.0/ParetoAlpha)

      #apply limit and deductible
      LossToLayer = np.minimum(np.maximum( Losses - Deductible, 0), Limit)

      #accumulate
      NCumSum = np.cumsum(N)
      NCumSumShifted = np.insert(NCumSum[0:-1],0,0.0)
      LCumSum = np.insert(np.cumsum(LossToLayer),0,0.0)
      Aggregate = LCumSum[NCumSum] - LCumSum[NCumSumShifted]

      #apply AAL
      Aggregate = np.minimum(Aggregate, AggregateLimit)

      #get rid of roundoff errors at atoms
      Epsilon = 1e-12
      if AggregateLimit < np.infty:
          MaxAtom = int(AggregateLimit/Limit)
          for i in range(MaxAtom+1):
              AtomVal = i*Limit
              Aggregate[np.logical_and(AtomVal < Aggregate,Aggregate < AtomVal +Epsilon)] = AtomVal

      #verbosity
      if Verbose:
          print(N)
          print('sum = ' + str(totalN))
          print('Losses')
          print(Losses)
          print('LossToLayer')
          print(LossToLayer)
          print(Aggregate)
      return Aggregate
