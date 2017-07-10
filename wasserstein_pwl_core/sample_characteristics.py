__author__ = 'chuarph'

import numpy as np
from .segment import Segment

class SampleCharacteristics:
    """
    An object of this class contains various statistics of a sample.
    This object can then be used to calculate delta_Reg as explained in the paper
    """
    Sample      = None
    SampleSize  = 0
    Mean        = 0
    Minimum     = 0
    Maximum     = 0

    IncreasingIntegers      = None
    CumSum_Sample           = None
    Regression_Cumsum       = None

    def __init__(self, Sample):
        """
        calculate all statistics of the sample required for compression. is done only once.
        """

        Sample = np.sort(np.asarray(Sample, dtype = np.float64))
        assert Sample.ndim == 1

        SampleSize = Sample.size
        assert SampleSize > 1

        # regression precalculations
        IncreasingIntegers = np.linspace(0, SampleSize-1, num = SampleSize) # 0, 1, 2, ... , SampleSize-1

        # sample regression precalculations
        CumSum_Sample           = np.cumsum(Sample)         # CumSum_Sample[k]     = Sample[0] + ... + Sample[k]
        Mean                    = np.float64(CumSum_Sample[-1]/SampleSize) # CumSum_Sample[SampleSize-1]/SampleSize = (Sample[0] + ... + Sample[SampleSize-1])/SampleSize
        Minimum                 = np.min(Sample)
        Maximum                 = np.max(Sample)

        Regression_Cumsum = np.cumsum( (IncreasingIntegers + 0.5)*Sample ) # incex starts from 0, therefore +0.5 and not -0.5

        #assign properties
        self.Sample                  = Sample
        self.SampleSize              = SampleSize
        self.Mean                    = Mean
        self.IncreasingIntegers      = IncreasingIntegers
        self.CumSum_Sample           = CumSum_Sample
        self.Regression_Cumsum       = Regression_Cumsum
        self.Minimum                 = Minimum
        self.Maximum                 = Maximum
        self.EstimatedWasserstein    = self.EstimatedWassersteinError(Sample)

    def CumSum_Sample_m1(self,index):
        # returns self.CumSum_Sample[index - 1]
        if index == 0:
            return 0.0
        else:
            return self.CumSum_Sample[index - 1]

    def Regression_PartialSum(self, SampleSet_Start, SampleSet_End):
        if SampleSet_Start == 0:
            return self.Regression_Cumsum[SampleSet_End]
        else:
            return self.Regression_Cumsum[SampleSet_End] - self.Regression_Cumsum[SampleSet_Start - 1]

    def LocalMean(self, SampleSet_Start, SampleSet_End):
        return (self.CumSum_Sample[SampleSet_End] - self.CumSum_Sample_m1(SampleSet_Start))/(SampleSet_End + 1 - SampleSet_Start)

    def EstimatedWassersteinError(self, Sample):
        # calculates sqrt(2/pi)*\int_{-inf}^{+inf} sqrt( F_n(x)*(1-F_n(x)) ) dx
        # for F_n empirical distribution of sample
        empdist = np.linspace(1 / self.SampleSize, 1 - 1 / self.SampleSize, self.SampleSize - 1)
        dx = np.diff(Sample)
        integrand = np.sqrt(empdist * (1 - empdist))
        integral = np.sum(integrand * dx)
        integral *= np.sqrt(2 / (np.pi*self.SampleSize))
        return integral

    def WD(self, SampleSet_Start, SampleSet_End, Multiplier, delta):
        return np.sum(np.abs(self.Sample[SampleSet_Start:SampleSet_End + 1] - self.LocalMean(SampleSet_Start, SampleSet_End) - delta * Multiplier ))

    def WDdiff(self, SampleSet_Start, SampleSet_End, Multiplier, delta):
        return -np.sum(Multiplier * np.sign(self.Sample[SampleSet_Start:SampleSet_End + 1] - self.LocalMean(SampleSet_Start, SampleSet_End) - delta * Multiplier))

    def Multiplier(self, SampleSet_Start, SampleSet_End):
        SegmentSize = int(SampleSet_End + 1 - SampleSet_Start)
        return (2 * (np.linspace(SampleSet_Start + 1 / 2, SampleSet_End + 1 / 2, SegmentSize) - SampleSet_Start)/(SegmentSize) - 1)

    def CalculateMinWassersteinDelta(self,SampleSet_Start, SampleSet_End):

        LocalMean = self.LocalMean(SampleSet_Start, SampleSet_End)
        SegmentSize = int(SampleSet_End + 1 - SampleSet_Start)

        Multiplier = self.Multiplier(SampleSet_Start, SampleSet_End)

        PossibleDeltas = (self.Sample[SampleSet_Start:SampleSet_End + 1] - LocalMean) / Multiplier

        if (SegmentSize & 1):
            PossibleDeltas[int((SegmentSize-1) / 2)] = PossibleDeltas[0]

        SortPermPossibleDeltas = np.argsort(PossibleDeltas)
        SortedPossibleDeltas = PossibleDeltas[SortPermPossibleDeltas]
        IndLeft = 0
        IndRight = SegmentSize-1

        while(IndRight-IndLeft) > 1:
            mid = int((IndLeft + IndRight) / 2)
            if round(self.WDdiff(SampleSet_Start, SampleSet_End, Multiplier, SortedPossibleDeltas[mid]), 10) < 0:
                IndLeft = mid
            else:
                IndRight = mid

        DeltaLeft  = SortedPossibleDeltas[IndLeft]
        DeltaRight = SortedPossibleDeltas[IndRight]

        WLeft = self.WD(SampleSet_Start, SampleSet_End, Multiplier, DeltaLeft)
        WRight = self.WD(SampleSet_Start, SampleSet_End, Multiplier, DeltaRight)

        if abs(WLeft - WRight) <= (WLeft + WRight) * 1e-8:
            DeltaMin = (DeltaLeft + DeltaRight)/2
        elif WLeft < WRight:
            DeltaMin = DeltaLeft
        elif WLeft > WRight:
            DeltaMin = DeltaRight

        return DeltaMin

    #Bisection methods
    def BisectionOLS(self, SampleSet_Start, SampleSet_End, SegmentSize):
        SubsetSum = self.CumSum_Sample[SampleSet_End] - self.CumSum_Sample_m1(SampleSet_Start)
        mu1alt = self.CumSum_Sample[SampleSet_Start:SampleSet_End] - self.CumSum_Sample_m1(SampleSet_Start)
        mu2alt = SubsetSum - mu1alt
        LeftSegmentLength = np.linspace(1, SegmentSize - 1, SegmentSize - 1)
        mu1alt /= LeftSegmentLength
        mu2alt /= (SegmentSize - LeftSegmentLength)

        Y_Left1 = SampleSet_Start / self.SampleSize
        Y_Right1 = Y_Left1 + LeftSegmentLength / self.SampleSize
        Y_Left2 = Y_Right1
        Y_Right2 = (SampleSet_End + 1) / self.SampleSize

        Regression_PartialSum1 = self.Regression_Cumsum[SampleSet_Start:SampleSet_End].copy()
        Regression_PartialSum2 = self.Regression_Cumsum[SampleSet_End] - Regression_PartialSum1
        if SampleSet_Start > 0:
            Regression_PartialSum1 -= self.Regression_Cumsum[SampleSet_Start - 1]

        delta1alt = -3 * mu1alt * (Y_Right1 + Y_Left1) / (Y_Right1 - Y_Left1) + 6 / (
        (self.SampleSize * (Y_Right1 - Y_Left1)) ** 2) * Regression_PartialSum1
        delta2alt = -3 * mu2alt * (Y_Right2 + Y_Left2) / (Y_Right2 - Y_Left2) + 6 / (
        (self.SampleSize * (Y_Right2 - Y_Left2)) ** 2) * Regression_PartialSum2

        Zet = self.IncreasingIntegers[SampleSet_Start + 1: SampleSet_End + 1]
        D = (1 / self.SampleSize) * (delta1alt * (mu1alt*(Zet + SampleSet_Start) - 2 * Regression_PartialSum1 / (Zet - SampleSet_Start))
        + delta2alt * (mu2alt * (Zet + SampleSet_End + 1) - 2 * Regression_PartialSum2 / (SampleSet_End + 1 - Zet))
        - mu1alt**2 * (Zet - SampleSet_Start) - mu2alt**2 * (SampleSet_End + 1 - Zet))

        MinIndex = np.argmin(D)
        return (SampleSet_Start + MinIndex)

    def FindBestSolutionLine(self, SampleSet_Start, SampleSet_End):

        if (SampleSet_Start == SampleSet_End) or (self.Sample[SampleSet_Start] == self.Sample[SampleSet_End]): #subsample has size one or solution line segment represents a jump part
            # return True, 0.0, 0.0, LinearFunction(self.Sample[SampleSet_Start],0.0)
            return Segment(
                SampleSet_Start         = SampleSet_Start,
                SampleSet_End           = SampleSet_End,
                SampleSize              = self.SampleSize,
                Mean                    = self.Sample[SampleSet_Start],
                Delta_Regression        = 0.0,
                BestBisectionPoint      = None,
                Sample                  = self.Sample)
        else: # subsample has size larger than one and is not jump
            # StartTime = clock()

            # construct vectors defining the inequalities to be solved
            SegmentSize      = (SampleSet_End + 1 - SampleSet_Start)

            # calculate regression delta
            Delta_Regression = self.CalculateMinWassersteinDelta(SampleSet_Start, SampleSet_End)
            BestBisectionPoint = self.BisectionOLS(SampleSet_Start, SampleSet_End, SegmentSize)

            # EndTime = clock()
            # print('Time required for Analysis of sement from '+str(SampleSet_Start)+' to '+str(SampleSet_End)+': '+str(EndTime-StartTime)+' seconds')

            return Segment(
                SampleSet_Start         = SampleSet_Start,
                SampleSet_End           = SampleSet_End,
                SampleSize              = self.SampleSize,
                Mean                    = self.LocalMean(SampleSet_Start, SampleSet_End),
                Delta_Regression        = Delta_Regression,
                BestBisectionPoint      = BestBisectionPoint,
                Sample                  = self.Sample)
