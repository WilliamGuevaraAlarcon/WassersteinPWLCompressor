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
        IncreasingIntegers = np.linspace(0, SampleSize-1, num = SampleSize, dtype = int) # 0, 1, 2, ... , SampleSize-1

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

    # def xtVaR(self,q):
    #     """
    #     returns lower xtVaR_q
    #     """
    #     SampleIndex = round(q*self.SampleSize)
    #     assert SampleIndex > 0
    #     tVaR = self.CumSum_Sample[SampleIndex-1]/SampleIndex
    #     return self.Mean - tVaR
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

    #Delta methods
    def CalculateRegressionDelta(self,SampleSet_Start, SampleSet_End):
        LocalMean = self.LocalMean(SampleSet_Start, SampleSet_End)
        Y_Right = (SampleSet_End+1)/self.SampleSize
        Y_Left = SampleSet_Start/self.SampleSize
        Regression_Delta_Part1 = -3*LocalMean*(Y_Right+Y_Left)/(Y_Right-Y_Left)
        Regression_Delta_Part2 = 6/((self.SampleSize*(Y_Right-Y_Left))**2)*self.Regression_PartialSum(SampleSet_Start, SampleSet_End)
        Delta_Regression = Regression_Delta_Part1 + Regression_Delta_Part2
        return Delta_Regression
    #
    # def CalculateMinWassersteinDelta(self,SampleSet_Start, SampleSet_End):
    #     LocalMean = self.LocalMean(SampleSet_Start, SampleSet_End)
    #     SegmentSize = SampleSet_End - SampleSet_Start + 1
    #
    #     Multiplier = (2 * (np.linspace(SampleSet_Start + 1 / 2, SampleSet_End + 1 / 2, SampleSet_End + 1 - SampleSet_Start) - SampleSet_Start) / (SampleSet_End + 1 - SampleSet_Start) - 1)
    #
    #     PossibleDeltas      = (self.Sample[SampleSet_Start:SampleSet_End + 1] - LocalMean)/Multiplier
    #     PossibleWasserstein = [np.sum(np.abs(self.Sample[SampleSet_Start:SampleSet_End + 1] - LocalMean - d * Multiplier)) / self.SampleSize for d in PossibleDeltas]
    #
    #     # if (SegmentSize & 1):
    #     #     midpoint = int((SegmentSize-1)/2)
    #     #
    #     #     ModDeltas = np.delete(PossibleDeltas, midpoint)
    #     #     PossibleDeltas = ModDeltas
    #     #     del PossibleWasserstein[midpoint]
    #
    #     MinIndex = np.nanargmin(PossibleWasserstein)
    #     MinWasser = PossibleWasserstein[MinIndex]
    #
    #     MinIndex = np.argwhere(PossibleWasserstein ==  MinWasser)
    #
    #     DeltaMin = np.mean(np.unique(PossibleDeltas[MinIndex]))
    #
    #     return DeltaMin

    def CalculateMinWassersteinDelta(self,SampleSet_Start, SampleSet_End):


        LocalMean = self.LocalMean(SampleSet_Start, SampleSet_End)
        SegmentSize = int(SampleSet_End - SampleSet_Start + 1)

        Multiplier = (2 * (np.linspace(SampleSet_Start + 1 / 2, SampleSet_End + 1 / 2,
                                       SampleSet_End + 1 - SampleSet_Start) - SampleSet_Start) / (
                      SampleSet_End + 1 - SampleSet_Start) - 1)

        def WDdiff(d):
            return -np.sum(Multiplier * np.sign(self.Sample[SampleSet_Start:SampleSet_End + 1] - LocalMean - d * Multiplier))

        def WD(d):
            return np.sum(np.abs(self.Sample[SampleSet_Start:SampleSet_End + 1] - LocalMean - d * Multiplier ))


        PossibleDeltas = (self.Sample[SampleSet_Start:SampleSet_End + 1] - LocalMean) / Multiplier

        if (SegmentSize & 1):
            PossibleDeltas[(SegmentSize-1) / 2] = PossibleDeltas[0]

        SortPermPossibleDeltas = np.argsort(PossibleDeltas)
        SortedPossibleDeltas = PossibleDeltas[SortPermPossibleDeltas]
        IndLeft = 0
        IndRight = SegmentSize-1

        while(IndRight-IndLeft) > 1:
            mid = int((IndLeft + IndRight) / 2)
            if WDdiff(SortedPossibleDeltas[mid]) < 0:
                IndLeft = mid
            else:
                IndRight = mid

        DeltaLeft  = SortedPossibleDeltas[IndLeft]
        DeltaRight = SortedPossibleDeltas[IndRight]

        WLeft = WD(DeltaLeft)
        WRight = WD(DeltaRight)

        if round(WLeft,8) == round(WRight,8):
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
        D = (1 / self.SampleSize) * (
        1 / 3 * (delta1alt ** 2 * (Zet - SampleSet_Start) + delta2alt ** 2 * (SampleSet_End + 1 - Zet))
        - (mu1alt ** 2 * (Zet - SampleSet_Start) + mu2alt ** 2 * (SampleSet_End + 1 - Zet))
        + 2 * (delta1alt * mu1alt * (Zet + SampleSet_Start) + delta2alt * mu2alt * (Zet + SampleSet_End + 1))
        - 4 * (delta1alt * Regression_PartialSum1 / (Zet - SampleSet_Start) + delta2alt * Regression_PartialSum2 / (SampleSet_End + 1 - Zet)))

        MinIndex = np.argmin(D)
        return (SampleSet_Start + MinIndex)

    def BisectionOriginal(self, SampleSet_Start, SampleSet_End, SegmentSize, LocalMean, Delta_Regression):
        PartialSumPart = self.CumSum_Sample[SampleSet_Start:SampleSet_End + 1] - self.CumSum_Sample_m1(SampleSet_Start)
        LowerIntegralSum = (self.IncreasingIntegers[SampleSet_Start:SampleSet_End + 1] - self.IncreasingIntegers[
            SampleSet_Start] + 1)  # cumsum(np.ones(SampleSet_End - SampleSet_Start + 1))
        MeanIntegralPart = LowerIntegralSum * LocalMean
        NonEpsilonPart = PartialSumPart - MeanIntegralPart
        Ineq_Middle = (
        (self.IncreasingIntegers[SampleSet_Start:SampleSet_End + 1] - self.IncreasingIntegers[SampleSet_Start] + 1) *
        (self.IncreasingIntegers[SampleSet_Start:SampleSet_End + 1] - self.IncreasingIntegers[
            SampleSet_End]) / SegmentSize)
        IntegralDiffWithRegressionDelta = np.abs(Delta_Regression * Ineq_Middle - NonEpsilonPart)

        MaxIndex = np.argmax(IntegralDiffWithRegressionDelta)
        return (SampleSet_Start + MaxIndex)

    # def BisectionWasserstein(self, SampleSet_Start, SampleSet_End):
    #     Zet = self.IncreasingIntegers[SampleSet_Start + 1: SampleSet_End + 1]
    #     res = []
    #     for z in Zet:
    #         res.append(sum(np.abs(self.Sample[SampleSet_Start:z] -
    #                               (self.LocalMean(SampleSet_Start, z) + self.CalculateRegressionDelta(SampleSet_Start,
    #                                                                                                   z) * (
    #                                2 * (z - 1 / 2 - SampleSet_Start) / (
    #                                    z + 1 - SampleSet_Start) - 1))) / self.SampleSize))
    #
    #     MinIndex = np.argmin(res)
    #
    #     return (MinIndex + SampleSet_Start)

    def FindBestSolutionLine(self, SampleSet_Start, SampleSet_End, Bisection = 'Original'):

        if (SampleSet_Start == SampleSet_End) or (self.Sample[SampleSet_Start] == self.Sample[SampleSet_End]): #subsample has size one or solution line segment represents a jump part
            # return True, 0.0, 0.0, LinearFunction(self.Sample[SampleSet_Start],0.0)
            return Segment(
                SampleSet_Start         = SampleSet_Start,
                SampleSet_End           = SampleSet_End,
                SampleSize              = self.SampleSize,
                Mean                    = self.Sample[SampleSet_Start],
                Delta_Regression        = 0.0,
                BestBisectionPoint      = None)
        else: # subsample has size larger than one and is not jump
            # StartTime = clock()

            # construct vectors defining the inequalities to be solved
            SegmentSize      = (SampleSet_End + 1 - SampleSet_Start)
            LocalMean        = self.LocalMean(SampleSet_Start, SampleSet_End)

            # calculate regression delta
            Delta_Regression = self.CalculateRegressionDelta(SampleSet_Start, SampleSet_End)
            #Delta_Regression = self.CalculateMinWassersteinDelta(SampleSet_Start, SampleSet_End)

            ######################New Bisection
            if Bisection == 'OLS':
                BestBisectionPoint = self.BisectionOLS(SampleSet_Start, SampleSet_End, SegmentSize)

            if Bisection == 'Original':
                BestBisectionPoint = self.BisectionOriginal(SampleSet_Start, SampleSet_End, SegmentSize, LocalMean, Delta_Regression)

            # if Bisection == 'Wasserstein_low':
            #     BestBisectionPoint = self.BisectionWasserstein(SampleSet_Start, SampleSet_End)
            #MinimumDifference = IntegralDiffWithRegressionDelta[MinIndex]/self.SampleSize

            # EndTime = clock()
            # print('Time required for Analysis of sement from '+str(SampleSet_Start)+' to '+str(SampleSet_End)+': '+str(EndTime-StartTime)+' seconds')




            return Segment(
                SampleSet_Start         = SampleSet_Start,
                SampleSet_End           = SampleSet_End,
                SampleSize              = self.SampleSize,
                Mean                    = LocalMean,
                Delta_Regression        = Delta_Regression,
                BestBisectionPoint      = BestBisectionPoint,
                Sample                  = self.Sample)
