__author__ = 'chuarph'

import numpy as np
from .solve_simple_linear_inequality_list import SolveSimpleLinearInequailityList
from .solution_line import SolutionLine

class SampleCharacteristics:
    """
    An object of this class contains various statistics of a sample.
    This object can then be used to calculate the delta_min, delta_max, and delta_Reg as explained in the paper
    """
    Sample      = None
    SampleSize  = 0
    Mean        = 0

    Accuracy    = 0

    IncreasingIntegers      = None
    CumSum_Sample           = None
    Regression_Cumsum       = None
    ScaledXTVaR             = None

    def __init__(self, Sample, Accuracy = 0.001):
        """
        calculate all statistics of the sample required for compression. is done only once.
        """

        Sample = np.sort(np.asarray(Sample, dtype = np.float64))
        assert Sample.ndim == 1

        SampleSize = Sample.size
        assert SampleSize > 1

        # regression precalculations
        IncreasingIntegers = np.linspace(0,SampleSize-1,num = SampleSize) # 0, 1, 2, ... , SampleSize-1

        # sample regression precalculations
        CumSum_Sample           = np.cumsum(Sample)         # CumSum_Sample[k]     = Sample[0] + ... + Sample[k]
        Mean                    = np.float64(CumSum_Sample[-1]/SampleSize) # CumSum_Sample[SampleSize-1]/SampleSize = (Sample[0] + ... + Sample[SampleSize-1])/SampleSize
        Minimum                 = np.min(Sample)
        Maximum                 = np.max(Sample)
        #scaled xtVaR:    ScaledXTVaR[i] = ( (i+1)/n*xtVaR_{(i+1)/n}(F) )    (?)
        ScaledXTVaR = IncreasingIntegers*Mean + Mean - CumSum_Sample #[mu-X[0], 2*mu-X[0]-X[1], 3*mu-X[0]-X[1]-X[2],...]
        ScaledXTVaR[-1] = 0.0 #overwrite to avoid roundoff error issues

        Regression_Cumsum = np.cumsum( (IncreasingIntegers+0.5)*Sample ) # incex starts from 0, therefore +0.5 and not -0.5

        #assign properties
        self.Sample                  = Sample
        self.Accuracy                = Accuracy
        self.SampleSize              = SampleSize
        self.Mean                    = Mean
        self.IncreasingIntegers      = IncreasingIntegers
        self.CumSum_Sample           = CumSum_Sample
        self.Regression_Cumsum       = Regression_Cumsum
        self.ScaledXTVaR             = ScaledXTVaR
        self.Minimum                 = Minimum
        self.Maximum                 = Maximum

    def xtVaR(self,q):
        """
        returns lower xtVaR_q
        """
        SampleIndex = round(q*self.SampleSize)
        assert SampleIndex > 0
        tVaR = self.CumSum_Sample[SampleIndex-1]/SampleIndex
        return self.Mean - tVaR

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

    def CalculateRegressionDelta(self,SampleSet_Start, SampleSet_End):
        LocalMean = self.LocalMean(SampleSet_Start, SampleSet_End)
        Y_Right = (SampleSet_End+1)/self.SampleSize
        Y_Left = SampleSet_Start/self.SampleSize
        Regression_Delta_Part1 = -3*LocalMean*(Y_Right+Y_Left)/(Y_Right-Y_Left)
        Regression_Delta_Part2 = 6/((self.SampleSize*(Y_Right-Y_Left))**2)*self.Regression_PartialSum(SampleSet_Start, SampleSet_End)
        Delta_Regression = Regression_Delta_Part1 + Regression_Delta_Part2
        return Delta_Regression

    def FindBestSolutionLine(self, SampleSet_Start, SampleSet_End):

        if (SampleSet_Start == SampleSet_End) or (self.Sample[SampleSet_Start] == self.Sample[SampleSet_End]): #subsample has size one or solution line segment represents a jump part
            # return True, 0.0, 0.0, LinearFunction(self.Sample[SampleSet_Start],0.0)
            return SolutionLine(
                SampleSet_Start         = SampleSet_Start,
                SampleSet_End           = SampleSet_End,
                SampleSize              = self.SampleSize,
                SolutionState           = True,
                Mean                    = self.Sample[SampleSet_Start],
                Delta_LowerBound        = 0.0,
                Delta_UpperBound        = 0.0,
                Delta_Regression        = 0.0,
                BestBisectionPoint      = None)
        else: # subsample has size larger than one and is not jump
            # StartTime = clock()

            # construct vectors defining the inequalities to be solved
            SegmentSize      = (SampleSet_End + 1 - SampleSet_Start)
            LocalMean        = self.LocalMean(SampleSet_Start, SampleSet_End)
            EpsilonPart      = self.Accuracy * self.ScaledXTVaR[SampleSet_Start:SampleSet_End+1]
            # EpsilonPart_Null = self.Accuracy * self.ScaledXTVaR[SampleSet_Start-1] if SampleSet_Start > 0 else 0
            PartialSumPart   = self.CumSum_Sample[SampleSet_Start:SampleSet_End+1] - self.CumSum_Sample_m1(SampleSet_Start)
            MeanIntegralPart = (self.IncreasingIntegers[SampleSet_Start:SampleSet_End+1] - self.IncreasingIntegers[SampleSet_Start] + 1)*LocalMean #cumsum(np.ones(SampleSet_End - SampleSet_Start + 1))
            NonEpsilonPart   = PartialSumPart - MeanIntegralPart
            # calculate the A, B, and C components of the inequality list.
            Ineq_Left   = -EpsilonPart + NonEpsilonPart
            Ineq_Right  = +EpsilonPart + NonEpsilonPart
            Ineq_Middle = ((self.IncreasingIntegers[SampleSet_Start:SampleSet_End+1] - self.IncreasingIntegers[SampleSet_Start] + 1)*
                                 (self.IncreasingIntegers[SampleSet_Start:SampleSet_End+1] - self.IncreasingIntegers[SampleSet_End])/SegmentSize)

            SolutionState, Delta_LowerBound, Delta_UpperBound = SolveSimpleLinearInequailityList(Ineq_Left, Ineq_Middle, Ineq_Right, 1e-10*abs(LocalMean))

            # calculate regression delta
            Delta_Regression = self.CalculateRegressionDelta(SampleSet_Start, SampleSet_End)

            # Calculate best Bisectionpoint...
            # #TODO: following should work better, since it looks at the correctly scaled xtVaR, but strangely works worse than without "/AlphaVector"... investigate!!
            # AlphaVector = np.linspace( (SampleSet_Start+1)/self.SampleSize, (SampleSet_End+1)/self.SampleSize, num = SegmentSize)
            # IntegralDiffWithRegressionDelta = np.abs(Delta_Regression*Ineq_Middle - NonEpsilonPart)/AlphaVector
            IntegralDiffWithRegressionDelta = np.abs(Delta_Regression*Ineq_Middle - NonEpsilonPart)
            # BestBisectionPoint = SampleSet_Start + np.argmax(IntegralDiffWithRegressionDelta) #round( (SampleSet_Start + SampleSet_End)/2 )
            # MaximumDifference = np.max(IntegralDiffWithRegressionDelta)/self.SampleSize
            MaxIndex = np.argmax(IntegralDiffWithRegressionDelta) #round( (SampleSet_Start + SampleSet_End)/2 )
            MaximumDifference = IntegralDiffWithRegressionDelta[MaxIndex]/self.SampleSize
            BestBisectionPoint = MaxIndex + SampleSet_Start
			
            # EndTime = clock()
            # print('Time required for Analysis of sement from '+str(SampleSet_Start)+' to '+str(SampleSet_End)+': '+str(EndTime-StartTime)+' seconds')

            return SolutionLine(
                SampleSet_Start         = SampleSet_Start,
                SampleSet_End           = SampleSet_End,
                SampleSize              = self.SampleSize,
                SolutionState           = SolutionState,
                Mean                    = LocalMean,
                Delta_LowerBound        = Delta_LowerBound,
                Delta_UpperBound        = Delta_UpperBound,
                Delta_Regression        = Delta_Regression,
                BestBisectionPoint      = BestBisectionPoint,
                MaximumDifference       = MaximumDifference,
                AdmissibilityInequality_Left   = Ineq_Left,
                AdmissibilityInequality_Middle = Ineq_Middle,
                AdmissibilityInequality_Right  = Ineq_Right)



