__author__ = 'chuarph'

from .problem_stack import ProblemInterval
import numpy as np

class SolutionLine:

    SampleSet_Start         = 0
    SampleSet_End           = 0
    SampleSize              = 0
    SolutionState           = None
    Mean                    = 0.0 #local mean
    Delta_LowerBound        = 0.0
    Delta_UpperBound        = 0.0
    Delta_Regression        = 0.0
    Delta_Selected          = None
    BestBisectionPoint      = None
    MaximumDifference       = None

    Segment_Line_Start_Y = 0.0
    Segment_Line_End_Y   = 0.0
    Segment_Line_Start_X = 0.0
    Segment_Line_End_X   = 0.0

    AdmissibilityInequality_Left   = None
    AdmissibilityInequality_Middle = None
    AdmissibilityInequality_Right  = None

    def __init__(self,
                 SampleSet_Start                = 0,
                 SampleSet_End                  = 0,
                 SampleSize                     = 0,
                 SolutionState                  = None,
                 Mean                           = 0.0,
                 Delta_LowerBound               = 0.0,
                 Delta_UpperBound               = 0.0,
                 Delta_Regression               = 0.0,
                 BestBisectionPoint             = None,
                 MaximumDifference              = None,
                 AdmissibilityInequality_Left   = None,
                 AdmissibilityInequality_Middle = None,
                 AdmissibilityInequality_Right  = None):

        self.SampleSet_Start    = SampleSet_Start
        self.SampleSet_End      = SampleSet_End
        self.SampleSize         = SampleSize
        self.SolutionState      = SolutionState
        self.Mean               = Mean
        self.Delta_LowerBound   = Delta_LowerBound
        self.Delta_UpperBound   = Delta_UpperBound
        self.Delta_Regression   = Delta_Regression
        self.BestBisectionPoint = BestBisectionPoint
        self.MaximumDifference  = MaximumDifference

        self.AdmissibilityInequality_Left   = AdmissibilityInequality_Left
        self.AdmissibilityInequality_Middle = AdmissibilityInequality_Middle
        self.AdmissibilityInequality_Right  = AdmissibilityInequality_Right

        if SolutionState:
            assert Delta_LowerBound <= Delta_UpperBound
            self.SetDelta(Delta_Regression)

    def Size(self):
        return self.SampleSet_End - self.SampleSet_Start + 1

    def ClosestAcceptableDelta(self,R2):
        return min(max(R2,self.Delta_LowerBound),self.Delta_UpperBound)

    def SetDelta(self,Delta):
        assert self.SolutionState
        self.Delta_Selected = self.ClosestAcceptableDelta(Delta)
        self.SetPWLCoordinates()

    def isBisectable(self):
        # returns true if it can be bisected
        if (self.SampleSet_End > self.SampleSet_Start) and (self.BestBisectionPoint is not None) and (self.Delta_Regression > 0.0):
            return True
        else:
            return False

    def Bisect(self):
        # BisectionPoint = int(round((self.SampleSet_Start + self.SampleSet_End)/2.0)) #alternative: take the middle!
        LeftInterval  = ProblemInterval(self.SampleSet_Start, self.BestBisectionPoint)
        RightInterval = ProblemInterval(self.BestBisectionPoint +1, self.SampleSet_End)
        return [LeftInterval, RightInterval]

    def SetPWLCoordinates(self):
        self.Segment_Line_Start_Y = self.SampleSet_Start/self.SampleSize
        self.Segment_Line_End_Y   = (self.SampleSet_End+1)/self.SampleSize
        self.Segment_Line_Start_X = self.Calculate_StartXFromDelta(self.Delta_Selected)
        self.Segment_Line_End_X   = self.Calculate_EndXFromDelta(self.Delta_Selected)

    def Calculate_EndXFromDelta(self,Delta):
        return self.Mean + Delta

    def Calculate_StartXFromDelta(self,Delta):
        return self.Mean - Delta

    def Calculate_DeltaFromEndX(self,EndX):
        return EndX - self.Mean

    def Calculate_DeltaFromStartX(self,StartX):
        return self.Mean - StartX

    def _CalculateMminusL(self, SampleStats):
        PrecisionFactor = 1e-10
        assert(np.all(self.Delta_Selected*self.AdmissibilityInequality_Middle - self.AdmissibilityInequality_Right < abs(self.Mean)*PrecisionFactor))
        MminusL = np.empty(self.Size() + 1)
        MminusL[1:] = self.Delta_Selected * self.AdmissibilityInequality_Middle - self.AdmissibilityInequality_Left
        # first part of segment. Since first value is not in MminusL vector, we have to do the first part separately.
        #works also for SampleSet_Start == 0 since ScaledXTVaR[-1]=0
        # We have: M^0-L^0 = MminusLm1 = "MminusL[-1]", M^1-L^1 = MminusL[0], MminusLdiff = MminusL[0]-MminusLm1
        MminusL[0] = SampleStats.Accuracy * SampleStats.ScaledXTVaR[self.SampleSet_Start-1]
        return MminusL

    def IsStrictyAdmissible(self, SampleStats, Accuracy):
        # returns True if the inequality |xtVaR_a(G) - xtVaR_a(F)| <= epsilon xtVaR_a(F) is satisfied
        # for all values a between Segment_Line_Start_Y and Segment_Line_End_Y
        if self.Delta_UpperBound == 0: #catch degenerate cases
            return True

        MminusL = self._CalculateMminusL(SampleStats)
        MminusLdiff = np.diff(MminusL)
        M2 = self.Delta_Selected/self.Size()

        # check all parts of the segment from z_s+1/n to z_{s+1}
        IndicesToCheck = np.where(np.logical_and( np.abs(MminusLdiff)<M2, np.minimum(MminusL[:-1], MminusL[1:])<0.25*M2 ))
        for i in np.nditer(IndicesToCheck,['zerosize_ok']):
            if not SubsegmentIsStrictlyAdmissible(MminusL[i], MminusLdiff[i], M2):
                return False

        # check whether strict admissibility is also satisfied for threshold converging to 0. This corresponds to the
        # boundary case for the first segment. Relative error is obtained by taking the limit going to 0.
        if self.Segment_Line_Start_Y == 0:
            if np.abs(self.Segment_Line_Start_X - SampleStats.Minimum)/(SampleStats.Mean - SampleStats.Minimum) >= Accuracy:
                return False

        # check whether strict admissibility is also satisfied for threshold converging to 1. This corresponds to the
        # boundary case for the first segment. Relative error is obtained by taking the limit going to 1.
        if self.Segment_Line_End_Y == 1:
            if np.abs(self.Segment_Line_End_X - SampleStats.Maximum)/(SampleStats.Maximum - SampleStats.Mean) >= Accuracy:
                return False

        # if none of the checks above resulted into False then the segment satisfies strict admissibility
        return True

def SubsegmentIsStrictlyAdmissible(M0minusL0, MminusLdiffValue, M2):
    # M0minusL0 = M^0 - L^0
    # MminusLdiffValue = (M^1 - M^0) - (L^1 - L^0)
    # M2 = M2
    # Returns True if inequality L(x) <= M(x) is satisfied for 0<=x<=1. False if not.
    assert(abs(MminusLdiffValue) < M2)
    Xstar = 0.5-0.5*MminusLdiffValue/M2
    MinValue = M0minusL0 + Xstar*MminusLdiffValue + M2*Xstar*(Xstar-1) # MinValue = M(X^star) - L(X^star)
    return MinValue >= 0
