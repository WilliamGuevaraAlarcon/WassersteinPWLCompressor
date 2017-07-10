__author__ = 'chuarph'

import numpy as np

class Segment:

    SampleSet_Start         = 0
    SampleSet_End           = 0
    SampleSize              = 0
    Mean                    = 0.0 #local mean
    Delta_Regression        = 0.0
    Delta_Selected          = None
    BestBisectionPoint      = None


    Segment_Line_Start_Y        = 0.0
    Segment_Line_End_Y          = 0.0
    Segment_Line_Start_X        = 0.0
    Segment_Line_End_X          = 0.0
    WDiscretized_i              = None
    WassersteinDiscretized      = None
    WassersteinExact            = None

    def __init__(self,
                 SampleSet_Start                = 0,
                 SampleSet_End                  = 0,
                 SampleSize                     = 0,
                 Mean                           = 0.0,
                 Delta_Regression               = 0.0,
                 BestBisectionPoint             = None,
                 Sample                         = None):

        self.SampleSet_Start    = SampleSet_Start
        self.SampleSet_End      = SampleSet_End
        self.SampleSize         = SampleSize
        self.Mean               = Mean
        self.Delta_Regression   = Delta_Regression
        self.BestBisectionPoint = BestBisectionPoint
        self.DeltaMultiplier    = (2 * (np.linspace(self.SampleSet_Start + 1 / 2, self.SampleSet_End + 1 / 2, self.SampleSet_End + 1 - self.SampleSet_Start) - self.SampleSet_Start) / (self.SampleSet_End + 1 - self.SampleSet_Start) - 1)
        self.SubSample =  Sample[self.SampleSet_Start:self.SampleSet_End + 1]
        self.SetDelta(Delta_Regression)

    def Size(self):
        return self.SampleSet_End - self.SampleSet_Start + 1


    def SetDelta(self, Delta):
        self.Delta_Selected = max(0.0, Delta)
        self.SetPWLCoordinates()
        self.SetWasserstein()

    def SetWasserstein(self):
        self.WDiscretized_i = np.abs(self.SubSample - self.Mean - self.Delta_Selected * self.DeltaMultiplier) / self.SampleSize
        self.WassersteinDiscretized = np.sum(self.WDiscretized_i)

    def WassersteinDis(self, Type = "Discretized"):
        if Type == "Discretized":
            return self.WassersteinDiscretized
        elif Type == "Exact":
            delta_star = self.Delta_Selected / (self.Size())
            if delta_star == 0:
                self.WassersteinExact = self.WassersteinDiscretized
            else:
                self.WassersteinExact = self.WassersteinDiscretized + np.sum(1 / 2 * (1 / self.SampleSize - self.WDiscretized_i / delta_star) * np.maximum(
                                                       delta_star - self.SampleSize * self.WDiscretized_i, 0))
            return self.WassersteinExact

    def isBisectable(self):
        # returns true if it can be bisected
        return bool(self.SampleSet_End > self.SampleSet_Start) and (self.BestBisectionPoint is not None) and (self.Delta_Regression > 0.0)

    def Bisect(self, SampleStats):
        # BisectionPoint = int(round((self.SampleSet_Start + self.SampleSet_End)/2.0)) #alternative: take the middle!
        LeftSegment  = SampleStats.FindBestSolutionLine(self.SampleSet_Start, self.BestBisectionPoint)
        RightSegment = SampleStats.FindBestSolutionLine(self.BestBisectionPoint + 1, self.SampleSet_End)

        return [LeftSegment, RightSegment]

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
