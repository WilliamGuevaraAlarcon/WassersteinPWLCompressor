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
    Bisectable            = None

    Segment_Line_Start_Y = 0.0
    Segment_Line_End_Y   = 0.0
    Segment_Line_Start_X = 0.0
    Segment_Line_End_X   = 0.0
    WassersteinDistance  = 0.0

    def __init__(self,
                 SampleSet_Start                = 0,
                 SampleSet_End                  = 0,
                 SampleSize                     = 0,
                 Mean                           = 0.0,
                 Delta_Regression               = 0.0,
                 Bisectable                     = None,
                 BestBisectionPoint             = None,
                 Sample                         = None):

        self.SampleSet_Start    = SampleSet_Start
        self.SampleSet_End      = SampleSet_End
        self.SampleSize         = SampleSize
        self.Mean               = Mean
        self.Delta_Regression   = Delta_Regression
        self.BestBisectionPoint = BestBisectionPoint
        self.Bisectable         = Bisectable

        self.SetDelta(Delta_Regression, Sample)

    def Size(self):
        return self.SampleSet_End - self.SampleSet_Start + 1


    def SetDelta(self, Delta, Sample):
        self.Delta_Selected = max(0.0, Delta)
        self.SetPWLCoordinates()
        self.SetWasserstein(Sample)

    def SetWasserstein(self, Sample):
        if self.Bisectable == False:
            if self.Segment_Line_Start_X == self.Segment_Line_End_X and self.Segment_Line_Start_X == self.Mean:
                self.WassersteinDistance = 0
            else:
                raise Exception('Constant segment with no zero Wasserstein')
        else:
            self.WassersteinDistance = sum(np.abs(Sample[self.SampleSet_Start:self.SampleSet_End + 1] -
                       (self.Mean + self.Delta_Selected * (2 * (np.linspace(self.SampleSet_Start + 1 / 2, self.SampleSet_End + 1 / 2, self.SampleSet_End + 1 - self.SampleSet_Start) -
                                                                self.SampleSet_Start) / (self.SampleSet_End + 1 - self.SampleSet_Start) - 1))) / self.SampleSize)

        #Condition_X = self.Sample[SampleSet_Start:SampleSet_End] - LocalMean + Delta*((SampleSet_End + 1 + SampleSet_Start) / (SampleSet_End + 1 - SampleSet_Start))
        #Condition_S_start = (2 * Delta * self.IncreasingIntegers[SampleSet_Start:SampleSet_End]) / (SampleSet_End + 1 - SampleSet_Start)
        #Condition_S_end = (2 * Delta * (self.IncreasingIntegers[SampleSet_Start+1:SampleSet_End+1])) / (SampleSet_End + 1 - SampleSet_Start)


        #a = Condition_X < Condition_S_start
        #b = Condition_S_end < Condition_X

        #re = self.Sample[SampleSet_Start + Condition_S_end < Condition_X]

        #c = (Condition_S_start < Condition_X) and  (Condition_X< Condition_S_end)


    # def WassersteinDis(self, Start, End):
    #     Condition_X = self.Sample - mu_s + delta_s * ((z1 + z2) / sample_length)
    #     Condition_S_start = (2 * delta_s * (i - 1)) / (n * sample_length)
    #     Condition_S_end = (2 * delta_s * i) / (n * sample_length)


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

class ProblemInterval:
    SampleSet_Start = None
    SampleSet_End   = None
    def __init__(self, Start, End):
        self.SampleSet_Start = Start
        self.SampleSet_End = End