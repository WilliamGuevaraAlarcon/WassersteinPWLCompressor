__author__ = 'chuarph'

# import numpy as np
from .segment import Segment
import numpy as np

class SegmentStack:

    Stack               = None
    verbose             = False
    #minimum sample size such that atom detection is applied
    AtomDetectionMinimumSampleSize = 1000# 1000

    # if the number of Samplepoints equal to x is larger than AtomDetectionThreshold*Samplesize, then it is declared an atom.
    RelativeAtomDetectionThreshold = 0.005

    def __init__(self, SampleStats, AtomDetection = True, EnforcedInterpolationQuantiles = [], Bisection = 'Original', Verbose = False, RemoveNegativeJumps = True):

        self.Stack = []
        self.verbose = Verbose

        assert(all([0 <= x <= 1 for x in EnforcedInterpolationQuantiles]))

        SampleSize = SampleStats.SampleSize
        CuttingPoints = [0,SampleSize]

        # append enforced points
        for EnforcedQuantile in EnforcedInterpolationQuantiles:
            CuttingPoints.append( round(EnforcedQuantile * SampleSize) )



        if (not isinstance(AtomDetection, bool) and not isinstance(AtomDetection, tuple)):
            raise Exception('Incorrect format of the atom detection parameter')


        if isinstance(AtomDetection, tuple):
            self.AtomDetectionMinimumSampleSize = AtomDetection[0]
            self.RelativeAtomDetectionThreshold = AtomDetection[1]

        # run atom detection if conditions are given
        if AtomDetection != False and (SampleSize >= self.AtomDetectionMinimumSampleSize):
            AtomDetectionThreshold = int(self.RelativeAtomDetectionThreshold*SampleSize)
            AtomLocations = (SampleStats.Sample[AtomDetectionThreshold:] == SampleStats.Sample[:SampleSize-AtomDetectionThreshold])
            AtomList = np.unique(SampleStats.Sample[AtomLocations])

            #find begin and end coordinates of atoms in the sample
            for Atom in AtomList:
                LeftAtomEnd  = np.searchsorted(SampleStats.Sample, Atom, side = 'left' )
                RightAtomEnd = np.searchsorted(SampleStats.Sample, Atom, side = 'right')
                CuttingPoints.append(LeftAtomEnd)
                CuttingPoints.append(RightAtomEnd)

        # make cutting points sorted and unique
        CuttingPoints = list(np.sort(np.unique(CuttingPoints)))

        for i in range(len(CuttingPoints)-1):
            self.Stack.append(SampleStats.FindBestSolutionLine(CuttingPoints[i], CuttingPoints[i + 1] - 1, Bisection))

        # for i in range(StackLength - 1):
        #     AbsoluteEpsilon = abs(self.Stack[i + 1].Segment_Line_Start_X) * Epsilon
        #     if (((self.Stack[i].SampleSet_End + 1) == self.Stack[
        #             i + 1].SampleSet_Start) &  # end-y and start-y are equal, i.e. segments attach
        #             (self.Stack[i + 1].Segment_Line_Start_X - self.Stack[
        #                 i].Segment_Line_End_X < -AbsoluteEpsilon)):  # there is a negative jump in the x-coordinate, which is not allowed for cdf's
        #         IndicesWithNegativeIncrements.append(i)


        if RemoveNegativeJumps:
            self.CorrectNegativeIncrements(SampleStats.Sample)

    def TotalWasserstein(self):
        return sum(Segment.WassersteinDistance for Segment in self.Stack)

    def StackLength(self):
        return len(self.Stack)

    def isNotEmpty(self):
        return bool(self.Stack)

    def isEmpty(self): #
        return not self.isNotEmpty()

    # TODO not sure if the following two functions can be useful
    #def pop(self):
    #   return self.Stack.pop()

    #def extend(self,x):
    #    self.Stack.extend(x)

    def append(self,ObjectToPush):
        if not(isinstance(ObjectToPush, Segment)):
            raise Exception('pushed object must be of type SolutionLine')
        self.Stack.append(ObjectToPush)
        if self.isNotEmpty(): #if not empty
            self.Stack = sorted(self.Stack, key = lambda y : y.SampleSet_Start) #sort solutions according to y start point

    def CorrectNegativeIncrements(self, Sample):
        if self.isNotEmpty():
            Epsilon = 1e-9
            for i in range(self.StackLength() - 1):
                AbsoluteEpsilon = abs(self.Stack[i + 1].Segment_Line_Start_X) * Epsilon
                if (self.Stack[i + 1].Segment_Line_Start_X - self.Stack[i].Segment_Line_End_X < -AbsoluteEpsilon):  # there is a negative jump in the x-coordinate, which is not allowed for cdf's
                    self.CorrectSegments(self.Stack[i], self.Stack[i+1], Sample)

    def CorrectSegments(self, Seg_Down, Seg_Up, Sample):

        if Seg_Down.isBisectable() == True and Seg_Up.isBisectable() == True:
            ConnectingPoint = (Seg_Up.Segment_Line_Start_X + Seg_Down.Segment_Line_End_X) / 2.0
            ConnectingPoint = max(ConnectingPoint, Seg_Down.Mean)
            Seg_Down.SetDelta(Seg_Down.Calculate_DeltaFromEndX(ConnectingPoint), Sample)
            Seg_Up.SetDelta(Seg_Up.Calculate_DeltaFromStartX(ConnectingPoint), Sample)
        elif Seg_Down.isBisectable() == True and Seg_Up.isBisectable() == False:
            Seg_Down.SetDelta(Seg_Down.Calculate_DeltaFromEndX(Seg_Up.Segment_Line_Start_X), Sample)
        elif Seg_Down.isBisectable() == False and Seg_Up.isBisectable() == True:
            Seg_Up.SetDelta(Seg_Up.Calculate_DeltaFromStartX(Seg_Down.Segment_Line_End_X), Sample)
        else:  # this should never happen...
            raise Exception('Adjacent solutions are incompatible, but neither is bisectable.')


    def CheckCompletenessOfStack(self):
        """
        return true if the whole interval [0,1] is covered with adjacent solutions. false otherwise
        """
        StackLength = self.StackLength()
        if (self.Stack[0].Segment_Line_Start_Y != 0) or (self.Stack[StackLength-1].Segment_Line_End_Y != 1):
            return False
        for i in range(StackLength-1):
            if self.Stack[i].Segment_Line_End_Y != self.Stack[i+1].Segment_Line_Start_Y:
                return False
        return True

    def BisectBiggestWasserstein(self, SampleStats, Bisection):
        """
        return the bisected segment where the Wasserstein distance is biggest
        """
        i_to_pop = max(enumerate(self.Stack), key = lambda x: x[1].WassersteinDistance)[0]
        if i_to_pop == None:
            raise Exception("No Bisectable Interval left, accuracy can't be reached")

        LeftSegment, RightSegment = self.Stack.pop(i_to_pop).Bisect(SampleStats, Bisection)
        self.append(LeftSegment)
        self.append(RightSegment)



