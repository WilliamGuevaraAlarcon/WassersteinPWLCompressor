__author__ = 'chuarph'

import numpy as np

class ProblemStack:

    Stack = None

    #minimum sample size such that atom detection is applied
    AtomDetectionMinimumSampleSize = 1000# 1000

    # if the number of Samplepoints equal to x is larger than AtomDetectionThreshold*Samplesize, then it is declared an atom.
    RelativeAtomDetectionThreshold = 0.005 #0.005

    def __init__(self,SampleStats, AtomDetection = True, AtomDetectionMinimumSampleSize = 1000,
                 RelativeAtomDetectionThreshold = 0.005, EnforcedInterpolationQuantiles = []):
        """
        initialize the problem stack.
        if AtomDetection = True, then enforce all atoms to be covered by a single ProblemInterval, such that all atoms
        are covered by single straight vertical PWL component

        Example: if Sample = [ 24.  ,  26.75,  27.4 ,  27.45,  30, 30, 30, 30, 30, 30, 30.15,  30.5 ,  31.45,  32.7 ,  33.8 ]
        PS = ProblemStack(SubsampleApproximation([ 24.  ,  26.75,  27.4 ,  27.45,  30, 30, 30, 30, 30, 30, 30.15,  30.5 ,  31.45,  32.7 ,  33.8 ]),
                                                  AtomDetectionMinimumSampleSize = 10, RelativeAtomDetectionThreshold = 0.1)
        gives
        PS.Stack = [ProblemInterval(0,3), ProblemInterval(4,9), ProblemInterval(10,14)]
        """

        self.AtomDetectionMinimumSampleSize = AtomDetectionMinimumSampleSize
        self.RelativeAtomDetectionThreshold = RelativeAtomDetectionThreshold
        assert(all([0 <= x <= 1 for x in EnforcedInterpolationQuantiles]))

        SampleSize = SampleStats.SampleSize
        CuttingPoints = [0,SampleSize]

        # append enforced points
        for EnforcedQantile in EnforcedInterpolationQuantiles:
            CuttingPoints.append( round(EnforcedQantile * SampleSize) )

        # run atom detection if conditions are given
        if AtomDetection and (SampleSize >= self.AtomDetectionMinimumSampleSize):
            AtomDetectionThreshold = int(self.RelativeAtomDetectionThreshold*SampleSize)
            AtomLocations = (SampleStats.Sample[AtomDetectionThreshold:] == SampleStats.Sample[:SampleSize-AtomDetectionThreshold])
            AtomList = np.unique(SampleStats.Sample[AtomLocations])

            #find begin and end coordinates of atoms in the sample
            for Atom in AtomList:
                LeftAtomEnd  = np.searchsorted(SampleStats.Sample,Atom, side = 'left' )
                RightAtomEnd = np.searchsorted(SampleStats.Sample,Atom, side = 'right')
                CuttingPoints.append(LeftAtomEnd)
                CuttingPoints.append(RightAtomEnd)

        # make cutting points sorted and unique
        CuttingPoints = list(np.sort(np.unique(CuttingPoints)))

        #create ProblemInterval according to CuttingPoints
        self.Stack = []
        for i in range(len(CuttingPoints)-1):
            self.Stack.append(ProblemInterval(CuttingPoints[i],CuttingPoints[i+1]-1))


    def isNotEmpty(self):
        """
        returns True if the stack contains an item, False otherwise
        """
        if self.Stack:
            return True
        else:
            return False

    def isEmpty(self): #    returns False if the stack contains an item, True otherwise
        return not self.isNotEmpty()

    def pop(self):
        return self.Stack.pop()

    def extend(self,x):
        self.Stack.extend(x)



class ProblemInterval:
    SampleSet_Start = None
    SampleSet_End   = None
    def __init__(self, Start, End):
        self.SampleSet_Start = Start
        self.SampleSet_End = End





