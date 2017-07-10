from .sample_characteristics import SampleCharacteristics
from .segment_stack import SegmentStack
from .pwl_distribution import EmpiricalDistributionFromSample
from time import clock
import matplotlib.pyplot as plt
import pandas as pd

class PWLcompressor:

    SampleStats   = None
    SegmentStack  = None
    Result        = None

    def __init__(self, Sample, RemoveNegativeJumps = True,
                 AccuracyMode = "Relative", AccuracyParameter = 0.1,
                 AtomDetection = True, CheckStrictWasserstein = True,
                 Verbose = True, PlotIntermediate = False):
        # Implements the Compression algorithm for a sample with a certain accuracy of choice.
        # INPUT:
        #   Sample:                 The sample, in form of list or numpy array
        #                           Example:  Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
        #   RemoveNegativeJumps:    boolean switch to determine whether negative increments should be removed in the PWL
        #                           approximation. This parameter should always be set to True except for testing and
        #                           illustration purposes.
        #                           Example:  RemoveNegativeJumps = True
        #   AccuracyMode:           Two possible options: "Relative" or "Absolute". If "Relative" is chosen, the accuracy
        #                           epsilon is equal to estimated Wasserstein distance W(F, F_n) multiplied by the
        #                           AccuracyParameter. If "Absolute" is chosen, the accuracy epsilon is simply equal to
        #                           the AccuracyParameter.
        #   AccuracyParameter:      When AccuracyMode = "Relative", it is the proportion of the estimated Wasserstein distance W(F, F_n)
        #                           When AccuracyMode = "Absolute", it is the epsilon used for the accuracy of the approximation
        #   AtomDetection:          It can be a boolean or a tuple with two numbers.
        #                           If False: No atom detection is applied
        #                           If True: Atom detection is applied with the default parameters: minimum sample size which
        #                           needs to be satisfied such that the algorithm looks for atoms, AtomDetectionMinimumSampleSize = 1000
        #                           and RelativeAtomDetectionThreshold = 0.005, so that the atoms that are bigger than 0.5%
        #                           of the Sample Size are detected
        #                           If a tuple: Atom detection is applied with the first number in the tuple as AtomDetectionMinimumSampleSize,
        #                           the minimum sample size which needs to be satisfied such that the algorithm looks for atoms,
        #                           and the second number as RelativeAtomDetectionThreshold, the proportion respect to the
        #                           SampleSize that a value needs at least to be repeated in order to be detected as an atom
        #   CheckStrictWasserstein: boolean. When True the approximation is done based on the Wasserstein distance when False
        #                           the approximation is done based on the discretized version of the Wasserstein distance
        #   Verbose:               Print status messages to console
        # OUTPUT: an object with the following member properties
        #    * Result:  an admissible PWL approximation of the Sample with accuracy (epsilon) depending on AccuracyMode and
        #               AccuracyParameter selection
        #    * SampleStats, SegmentStack, : intermediate properties and values not relevant for the user
        #                         who is only interested in the resulting PWL approximation

        StartTime = clock() #startpoint of time

        # initialize properties of object
        self.SampleStats = SampleCharacteristics(Sample) #initialize sample statistics

        self.SegmentStack = SegmentStack(self.SampleStats, AtomDetection = AtomDetection,
                                         RemoveNegativeJumps = RemoveNegativeJumps, Verbose = Verbose) # C

        if PlotIntermediate:
            Temp_Result = self.GivePWLPoints()
            self.plotIntermediate(Temp_Result, color='b')

        assert (AccuracyMode in ["Relative" , "Absolute"]), "AccuracyMode only accepts options Relative or Absolute"

        if AccuracyMode == "Relative":
            Accuracy = AccuracyParameter*self.SampleStats.EstimatedWasserstein
        elif AccuracyMode == "Absolute":
            Accuracy = AccuracyParameter

        while self.SegmentStack.TotalWasserstein("Discretized") > Accuracy:
            if Verbose:
                print('Wasserstein discretized = ' + str(self.SegmentStack.TotalWasserstein("Discretized"))
                       + " is bigger than accuracy. ", end='')

            self.SegmentStack.BisectBiggestWasserstein(Type = "Discretized")

            if RemoveNegativeJumps:
                self.SegmentStack.CorrectNegativeIncrements()

            if PlotIntermediate:
                Temp_Result = self.GivePWLPoints()
                self.plotIntermediate(Temp_Result, color='b')

        if CheckStrictWasserstein:
            while self.SegmentStack.TotalWasserstein("Exact") > Accuracy:
                if Verbose:
                    print('Wasserstein = ' + str(self.SegmentStack.TotalWasserstein("Exact"))
                          + " is bigger than accuracy. ", end='')

                self.SegmentStack.BisectBiggestWasserstein(Type = "Exact")

                if RemoveNegativeJumps:
                    self.SegmentStack.CorrectNegativeIncrements()

                if PlotIntermediate:
                    Temp_Result = self.GivePWLPoints()
                    self.plotIntermediate(Temp_Result, color='b')

        self.SegmentStack.CheckCompletenessOfStack()

        self.Result = self.GivePWLPoints()

        EndTime = clock()

        if Verbose:
            if CheckStrictWasserstein:
                print('\n Number of grid points on PWL distribution: ' + str(len(self.Result['PWLX']))
                    +'\n Time required: '+'{:.3f}'.format(EndTime-StartTime)+' seconds \n '
                    +'\n Wasserstein distance achieved: '+ str(self.SegmentStack.TotalWasserstein("Exact"))
                    +'\n Accuracy: '+ str(Accuracy) + '\n'
                    +'\n ======== COMPRESSION FINISHED! ========')
            else:
                print('\n Number of grid points on PWL distribution: ' + str(len(self.Result['PWLX']))
                    + '\n Time required: ' + '{:.3f}'.format(EndTime - StartTime) + ' seconds \n '
                    + '\n Wasserstein discretized distance achieved: ' + str(self.SegmentStack.TotalWasserstein("Discretized"))
                    + '\n Accuracy: ' + str(Accuracy) + '\n'
                    + '\n ======== COMPRESSION FINISHED! ========')

    def GivePWLPoints(self):
        # produce the PWL coordinates corresponding to the current solutionstack.
        assert self.SegmentStack.isNotEmpty()

        PrecisionFactor = 1e-7 #TODO: unify all comparisons of coordinates

        X = []
        P = []
        for Sol in self.SegmentStack.Stack:
            X.extend([Sol.Segment_Line_Start_X,Sol.Segment_Line_End_X])
            P.extend([Sol.Segment_Line_Start_Y,Sol.Segment_Line_End_Y])

        #remove duplicates
        Xfixed = [X[0]]
        Pfixed = [P[0]]
        for i in range(1,len(X)):
            assert(P[i] >= Pfixed[-1])
            PointEpsilon = PrecisionFactor*max(1,abs(X[i]),abs(Xfixed[-1]))
            #append points only if necessary
            Xdifferent = (abs(X[i] - Xfixed[-1]) > PointEpsilon)
            Ydifferent = (P[i] != Pfixed[-1])
            # in case X[j] == Xfixed.back(), only add (X[j],Y[j]) if X[j+1]>X[j].
            # I.e., do not add (X[j],Y[j]) if X[j+1] == X[j] == Xfixed.back()
            PointRedundant = ((not Xdifferent) and (i + 1 < len(X)) and (abs(X[i+1] - Xfixed[-1]) < PointEpsilon))
            if (not PointRedundant) and (Xdifferent or Ydifferent):
                Xfixed.append(X[i])
                Pfixed.append(P[i])

        return dict(PWLX = Xfixed, PWLY = Pfixed)

    def GiveSolIntervals(self, filename):

        assert self.SegmentStack.isNotEmpty()

        X = []
        P = []
        for Sol in self.SegmentStack.Stack:
            if Sol.Size() > 1:
                X = [Sol.Delta_LowerBound, Sol.Delta_Regression, Sol.Delta_Selected, Sol.Delta_UpperBound, Sol.Mean, Sol.Segment_Line_Start_X,
                     Sol.Segment_Line_End_X, Sol.Segment_Line_Start_Y, Sol.Segment_Line_End_Y]
                P.append(X)

        my_df = pd.DataFrame(P)
        my_df.to_csv(filename, index=False, header=False)

    def plot(self, color = 'g', ShowPlot = True):
        Xpoints, Ypoints = self.Result['PWLX'], self.Result['PWLY']
        # Xpoints, Ypoints = self.GivePWLPoints(CorrectRoundoffNegativeIncrements = False)
        # print('Number of grid points on PWL distribution:' + str(len(Xpoints))) #len(self.SolutionStack.Stack))

        SampleDist = EmpiricalDistributionFromSample(self.SampleStats.Sample)
        if ShowPlot:
            plt.plot(SampleDist.Xvalues,SampleDist.Fvalues,color='black')
            plt.plot(Xpoints,Ypoints, linewidth=2.0, linestyle = '-', marker = 'o', color = color)
            plt.show()  # plt.show(block=False) # plt.hold(True)

        return dict(SampleX = SampleDist.Xvalues,
                    SampleY = SampleDist.Fvalues,
                    PWLX    = Xpoints,
                    PWLY    = Ypoints)

    def plotIntermediate(self, Int_Result, color='b'):
        Xpoints=Int_Result['PWLX']
        Ypoints=Int_Result['PWLY']
        SampleDist = EmpiricalDistributionFromSample(self.SampleStats.Sample)
        plt.plot(SampleDist.Xvalues, SampleDist.Fvalues, color='black')
        plt.plot(Xpoints, Ypoints, linewidth=2.0, linestyle='--', marker='o', color=color, fillstyle='none')
        plt.show()
