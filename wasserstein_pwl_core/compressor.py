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
                 Accuracy = 0.001, EnforcedInterpolationQuantiles = list(),
                 AtomDetection = True,
                 Verbose = True, Bisection = 'Original', PlotIntermediate = False):
        # Implements the Compression algorithm for a sample with a certain accuracy of choice.
        # INPUT:
        #    Sample:              The sample, in form of list or numpy array
        #                         Example:  Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
        #    RemoveNegativeJumps: boolean switch to determine whether negative increments should be removed in the PWL
        #                         approximation. This parameter should always be set to True except for testing and
        #                         illustration purposes.
        #                         Example:  RemoveNegativeJumps = True
        #    Accuracy:            Accuracy of approximation (called epsilon in paper)
        #                         Example:  Accuracy = 0.01
        #    EnforcedInterpolationQuantiles: List of quantiles which shall be contained in the basis of the
        #                         approximation. However, note that in case the corresponding basis
        #                         point is redundant in the PWL approximation (i.e., if the interpolation points below
        #                         and above are on a straight line) then it will be removed
        #                         Example:  EnforcedInterpolationQuantiles = [0.1, 0.5, 0.9]
        #    AtomDetection:       boolean switch to determine whether atoms should be detected in the sample
        #                         distribution. (for details see RelativeAtomDetectionThreshold description)
        #                         Example:  AtomDetection = True
        #    AtomDetectionMinimumSampleSize: Minimum sample size which needs to be satisfied such that the algorithm
        #                         looks for atoms. (for details see RelativeAtomDetectionThreshold description)
        #                         Example:  AtomDetectionMinimumSampleSize = 1000
        #    RelativeAtomDetectionThreshold: Relative threshold for atoms to be detected. In case AtomDetection = True
        #                         and len(Sample) = Samplesize >= AtomDetectionMinimumSampleSize then the algorithm
        #                         calculates "AtomDetectionThreshold =round(SampleSize*RelativeAtomDetectionThreshold)".
        #                         In case a value in Sample occurs at least AtomDetectionThreshold times, then it will
        #                         be approximated by an atom in the PWL approximation.
        #                         Example:  RelativeAtomDetectionThreshold = 0.005 (I.e., for sample size 10'000, a
        #                                   value needs to occur at least 50 times in order to be approximated by a jump)
        #    Verbose:             Print status messages to console
        #    Bisection:           Choose between bisection in the original paper 'Original' and the bisection that
        #                         minimizes the L2 distance 'OLS'
        # OUTPUT: an object with the following member properties
        #    * Result:  an admissible PWL approximation of the Sample with accuracy "Accuracy" (epsilon)
        #    * SampleStats, ProblemStack, : intermediate properties and values not relevant for the user
        #                         who is only interested in the resulting PWL approximation

        StartTime = clock() #startpoint of time

        # initialize properties of object
        self.SampleStats = SampleCharacteristics(Sample) #initialize sample statistics

        self.SegmentStack = SegmentStack(self.SampleStats, AtomDetection = AtomDetection,
                                         EnforcedInterpolationQuantiles = EnforcedInterpolationQuantiles,
                                         Bisection = Bisection, RemoveNegativeJumps = RemoveNegativeJumps) # C

        if PlotIntermediate:
            Temp_Result = self.GivePWLPoints()
            self.plotIntermediate(Temp_Result, color='b')
        print("Wasserstein=", self.SegmentStack.TotalWasserstein())

        while self.SegmentStack.TotalWasserstein() > Accuracy:
            self.SegmentStack.BisectBiggestWasserstein(SampleStats = self.SampleStats, Bisection = Bisection)

            if RemoveNegativeJumps:
                self.SegmentStack.CorrectNegativeIncrements(self.SampleStats.Sample)

            Temp_Result = self.GivePWLPoints()
            if PlotIntermediate:
                self.plotIntermediate(Temp_Result, color='b')
            print("Wasserstein=", self.SegmentStack.TotalWasserstein())

        self.SegmentStack.CheckCompletenessOfStack()

        self.Result = self.GivePWLPoints()


        EndTime = clock()
        if Verbose:
            print('\n Number of grid points on PWL distribution: ' + str(len(self.Result['PWLX']))
                  +'\n Time required: '+'{:.3f}'.format(EndTime-StartTime)+' seconds \n '
                  +'\n Wasserstein distance achieved: '+ str(self.SegmentStack.TotalWasserstein())
                  +'\n ======== COMPRESSION FINISHED! ========')




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
