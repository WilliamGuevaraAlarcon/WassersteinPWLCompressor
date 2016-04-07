
from .sample_characteristics import SampleCharacteristics
from .solution_stack import SolutionStack
from .problem_stack import ProblemStack
import matplotlib.pyplot as plt
from .pwl_distribution import EmpiricalDistributionFromSample
from time import clock

class PWLcompressor:

    SampleStats   = None
    ProblemStack  = None
    SolutionStack = None
    Result        = None

    def __init__(self, Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = True,
                 Accuracy = 0.001, EnforcedInterpolationQuantiles = list(),
                 AtomDetection = True, AtomDetectionMinimumSampleSize = 1000, RelativeAtomDetectionThreshold = 0.005):
        # Implements the Compression algorithm for a sample with a certain accuary of choice.
        # INPUT:
        #    Sample:              The sample, in form of list or numpy array
        #                         Example:  Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
        #    RemoveNegativeJumps: boolean switch to determine whether negative increments should be removed in the PWL
        #                         approximation. This parameter should always be set to True except for testing and
        #                         illustration purposes.
        #                         Example:  RemoveNegativeJumps = True
        #    MakePWLsmoother:     boolean switch to determine whether smoothing should be applied or not.
        #                         Example:  MakePWLsmoother = True
        #    CheckStrictAdmissibility: boolean switch to determine whether the algorithm should check whether an
        #                         admissible approximation is also strictly admissible
        #                         Example:  CheckStrictAdmissibility = True
        #    Accuracy:            Accuracy of approximation (called epsilon in paper)
        #                         Example:  Accuracy = 0.01
        #    EnforcedInterpolationQuantiles: List of quantiles which shall be contained in the basis of the
        #                         approximation. Hence, for these quantiles the PWL distribution will have the same
        #                         xTVaR as the sample distribution. However, note that in case the corresponding basis
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
        # OUTPUT: an object with the following member properties
        #    * Result:  an admissible PWL approximation of the Sample with accuracy "Accuracy" (epsilon)
        #    * SampleStats, ProblemStack, SolutionStack: intermediate properties and values not relevant for the user
        #                         who is only interested in the resulting PWL approximation

        StartTime = clock() #startpoint of time

        # initialize properties of object
        self.SampleStats = SampleCharacteristics(Sample, Accuracy = Accuracy) #initialize sample statistics
        self.ProblemStack = ProblemStack(self.SampleStats, AtomDetection = AtomDetection,
                                         AtomDetectionMinimumSampleSize = AtomDetectionMinimumSampleSize,
                                         RelativeAtomDetectionThreshold = RelativeAtomDetectionThreshold,
                                         EnforcedInterpolationQuantiles = EnforcedInterpolationQuantiles) # C
        self.SolutionStack = SolutionStack()

        # compress:
        while self.ProblemStack.isNotEmpty(): # while ProblemStack is not empty
            # pop a problem and find the corresponding solution
            ThisProblem = self.ProblemStack.pop()
            ThisSolutionLine = self.SampleStats.FindBestSolutionLine(ThisProblem.SampleSet_Start,ThisProblem.SampleSet_End)

            # if solution is acceptable, use it. otherwise bisect the problemInterval and push problem stack
            if ThisSolutionLine.SolutionState: #if ThisSolutionLine has been accepted to approximate sample subset
                self.SolutionStack.append(ThisSolutionLine)
            else: #else, bisect the interval and look for solutions in the smaller intervals in the following loops
                BisectedProblem = ThisSolutionLine.Bisect()
                self.ProblemStack.extend(BisectedProblem)

            #if problemstack is empty, i.e., when the last problem has been removed, check for negative increments
            #if negative increments exist, either fix by making solutions steeper or pop one.
            #Note that ProblemIntervalsToAdd can be empty
            if self.ProblemStack.isEmpty() and RemoveNegativeJumps:  #if problemstack is empty and RemoveNegativeJumps switch is True
                ProblemIntervalsToAdd = self.SolutionStack.CorrectOrPopNegativeIncrements()
                self.ProblemStack.extend(ProblemIntervalsToAdd)

            #if problemstack is empty, i.e., if all problems have been solved and no negative jumps exist, then smoothen
            if self.ProblemStack.isEmpty() and MakePWLsmoother:
                self.SolutionStack.SmoothenSolutions()

            #if an admissible solution has been found, check whether it is strictly admissible
            if self.ProblemStack.isEmpty() and CheckStrictAdmissibility:
                ProblemIntervalsToAdd = self.SolutionStack.CheckStrictAdmissibility(self.SampleStats, Accuracy)
                self.ProblemStack.extend(ProblemIntervalsToAdd)

        self.SolutionStack.CheckCompletenessOfSolutionStack()

        self.Result = self.GivePWLPoints()

        EndTime = clock()
        print('\n Number of grid points on PWL distribution: ' + str(len(self.Result['PWLX']))
              +'\n Time required: '+'{:.3f}'.format(EndTime-StartTime)+' seconds \n '
              +'\n ======== COMPRESSION FINISHED! ========')

    def GivePWLPoints(self):
        # produce the PWL coordinates corresponding to the current solutionstack.
        assert self.SolutionStack.isNotEmpty()

        PrecisionFactor = 1e-7 #TODO: unify all comparisons of coordinates

        X = []
        P = []
        for Sol in self.SolutionStack.Stack:
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

