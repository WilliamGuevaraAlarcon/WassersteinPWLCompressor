
import numpy as np

from wasserstein_pwl_core.compressor import PWLcompressor
from wasserstein_pwl_core.pwl_distribution import PiecewiseLinearDistribution, EmpiricalDistributionFromSample, LinearInterpolation
from wasserstein_pwl_core.sample_characteristics import SampleCharacteristics
from wasserstein_pwl_core.segment_stack import SegmentStack

# ## INPUT Sample and segment to be bisected based on OLS criterium
# Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
# SegmentStart = 3
# SegmentEnd = 7

def RunAllTestCases():
    #RunTestCases_SegmentStack()
    #RunTestCases_AtomOnEnforcedQuantile()
    RunTestCasesOLSCutPoint()
    print("All Test Cases successfully completed")

def RunTestCases_SegmentStack():
    """
    Example: if Sample = [ 24.  ,  26.75,  27.4 ,  27.45,  30, 30, 30, 30, 30, 30, 30.15,  30.5 ,  31.45,  32.7 ,  33.8 ]
    SS = SegmentStack(SubsampleApproximation([ 24.  ,  26.75,  27.4 ,  27.45,  30, 30, 30, 30, 30, 30, 30.15,  30.5 ,  31.45,  32.7 ,  33.8 ]),
                                              AtomDetection = (10, 0.1))
    gives
    SS.Stack = [Segment(0,3), Segment(4,9), Segment(10,14)]

    The two options AtomDetection = True and AtomDetection = (10, 0.1) are tested. The first one does not lead to atom
    detection in this case because the sample size is too small for it
    """
    #          0        1       2       3      4   5   6   7   8   9   10      11      12      13      14
    Sample = [ 24.,  26.75,  27.4 ,  27.45,  30, 30, 30, 30, 30, 30, 30.15,  30.5 ,  31.45,  32.7 ,  33.8 ]
    SC = SampleCharacteristics(Sample)
    SS1 = SegmentStack(SC, AtomDetection = True)
    SS2 = SegmentStack(SC, AtomDetection = (10, 0.1))

    assert (len(SS2.Stack) == 3)
    for i in range(3):
        SI = SS2.Stack.pop()
        assert( SI.SampleSet_Start in [0,4,10] )
        assert( SI.SampleSet_End in [3,9,14] )

    assert (len(SS1.Stack) == 1)
    SI = SS1.Stack.pop()
    assert (SI.SampleSet_Start == 0)
    assert (SI.SampleSet_End == 14)

def RunTestCases_AtomOnEnforcedQuantile():
    # ensure that an enforced quantile on an atom is removed from the solution. the point is redundant.
    # I.e., the result of this compression should be  [0.333, 4.333, 5.0, 5.0, 6.666, 12],   [0.0, 0.3, 0.3, 0.7, 0.7, 1.0]
    # and NOT  [0.333, 4.333, 5.0, 5.0, 5.0, 6.666, 12],   [0.0, 0.3, 0.3, 0.5, 0.7, 0.7, 1.0]
    Sample = [1, 2, 4, 5 , 5, 5, 5, 7, 10, 11]
    Accuracy = 0.3
    CompressedSample = PWLcompressor(Sample, Accuracy = Accuracy,
                                     AtomDetection= (10, 0.1),
                                     EnforcedInterpolationQuantiles=[0.5])
    PWL = CompressedSample.Result
    AssertListsAreAlmostEqual(PWL['PWLX'],[0.333, 4.333, 5.0, 5.0, 6.666, 12.0], 1e-3 )
    AssertListsAreAlmostEqual(PWL['PWLY'],[0.0, 0.3, 0.3, 0.7, 0.7, 1.0], 1e-6 )


def RunTestCasesOLSCutPoint():

    # Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
    # SegmentStart = 0
    # SegmentEnd = 9
    # #Original OptimalCutIndex was one integer bigger, i.e. OptimalCutIndex + 1
    # #OptimalCutIndex = 6
    # OptimalCutIndex = SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd, Bisection = "OLS").BestBisectionPoint
    # assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)
    #
    # SegmentStart = 3
    # SegmentEnd = 7
    # #OptimalCutIndex = 6
    # OptimalCutIndex = SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd, Bisection = "OLS").BestBisectionPoint
    # assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)
    #
    # SegmentStart = 6
    # SegmentEnd = 9
    # #OptimalCutIndex = 8
    # OptimalCutIndex = SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd, Bisection = "OLS").BestBisectionPoint
    # assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)
    #
    # SegmentStart = 0
    # SegmentEnd = 5
    # #OptimalCutIndex = 2
    # OptimalCutIndex = SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd, Bisection = "OLS").BestBisectionPoint
    # assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

    # ----------------
    Sample = [1.0, 1.1, 1.2, 1.6, 4.3, 4.5, 4.6, 6, 6.1, 6.6,
              7.1, 13, 13.4, 16, 18.8, 22, 30, 32, 39, 40]
    SegmentStart = 0
    SegmentEnd = 19
    #OptimalCutIndex = 11
    OptimalCutIndex = SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd, Bisection = "OLS").BestBisectionPoint
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

    App = PWLcompressor(Sample, Accuracy = 1, PlotIntermediate=True)
    SegmentStart = 0
    SegmentEnd = 10
    #OptimalCutIndex = 4
    OptimalCutIndex = SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd, Bisection = "OLS").BestBisectionPoint
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

    SegmentStart = 6
    SegmentEnd = 17
    #OptimalCutIndex = 16
    OptimalCutIndex = SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd, Bisection = "OLS").BestBisectionPoint
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)


def FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd):
    ## calculate optimal cutting point based on OLS
    SampleSize = len(Sample)
    SegmentStartCoord = SegmentStart/SampleSize
    SegmentEndCoord = (SegmentEnd + 1)/SampleSize
    pwl_sample = EmpiricalDistributionFromSample(Sample)

    nrpoints = max(10000, SampleSize*10)
    discretizerpoints = np.linspace(0.5/nrpoints,1-0.5/nrpoints,nrpoints)
    discretizedsample = pwl_sample.quantile(discretizerpoints)

    cutpoints = []
    L2Dist = []

    for i in range(SegmentStart, SegmentEnd):
        cutpoint = (i+1)/SampleSize
        # this gives a L2 regression without smoothing and without enforcing admissibility
        CompressedSample = PWLcompressor(Sample, EnforcedInterpolationQuantiles = [SegmentStartCoord,cutpoint,SegmentEndCoord],
                                         RemoveNegativeJumps = False, Accuracy = 1e250, AtomDetection = True)

        pwl_approx_x, pwl_approx_y = CompressedSample.Result['PWLX'], CompressedSample.Result['PWLY']

        discretizedpwl = LinearInterpolation(discretizerpoints, pwl_approx_y, pwl_approx_x, DiscontinuitiesBehaviour = 'LeftContinuous')
        ## plot intermediate steps
        # plt.plot(discretizedpwl, discretizerpoints, linewidth=1.0, linestyle = '-', color = 'r')
        # pwl_sample.plot()
        # plt.show()

        cutpoints.append(cutpoint)
        L2Dist.append(np.sum(np.abs(discretizedpwl - discretizedsample)**2)/nrpoints)

    #OptimalCutIndex = np.argmin(L2Dist) + SegmentStart +1
    OptimalCutIndex = np.argmin(L2Dist) + SegmentStart
    OptimalCutPoint = cutpoints[np.argmin(L2Dist)]

    print('OptimalCutIndex: '+str(OptimalCutIndex))
    print('OptimalCutPoint: '+str(OptimalCutPoint))


    return OptimalCutIndex


def AssertListsAreAlmostEqual(L1, L2, Epsilon):
    assert len(L1) == len(L2)
    for x, y in zip(L1, L2):
        assert (abs(x-y) <= Epsilon)
        # raise Exception('bla')

if __name__ == "__main__":
    RunAllTestCases()