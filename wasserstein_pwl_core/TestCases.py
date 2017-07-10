
import numpy as np

from wasserstein_pwl_core.compressor import PWLcompressor
from wasserstein_pwl_core.pwl_distribution import PiecewiseLinearDistribution, EmpiricalDistributionFromSample, LinearInterpolation
from wasserstein_pwl_core.sample_characteristics import SampleCharacteristics
from wasserstein_pwl_core.segment_stack import SegmentStack

def RunAllTestCases():
    RunTestCases_SegmentStack()
    RunTestCasesOLSCutPoint()
    RunTestCases_AtomOnEnforcedQuantile()
    print("All Test Cases successfully completed")

def RunTestCases_SegmentStack():
    """
    Example: if Sample = [ 24.  ,  26.75,  27.4 ,  27.45,  30, 30, 30, 30, 30, 30, 30.15,  30.5 ,  31.45,  32.7 ,  33.8 ]
    SS = SegmentStack(SubsampleApproximation([ 24.  ,  26.75,  27.4 ,  27.45,  30, 30, 30, 30, 30, 30, 30.15,  30.5 ,  31.45,  32.7 ,  33.8 ]),
                                              AtomDetection = (10, 0.1))
    gives
    SS.Stack = [Segment(0,3), Segment(4,9), Segment(10,14)]

    The two options AtomDetection = True and AtomDetection = (10, 0.1) are tested. The first one does not lead to atom
    detection in this case because the sample size is too small for it. The second detect the atom because the sample size
    for atom detection is 10
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
    # I.e., the result of this compression should be  [0.083, 4.583, 5.0, 5.0, 6.333, 12.333],   [0.0, 0.3, 0.3, 0.7, 0.7, 1.0]
    # and NOT  [0.083, 4.583, 5.0, 5.0, 5.0, 6.333, 12.333],   [0.0, 0.3, 0.3, 0.5, 0.7, 0.7, 1.0]
    Sample = [1, 2, 4, 5, 5, 5, 5, 7, 10, 11]
    Accuracy = 0.3
    CompressedSample = PWLcompressor(Sample, AccuracyMode = "Absolute", AccuracyParameter = Accuracy,
                                     AtomDetection= (10, 0.1), CheckStrictWasserstein = False)
    PWL = CompressedSample.Result
    AssertListsAreAlmostEqual(PWL['PWLX'],[0.083, 4.583, 5.0, 5.0, 6.333, 12.333], 1e-3 )
    AssertListsAreAlmostEqual(PWL['PWLY'],[0.0, 0.3, 0.3, 0.7, 0.7, 1.0], 1e-6 )


def RunTestCasesOLSCutPoint():
    # Verify the bisection point when Bisection = OLS is selected. The file OLSbisectiontests.py in the code of the other
    # compressor provide the reference values of the OptimalCutIndex
    # The point found with the OLSbisectiontests.py file used other index convention here the bisection point is actually equal to
    # OptimalCutIndex - 1
    Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
    SegmentStart = 0
    SegmentEnd = 9
    OptimalCutIndex = 6
    assert(SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd).BestBisectionPoint == OptimalCutIndex - 1)

    SegmentStart = 3
    SegmentEnd = 7
    OptimalCutIndex = 6
    assert(SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd).BestBisectionPoint == OptimalCutIndex - 1)

    SegmentStart = 6
    SegmentEnd = 9
    OptimalCutIndex = 8
    assert(SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd).BestBisectionPoint == OptimalCutIndex - 1)

    SegmentStart = 0
    SegmentEnd = 5
    OptimalCutIndex = 2
    assert(SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd).BestBisectionPoint == OptimalCutIndex - 1)

    # ----------------
    Sample = [1.0, 1.1, 1.2, 1.6, 4.3, 4.5, 4.6, 6, 6.1, 6.6,
              7.1, 13, 13.4, 16, 18.8, 22, 30, 32, 39, 40]
    SegmentStart = 0
    SegmentEnd = 19
    OptimalCutIndex = 11
    assert(SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd).BestBisectionPoint == OptimalCutIndex - 1)

    #App = PWLcompressor(Sample, Accuracy = 1, PlotIntermediate=True)
    SegmentStart = 0
    SegmentEnd = 10
    OptimalCutIndex = 4
    assert(SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd).BestBisectionPoint == OptimalCutIndex - 1)

    SegmentStart = 6
    SegmentEnd = 17
    OptimalCutIndex = 16
    assert(SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd).BestBisectionPoint == OptimalCutIndex - 1)

    Sample = [0 - 0.2, 1 - 0.2, 2 - 0.6, 3 + 0.2, 4 + 0.3, 5 - 0.1, 6 - 0.2, 7 - 0.4, 8 + 0.5, 9 - 0.4,
              10 + 0.1, 11 + 0.4, 12 - 0.3, 13 - 0.5, 14 - 0.8, 15 - 0.9, 16 - 0.5, 17 - 0.4, 18 + 0.7, 19 + 0.2,
              20 - 0.2, 21 + 0.1, 22 - 0.3, 23 - 0.2, 24 + 0.0, 25 - 0.1, 26 + 0.6, 27 + 0.3, 28 - 0.3, 29 + 0.2, ]

    for (SegmentStart, SegmentEnd, OptimalCutIndex) in [[23, 26, 26], [4, 24, 18], [10, 26, 16], [6, 13, 8],
                                                         [8, 27, 18], [7, 18, 18], [12, 17, 16], [1, 11, 3],
                                                         [11, 24, 18], [1, 7, 3], [3, 22, 18], [16, 21, 18],
                                                         [6, 9, 8], [16, 21, 18], [12, 19, 18], [10, 19, 18], ]:
        assert (SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd).BestBisectionPoint == OptimalCutIndex - 1)





def AssertListsAreAlmostEqual(L1, L2, Epsilon):
    assert len(L1) == len(L2)
    for x, y in zip(L1, L2):
        assert (abs(x-y) <= Epsilon)
        # raise Exception('bla')

if __name__ == "__main__":
    RunAllTestCases()