
import numpy as np
import matplotlib.pyplot as plt

from pwl_compressor_core.compressor import PWLcompressor
from pwl_compressor_core.pwl_distribution import PiecewiseLinearDistribution, EmpiricalDistributionFromSample, LinearInterpolation


# ## INPUT Sample and segment to be bisected based on OLS criterium
# Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
# SegmentStart = 3
# SegmentEnd = 7

def FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd):
    ## calculate optimal cutting point based on OLS
    SampleSize = len(Sample)
    SegmentStartCoord = SegmentStart/SampleSize
    SegmentEndCoord = (SegmentEnd + 1)/SampleSize
    pwl_sample = EmpiricalDistributionFromSample(Sample)

    nrpoints = max(10000, SampleSize*10)
    discretizerpoints = np.linspace(0.5/nrpoints,1-0.5/nrpoints,nrpoints)
    discretizedsample = pwl_sample.quantile(discretizerpoints)

    # cutpoints = np.linspace(0.1,0.9,9)
    # L2Dist = np.zeros_like(cutpoints)
    # L1Dist = np.zeros_like(cutpoints)
    cutpoints = []
    L2Dist = []

    for i in range(SegmentStart, SegmentEnd):
        cutpoint = (i+1)/SampleSize
        # this gives a L2 regression without smoothing and without enforcing admissibility
        CompressedSample = PWLcompressor(Sample, EnforcedInterpolationQuantiles = [SegmentStartCoord,cutpoint,SegmentEndCoord],
                                         RemoveNegativeJumps = False, MakePWLsmoother = False, CheckStrictAdmissibility = False, Accuracy = 1e250, AtomDetection = True)

        pwl_approx_x, pwl_approx_y = CompressedSample.Result['PWLX'], CompressedSample.Result['PWLY']

        discretizedpwl = LinearInterpolation(discretizerpoints, pwl_approx_y, pwl_approx_x, DiscontinuitiesBehaviour = 'LeftContinuous')
        ## plot intermediate steps
        # plt.plot(discretizedpwl, discretizerpoints, linewidth=1.0, linestyle = '-', color = 'r')
        # pwl_sample.plot()
        # plt.show()

        cutpoints.append(cutpoint)
        L2Dist.append(np.sum(np.abs(discretizedpwl - discretizedsample)**2)/nrpoints)
        # L1Dist[i] = np.sum(np.abs(discretizedpwl - discretizedsample))/nrpoints

        ## check calculation
        # discretizerpointsFirstSegment = np.linspace(0.5/nrpoints, cutpoints[i] - 0.5/nrpoints,nrpoints)
        # discretizedsampleFirstSegment = pwl_sample.quantile(discretizerpointsFirstSegment)
        # discretizedpwlFirstSegment = LinearInterpolation(discretizerpointsFirstSegment, pwl_approx_y, pwl_approx_x, DiscontinuitiesBehaviour = 'LeftContinuous')
        # mupart = np.ones_like(discretizedpwlFirstSegment)*np.mean(discretizedpwlFirstSegment)
        # deltapart = discretizedpwlFirstSegment - mupart
        # assert(np.sum((discretizedpwlFirstSegment - mupart - deltapart))<1e-5)
        # FirstSegmentL2Dist = np.sum((discretizedsampleFirstSegment - discretizedpwlFirstSegment)**2)/nrpoints*cutpoint
        # FirstSegmentL2_PartA = np.sum((discretizedsampleFirstSegment - mupart)**2)/nrpoints*cutpoint
        # FirstSegmentL2_PartB = np.sum(((-2)*(discretizedsampleFirstSegment - mupart)*deltapart))/nrpoints*cutpoint
        # FirstSegmentL2_PartC = np.sum(deltapart**2)/nrpoints*cutpoint
        # print('\n ============================== \n ' +
        #       '\n Cutpoint ' + str(cutpoint) +
        #       '\n FirstSegmentL2Dist  ' + str(FirstSegmentL2Dist) +
        #       '\n FirstSegmentL2_PartA  ' + str(FirstSegmentL2_PartA) + # D1 in Anna Thesis
        #       '\n FirstSegmentL2_PartB  ' + str(FirstSegmentL2_PartB) + # D2 in Anna Thesis
        #       '\n FirstSegmentL2_PartC  ' + str(FirstSegmentL2_PartC) + # D3 in Anna Thesis
        #       '\n check (should be zero):   ' + str(FirstSegmentL2Dist - FirstSegmentL2_PartA - FirstSegmentL2_PartB - FirstSegmentL2_PartC)
        # )



    OptimalCutIndex = np.argmin(L2Dist) + SegmentStart + 1
    OptimalCutPoint = cutpoints[np.argmin(L2Dist)]

    print('OptimalCutIndex: '+str(OptimalCutIndex))
    print('OptimalCutPoint: '+str(OptimalCutPoint))

    # plt.plot(cutpoints, L2Dist, linewidth=2.0, linestyle = '-', marker = 'o', color = 'r')
    # # plt.plot(cutpoints, L1Dist, linewidth=2.0, linestyle = '-', marker = 'o', color = 'g')
    # plt.show()

    return OptimalCutIndex


if __name__ == "__main__":

    Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
    SegmentStart = 0
    SegmentEnd = 9
    OptimalCutIndex = 6
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

    SegmentStart = 3
    SegmentEnd = 7
    OptimalCutIndex = 6
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

    SegmentStart = 6
    SegmentEnd = 9
    OptimalCutIndex = 8
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

    SegmentStart = 0
    SegmentEnd = 5
    OptimalCutIndex = 2
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

    # ----------------
    Sample = [1.0, 1.1, 1.2, 1.6, 4.3, 4.5, 4.6, 6, 6.1, 6.6,
              7.1, 13, 13.4, 16, 18.8, 22, 30, 32, 39, 40]
    SegmentStart = 0
    SegmentEnd = 19
    OptimalCutIndex = 11
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

    SegmentStart = 0
    SegmentEnd = 10
    OptimalCutIndex = 4
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

    SegmentStart = 6
    SegmentEnd = 17
    OptimalCutIndex = 16
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)
