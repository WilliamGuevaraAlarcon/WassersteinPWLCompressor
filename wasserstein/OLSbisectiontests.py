
import numpy as np
import matplotlib.pyplot as plt

from wasserstein_pwl_core.compressor import PWLcompressor
from wasserstein_pwl_core.pwl_distribution import  EmpiricalDistributionFromSample, LinearInterpolation


# ## INPUT Sample and segment to be bisected based on OLS criterium
# Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
# SegmentStart = 3
# SegmentEnd = 7

def FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd, plotresults = False):
    ## calculate optimal cutting point based on OLS
    SampleSize = len(Sample)
    SegmentStartCoord = SegmentStart/SampleSize
    SegmentEndCoord = (SegmentEnd + 1)/SampleSize
    pwl_sample = EmpiricalDistributionFromSample(Sample)

    nrpoints = max(100000, SampleSize*10)
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
                                         RemoveNegativeJumps = False, Accuracy = 1e250, AtomDetection = True)

        pwl_approx_x, pwl_approx_y = CompressedSample.Result['PWLX'], CompressedSample.Result['PWLY']

        discretizedpwl = LinearInterpolation(discretizerpoints, pwl_approx_y, pwl_approx_x, DiscontinuitiesBehaviour = 'LeftContinuous')
        ## plot intermediate steps
        if plotresults:
            plt.plot(discretizedpwl, discretizerpoints, linewidth=1.0, linestyle = '-', color = 'r')
            # pwl_sample.plot()
            plt.show()

        cutpoints.append(cutpoint)
        L2Dist.append(np.sum(np.abs(discretizedpwl - discretizedsample)**2)/nrpoints)
        # L1Dist[i] = np.sum(np.abs(discretizedpwl - discretizedsample))/nrpoints


    OptimalCutIndex = np.argmin(L2Dist) + SegmentStart + 1
    OptimalCutPoint = cutpoints[np.argmin(L2Dist)]

    if plotresults:
        print('OptimalCutIndex: '+str(OptimalCutIndex))
        print('OptimalCutPoint: '+str(OptimalCutPoint))
        plt.plot(cutpoints, L2Dist, linewidth=2.0, linestyle = '-', marker = 'o', color = 'r')
        # plt.plot(cutpoints, L1Dist, linewidth=2.0, linestyle = '-', marker = 'o', color = 'g')
        plt.show()

    return OptimalCutIndex


if __name__ == "__main__":

    # Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
    # SegmentStart = 0
    # SegmentEnd = 9
    # OptimalCutIndex = 6
    # assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)
    #
    # SegmentStart = 3
    # SegmentEnd = 7
    # OptimalCutIndex = 6
    # assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)
    #
    # SegmentStart = 6
    # SegmentEnd = 9
    # OptimalCutIndex = 8
    # assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)
    #
    # SegmentStart = 0
    # SegmentEnd = 5
    # OptimalCutIndex = 2
    # assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)
    #
    # # ----------------
    Sample = [1.0, 1.1, 1.2, 1.6, 4.3, 4.5, 4.6, 6, 6.1, 6.6,
              7.1, 13, 13.4, 16, 18.8, 22, 30, 32, 39, 40]
    # SegmentStart = 0
    # SegmentEnd = 19
    # OptimalCutIndex = 11
    # assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)
    #
    SegmentStart = 0
    SegmentEnd = 10
    OptimalCutIndex = 4
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

    SegmentStart = 6
    SegmentEnd = 17
    OptimalCutIndex = 16
    w = FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd, plotresults=False)
    assert(w == OptimalCutIndex)

    # ----------------

    # Sample = [ 0- 0.2,  1- 0.2,  2- 0.6,  3+ 0.2,  4+ 0.3,  5- 0.1,  6- 0.2,  7- 0.4,  8+ 0.5,  9- 0.4,
    #           10+ 0.1, 11+ 0.4, 12- 0.3, 13- 0.5, 14- 0.8, 15- 0.9, 16- 0.5, 17- 0.4, 18+ 0.7, 19+ 0.2,
    #           20- 0.2, 21+ 0.1, 22- 0.3, 23- 0.2, 24+ 0.0, 25- 0.1, 26+ 0.6, 27+ 0.3, 28- 0.3, 29+ 0.2,]
    #
    # for (SegmentStart, SegmentEnd, OptimalCutIndex) in [[23, 26, 26], [4, 24, 18], [10, 26, 16], [6, 13, 8],
    #                                                     [8, 27, 18], [7, 18, 18], [12, 17, 16], [1, 11, 3],
    #                                                     [11, 24, 18], [1, 7, 3], [3, 22, 18], [16, 21, 18],
    #                                                     [6, 9, 8], [16, 21, 18], [12, 19, 18], [10, 19, 18], ]:
    #     assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)


    # plt.plot(Sample, np.linspace(0,1,30), linewidth=2.0, linestyle = '-', marker = 'o', color = 'r')
    #     # plt.plot(cutpoints, L1Dist, linewidth=2.0, linestyle = '-', marker = 'o', color = 'g')
    # plt.show()
    # for i in range(20):
    #     (SegmentStart, SegmentEnd) = sorted(np.random.randint(0,29,2))
    #     cp = FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd, plotresults=False)
    #     print('['+str(SegmentStart)+', '+str(SegmentEnd)+', '+str(cp)+'], ')

