
import numpy as np

from wasserstein_pwl_core.compressor import PWLcompressor
from wasserstein_pwl_core.pwl_distribution import PiecewiseLinearDistribution, EmpiricalDistributionFromSample, LinearInterpolation
from wasserstein_pwl_core.sample_characteristics import SampleCharacteristics

# ## INPUT Sample and segment to be bisected based on OLS criterium
# Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
# SegmentStart = 3
# SegmentEnd = 7

def RunAllTestCases():
    RunTestCases_SampleCharacteristics()
    RunTestCasesOLSCutPoint()
    print("All Test Cases successfully completed")

def RunTestCases_SampleCharacteristics():
    # testcases taken from excel sheet calculations

    Sample = [24., 26.75, 27.4, 27.45, 30, 30, 30, 30, 30, 30, 30.15, 30.5, 31.45, 32.7, 33.8]

    def Characteristics(Sol):
        LoMin = Sol.Calculate_StartXFromDelta(Sol.Delta_UpperBound)
        LoReg = Sol.Calculate_StartXFromDelta(Sol.Delta_Regression)
        LoMax = Sol.Calculate_StartXFromDelta(Sol.Delta_LowerBound)
        UpMin = Sol.Calculate_EndXFromDelta(Sol.Delta_LowerBound)
        UpReg = Sol.Calculate_EndXFromDelta(Sol.Delta_Regression)
        UpMax = Sol.Calculate_EndXFromDelta(Sol.Delta_UpperBound)
        BisecPt = Sol.BestBisectionPoint
        return dict(LoMin = LoMin, LoReg = LoReg, LoMax = LoMax, UpMin = UpMin, UpReg = UpReg, UpMax = UpMax, BisecPt = BisecPt)

    def CompareCharacteristics(eps, CA, CB):
        for key in CA:
            if abs(CA[key]-CB[key]) > eps:
                print('Values differ for '+key+': '+str(CA[key])+' , '+str(CB[key]))

    epsilon = 0.001

    SA = SampleCharacteristics(Sample, Accuracy = 0.1)
    Sol = SA.FindBestSolutionLine(10,14)
    ReferenceChar = dict(LoMin = 28.6358, LoReg = 29.4400, LoMax = 29.6433, UpMin = 33.7967, UpReg = 34.0000, UpMax = 34.8042, BisecPt = 12)# From Excel Testcase Generator
    CompareCharacteristics(epsilon,Characteristics(Sol),ReferenceChar)

    SA = SampleCharacteristics(Sample, Accuracy = 0.1)
    Sol = SA.FindBestSolutionLine(6,11)
    ReferenceChar = dict(LoMin = 29.1637, LoReg = 29.8625, LoMax = 30.1083, UpMin = 30.1083, UpReg = 30.3542, UpMax = 31.0530, BisecPt = 10)# From Excel Testcase Generator
    CompareCharacteristics(epsilon,Characteristics(Sol),ReferenceChar)

    SA = SampleCharacteristics(Sample, Accuracy = 0.1)
    Sol = SA.FindBestSolutionLine(0,3)
    ReferenceChar = dict(LoMin = 23.5747, LoReg = 24.3375, LoMax = 23.9484, UpMin = 28.8516, UpReg = 28.4625, UpMax = 29.2253, BisecPt = 0)# From Excel Testcase Generator
    CompareCharacteristics(epsilon,Characteristics(Sol),ReferenceChar)

    SA = SampleCharacteristics(Sample, Accuracy = 0.1)
    Sol = SA.FindBestSolutionLine(7,12)
    ReferenceChar = dict(LoMin = 29.0060, LoReg = 29.6083, LoMax = 30.1232, UpMin = 30.5768, UpReg = 31.0917, UpMax = 31.6940, BisecPt = 11)# From Excel Testcase Generator
    CompareCharacteristics(epsilon,Characteristics(Sol),ReferenceChar)

    SA = SampleCharacteristics(Sample, Accuracy = 0.01)
    Sol = SA.FindBestSolutionLine(11,13)
    ReferenceChar = dict(LoMin = 29.8384, LoReg = 30.0833, LoMax = 29.9341, UpMin = 33.1659, UpReg = 33.0167, UpMax = 33.2617, BisecPt = 12)# From Excel Testcase Generator
    CompareCharacteristics(epsilon,Characteristics(Sol),ReferenceChar)

def RunTestCasesOLSCutPoint():

    Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
    SegmentStart = 0
    SegmentEnd = 9
    #Original OptimalCutIndex was one integer bigger, i.e. OptimalCutIndex + 1
    #OptimalCutIndex = 6
    OptimalCutIndex = SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd, Bisection = "OLS").BestBisectionPoint
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)
    SegmentStart = 3
    SegmentEnd = 7
    #OptimalCutIndex = 6
    OptimalCutIndex = SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd, Bisection = "OLS").BestBisectionPoint
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

    SegmentStart = 6
    SegmentEnd = 9
    #OptimalCutIndex = 8
    OptimalCutIndex = SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd, Bisection = "OLS").BestBisectionPoint
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

    SegmentStart = 0
    SegmentEnd = 5
    #OptimalCutIndex = 2
    OptimalCutIndex = SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd, Bisection = "OLS").BestBisectionPoint
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

    # ----------------
    Sample = [1.0, 1.1, 1.2, 1.6, 4.3, 4.5, 4.6, 6, 6.1, 6.6,
              7.1, 13, 13.4, 16, 18.8, 22, 30, 32, 39, 40]
    SegmentStart = 0
    SegmentEnd = 19
    #OptimalCutIndex = 11
    OptimalCutIndex = SampleCharacteristics(Sample).FindBestSolutionLine(SegmentStart, SegmentEnd, Bisection = "OLS").BestBisectionPoint
    assert(FindOLSOptimalCutpoint(Sample, SegmentStart, SegmentEnd) == OptimalCutIndex)

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