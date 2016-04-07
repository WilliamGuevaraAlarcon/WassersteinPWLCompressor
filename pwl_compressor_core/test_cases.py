__author__ = 'chuarph'

import numpy as np
from pwl_compressor_core.sample_characteristics import SampleCharacteristics
from pwl_compressor_core.solve_simple_linear_inequality_list import SolveSimpleLinearInequailityList
from pwl_compressor_core.problem_stack import ProblemStack
from pwl_compressor_core.compressor import PWLcompressor
# from tempfile import TemporaryFile


from pwl_compressor_core.pwl_distribution import PiecewiseLinearDistribution
import matplotlib.pyplot as plt


def RunAllTestCases():
    #this function is called later...    if __name__ == "__main__":
    RunTestCases_SampleCharacteristics()
    RunTestCases_SolveInequailityList()
    RunTestCases_ProblemStack()
    RunTestCases_NegativeIncrementRemoval()
    RunTestCases_NegativeIncrementRemovalWithEnforcedInterpolationQuantile()
    RunTestCases_Smoothing()
    RunTestCases_SmoothingWithEnforcedInterpolationQuantile()
    RunTestCases_FullAlgorithm()
    RunTestCases_SimpleExemple()
    RunTestCases_AtomOnEnforcedQuantile()
    RunTestCases_StrictAdmissibility()
    RunTestCases_StrictAdmissibility_Second()
    RunTestCases_StrictAdmissibility_Third()

def RunTestCases_SampleCharacteristics():
    # testcases taken from excel sheet calculations

    Sample = [ 24.  ,  26.75,  27.4 ,  27.45,  30, 30, 30, 30, 30, 30, 30.15,  30.5 ,  31.45,  32.7 ,  33.8 ]

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


def RunTestCases_SolveInequailityList():
    assert (SolveSimpleLinearInequailityList(np.asarray([2,2]),
                                 np.asarray([2,2]),
                                 np.asarray([3,3]), 1e-10)) == (True, 1.0, 1.5)

    assert (SolveSimpleLinearInequailityList(np.asarray([1,2]),
                                 np.asarray([2,2]),
                                 np.asarray([4,3]), 1e-10)) == (True, 1.0, 1.5)

    assert (SolveSimpleLinearInequailityList(np.asarray([1,1]),
                                 np.asarray([0.5,1]),
                                 np.asarray([4,30]), 1e-10)) == (True, 2.0, 8)

    assert (SolveSimpleLinearInequailityList(np.asarray([1,2,-2]),
                                 np.asarray([-2,-2,0]),
                                 np.asarray([4,3,0]), 1e-10)) == (False, None, None)# (True, -1.5, -1.0)

    assert (SolveSimpleLinearInequailityList(np.asarray([1,2,2]), #check if np.any(A[B_EqualToZero] > 0) | np.any(C[B_EqualToZero] < 0)
                                 np.asarray([-2,-2,0]),
                                 np.asarray([4,3,1]), 1e-10)) == (False, None, None)

    assert (SolveSimpleLinearInequailityList(np.asarray([4,3]),
                                 np.asarray([-2,-2]),
                                 np.asarray([1,2]), 1e-10)) == (False, None, None)

    assert (SolveSimpleLinearInequailityList(np.asarray([-1,-2]),
                                 np.asarray([-2,-2]),
                                 np.asarray([4,3]), 1e-10)) == (True, 0, 0.5) #(True, -1.5, 0.5)

    assert (SolveSimpleLinearInequailityList(np.asarray([-1,-2,-1,0]),
                                 np.asarray([-2,-2,2,2]),
                                 np.asarray([4,3,4,3]), 1e-10)) == (True, 0, 0.5)


def RunTestCases_ProblemStack():
    """
    Example: if Sample = [ 24.  ,  26.75,  27.4 ,  27.45,  30, 30, 30, 30, 30, 30, 30.15,  30.5 ,  31.45,  32.7 ,  33.8 ]
    PS = ProblemStack(SubsampleApproximation([ 24.  ,  26.75,  27.4 ,  27.45,  30, 30, 30, 30, 30, 30, 30.15,  30.5 ,  31.45,  32.7 ,  33.8 ]),
                                              AtomDetectionMinimumSampleSize = 10, RelativeAtomDetectionThreshold = 0.1)
    gives
    PS.Stack = [ProblemInterval(0,3), ProblemInterval(4,9), ProblemInterval(10,14)]
    """
    #          0        1       2       3      4   5   6   7   8   9   10      11      12      13      14
    Sample = [ 24.  ,  26.75,  27.4 ,  27.45,  30, 30, 30, 30, 30, 30, 30.15,  30.5 ,  31.45,  32.7 ,  33.8 ]
    SC = SampleCharacteristics(Sample,0.1)
    PS = ProblemStack(SC,AtomDetection=True, AtomDetectionMinimumSampleSize = 10, RelativeAtomDetectionThreshold = 0.1)
    assert (len(PS.Stack) == 3)
    for i in range(3):
        PI = PS.pop()
        assert( PI.SampleSet_Start in [0,4,10] )
        assert( PI.SampleSet_End in [3,9,14] )


def AssertListsAreAlmostEqual(L1, L2, Epsilon):
    assert len(L1) == len(L2)
    for x, y in zip(L1, L2):
        assert (abs(x-y) <= Epsilon)
        # raise Exception('bla')


def RunTestCases_NegativeIncrementRemoval():
    #test NegativeIncrementRemoval
    Sample = [6,6,6,6,6,6,6,6,6,6,6,10,13,15,18,21,25]
    Accuracy = 0.03
    #result: [6.0, 6.0, 9.6452941176470581, 13.354705882352942, 26.145294117647058] [0.0, 0.6470588235294118, 0.6470588235294118, 0.76470588235294112, 1.0]
    CompressedSample = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False, Accuracy = Accuracy)
    PWL = CompressedSample.Result
    print(PWL)
    AssertListsAreAlmostEqual(PWL['PWLX'], [6.0, 6.0, 9.6452941, 13.3547058, 26.1452941], 1e-6 )
    AssertListsAreAlmostEqual(PWL['PWLY'], [0.0, 0.6470588, 0.6470588, 0.7647058, 1.0], 1e-6 )


def RunTestCases_NegativeIncrementRemovalWithEnforcedInterpolationQuantile():
    # Example 6.11 as in paper
    Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
    Accuracy = 0.25
    CompressedSample = PWLcompressor(Sample, CheckStrictAdmissibility = False, Accuracy = Accuracy, EnforcedInterpolationQuantiles=[0.3])
    PWL = CompressedSample.Result
    AssertListsAreAlmostEqual(PWL['PWLX'],[0.933, 3.667, 18.875], 1e-3 )
    AssertListsAreAlmostEqual(PWL['PWLY'],[0, 0.3, 1], 1e-6 )


def RunTestCases_Smoothing():
    #test smoothing
    Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 11, 13.4, 16, 18.8]
    Accuracy = 0.11
    #result: [-0.3854547619047608, 10.556883333333333, 21.576450000000001] [0.0, 0.69999999999999996, 1.0]
    CompressedSample = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False, Accuracy = Accuracy)
    PWL = CompressedSample.Result
    AssertListsAreAlmostEqual(PWL['PWLX'],[-0.3854547, 10.5568833, 21.57645], 1e-6 )
    AssertListsAreAlmostEqual(PWL['PWLY'],[0, 0.7, 1], 1e-6 )


def RunTestCases_SmoothingWithEnforcedInterpolationQuantile():
    # Example 6.15 as in paper
    Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
    Accuracy = 0.25
    CompressedSample = PWLcompressor(Sample, CheckStrictAdmissibility = False, Accuracy = Accuracy, EnforcedInterpolationQuantiles=[0.6])
    PWL = CompressedSample.Result #plot('g', ShowPlot = False)
    AssertListsAreAlmostEqual(PWL['PWLX'],[-0.730, 8.930, 21.670], 1e-3 )
    AssertListsAreAlmostEqual(PWL['PWLY'],[0, 0.6, 1], 1e-6 )


def RunTestCases_FullAlgorithm():
    # Example 6.13 as in paper
    Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
    Accuracy = 0.08
    CompressedSample = PWLcompressor(Sample, CheckStrictAdmissibility = False, Accuracy = Accuracy, EnforcedInterpolationQuantiles=[0.3])
    PWL = CompressedSample.Result
    AssertListsAreAlmostEqual(PWL['PWLX'],[0.233, 4.367, 7.433, 11.55, 19.05], 1e-3 )
    AssertListsAreAlmostEqual(PWL['PWLY'],[0.0, 0.3, 0.6, 0.6, 1.0], 1e-6 )


def RunTestCases_SimpleExemple():
    # as in C# code
    Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]
    Accuracy = 0.01
    CompressedSample = PWLcompressor(Sample, CheckStrictAdmissibility = False, Accuracy = Accuracy)
    PWL = CompressedSample.Result
    AssertListsAreAlmostEqual(PWL['PWLX'],[0.85, 1.75, 3.8656, 5.0344, 8.0656, 12.9, 13.5, 16, 16, 18.8, 18.8], 1e-3 )
    AssertListsAreAlmostEqual(PWL['PWLY'],[0, 0.2, 0.2, 0.4, 0.6, 0.6, 0.8, 0.8, 0.9, 0.9, 1], 1e-6 )


def RunTestCases_AtomOnEnforcedQuantile():
    # ensure that an enforced quantile on an atom is removed from the solution. the point is redundant.
    # I.e., the result of this compression should be  [-0.333, 5.0, 5.0, 6.666, 12],   [0.0, 0.3, 0.7, 0.7, 1.0]
    # and NOT  [-0.333, 5.0, 5.0, 5.0, 6.666, 12],   [0.0, 0.3, 0.5, 0.7, 0.7, 1.0]
    Sample = [1, 2, 4, 5 , 5, 5, 5, 7, 10, 11]
    Accuracy = 0.1
    CompressedSample = PWLcompressor(Sample, CheckStrictAdmissibility = False, Accuracy = Accuracy,
                                     AtomDetection=True, AtomDetectionMinimumSampleSize = 10, RelativeAtomDetectionThreshold = 0.1,
                                     EnforcedInterpolationQuantiles=[0.5])
    PWL = CompressedSample.Result
    AssertListsAreAlmostEqual(PWL['PWLX'],[-0.333, 5.0, 5.0, 6.666, 12], 1e-3 )
    AssertListsAreAlmostEqual(PWL['PWLY'],[0.0, 0.3, 0.7, 0.7, 1.0], 1e-6 )


def RunTestCases_StrictAdmissibility():
    #test StrictAdmissibility
    Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 11, 13.4, 16, 18.8]
    Accuracy = 0.12
    CompressedSampleStrict    = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = True,  Accuracy = Accuracy)
    CompressedSampleNonStrict = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False, Accuracy = Accuracy)
    pwlStrict = CompressedSampleStrict.Result
    pwlNonStrict = CompressedSampleNonStrict.Result

    AssertListsAreAlmostEqual(pwlStrict['PWLX'],[0.433, 7.767, 11.0, 11.0, 18.4, 18.8, 18.8], 1e-3 )
    AssertListsAreAlmostEqual(pwlStrict['PWLY'],[0.0, 0.6, 0.6, 0.7, 0.9, 0.9, 1.0], 1e-6 )
    AssertListsAreAlmostEqual(pwlNonStrict['PWLX'],[-0.429, 10.6, 21.533], 1e-3 )
    AssertListsAreAlmostEqual(pwlNonStrict['PWLY'],[0.0, 0.7, 1.0], 1e-6 )


def RunTestCases_StrictAdmissibility_Second():
    #test StrictAdmissibility
    Sample = [5.345, 7.343, 11.6, 5.539, 5.199, 7.402, 5.021, 5.212, 5.074, 8.391,
            5.642, 6.223, 6.241, 7.363, 14.392, 6.732, 5.292, 9.049, 8.82, 10.113,
            10.712, 28.547, 10.295, 9.305, 15.707, 5.966, 9.698, 18.212, 6.189, 12.805]

    Accuracy = 0.34
    CompressedSampleStrict    = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = True,  Accuracy = Accuracy)
    CompressedSampleNonStrict = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False, Accuracy = Accuracy)
    pwlStrict = CompressedSampleStrict.Result
    pwlNonStrict = CompressedSampleNonStrict.Result

    AssertListsAreAlmostEqual(pwlStrict['PWLX'],[3.764, 10.583, 12.002, 13.863, 27.781], 1e-3 )
    AssertListsAreAlmostEqual(pwlStrict['PWLY'],[0.0, 0.8, 0.8, 0.9, 1.0], 1e-6 )
    AssertListsAreAlmostEqual(pwlNonStrict['PWLX'],[3.544, 12.084, 29.561], 1e-3 )
    AssertListsAreAlmostEqual(pwlNonStrict['PWLY'],[0.0, 0.9, 1.0], 1e-6 )

def RunTestCases_StrictAdmissibility_Third():
    # test StrictAdmissibility
    # This test also contains segments which are not at the beginning or end, and which violate the strict admissibility condtions
    Sample = [5.021, 5.074,5.199,5.212,5.292,5.345,5.539,5.642,5.966,6.189,46.223,
              46.241,46.732,47.343,47.363,47.402,48.391,48.82,49.049,49.305,69.698,
              70.113,70.295,70.712,71.6,72.805,74.392,75.707,78.212,78.547]

    Accuracy = 0.00089
    CompressedSampleStrict    = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = True,  Accuracy = Accuracy)
    CompressedSampleNonStrict = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False, Accuracy = Accuracy)
    pwlStrict = CompressedSampleStrict.Result
    pwlNonStrict = CompressedSampleNonStrict.Result

    AssertListsAreAlmostEqual(pwlStrict['PWLX'],[5.021, 5.021, 5.074, 5.074, 5.086, 5.438, 5.531, 5.65, 6.505, 46.061, 47.707, 48.334, 49.448, 69.6, 70.809, 71.6, 71.6, 72.805, 72.805, 74.392, 74.392, 75.707, 75.707, 78.212, 78.212, 78.547, 78.547], 1e-3 )
    AssertListsAreAlmostEqual(pwlStrict['PWLY'],[0.0, 0.033, 0.033, 0.067, 0.067, 0.2, 0.2, 0.267, 0.333, 0.333, 0.533, 0.533, 0.667, 0.667, 0.8, 0.8, 0.833, 0.833, 0.867, 0.867, 0.9, 0.9, 0.933, 0.933, 0.967, 0.967, 1.0], 1e-3 )
    AssertListsAreAlmostEqual(pwlNonStrict['PWLX'],[4.949, 5.632, 6.523, 45.805, 49.569, 69.6, 70.809, 71.6, 71.6, 72.805, 72.805, 73.923, 76.176, 78.109, 78.65], 1e-3 )
    AssertListsAreAlmostEqual(pwlNonStrict['PWLY'],[0.0, 0.267, 0.333, 0.333, 0.667, 0.667, 0.8, 0.8, 0.833, 0.833, 0.867, 0.867, 0.933, 0.933, 1.0], 1e-3 )

    IllustrateSwitch = False
    if IllustrateSwitch:
        n=1000
        AlphaList = np.linspace(0.001,0.999,n)
        pwlS  = PiecewiseLinearDistribution(pwlStrict['PWLX'],pwlStrict['PWLY'])
        pwlNS = PiecewiseLinearDistribution(pwlNonStrict['PWLX'],pwlNonStrict['PWLY'])
        SampleDist = CompressedSampleNonStrict.plot(ShowPlot = False)
        pwlF  = PiecewiseLinearDistribution(SampleDist['SampleX'],SampleDist['SampleY'])
        xTVaR_S = np.asarray([pwlS.xTVaR(alpha) for alpha in AlphaList])
        xTVaR_NS = np.asarray([pwlNS.xTVaR(alpha) for alpha in AlphaList])
        xTVaR_F  = np.asarray([pwlF.xTVaR(alpha) for alpha in AlphaList])

        # plt.plot(AlphaList,xTVaR_S,  color = 'g', label='Strict')
        # plt.plot(AlphaList,xTVaR_NS, color = 'r', label='NonStrict')
        plt.plot(AlphaList,np.abs(xTVaR_S-xTVaR_F)/xTVaR_F,  color = 'g', label='Strict')
        plt.plot(AlphaList,np.abs(xTVaR_NS-xTVaR_F)/xTVaR_F, color = 'r', label='NonStrict')
        plt.plot([0,1],[Accuracy,Accuracy], color = 'k', label='Accuracy')
        # pwlIntermediate = PiecewiseLinearDistribution(pwlI['PWLX'],pwlI['PWLY'])
        # xTVaR_I = np.asarray([pwlIntermediate.xTVaR(alpha) for alpha in AlphaList])
        # plt.plot(AlphaList,np.abs(xTVaR_I-xTVaR_F)/xTVaR_F, color = 'y', label='Intermediate')
        plt.legend(loc='upper left', shadow=False)
        plt.show()

        # pwlStrict = CompressedSampleStrict.plot('g', ShowPlot = False)
        # pwlNonStrict = CompressedSampleNonStrict.plot('g', ShowPlot = False)
        #
        # plt.plot(pwlStrict['SampleX'],pwlStrict['SampleY'],color='black')
        # plt.plot(pwlStrict['PWLX'],pwlStrict['PWLY'], linewidth=2.0, linestyle = '-', marker = 'o', color = 'r')
        # plt.plot(pwlNonStrict['PWLX'],pwlNonStrict['PWLY'], linewidth=1.0, linestyle = '-', marker = 'o', color = 'b')


    # StrictAccuracies = []
    # for Accuracy in AccuracyList:
    #     CompressedSampleStrict    = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = True,  Accuracy = Accuracy)
    #     CompressedSampleNonStrict = PWLcompressor(Sample, RemoveNegativeJumps = True, MakePWLsmoother = True, CheckStrictAdmissibility = False, Accuracy = Accuracy)
    #     pwlStrict = CompressedSampleStrict.Result
    #     pwlNonStrict = CompressedSampleNonStrict.Result
    #     if len(pwlStrict['PWLX']) != len(pwlNonStrict['PWLX']):
    #         print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    #         print(Accuracy)
    #         print(pwlStrict)
    #         print(pwlNonStrict)
    #         StrictAccuracies.append(Accuracy)


    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # print(StrictAccuracies)




if __name__ == "__main__":
    RunAllTestCases()
