__author__ = 'u002311'

from pwl_compressor_core.sample_characteristics import SampleCharacteristics
# from SolutionStackClass import SolutionStack
# from ProblemStackClass import ProblemStack
# from ProblemStackClass import ProblemInterval
import numpy as np
# import matplotlib.pyplot as plt

class ConvergeRateCalculator:

    SampleStats         = None
    TreeCollection      = []

    def __init__(self, Sample, NrOfIterations = 10):

        # initialize properties of object
        self.SampleStats = SampleCharacteristics(Sample) #initialize sample statistics

        SampleSize = self.SampleStats.SampleSize
        CurrentSolutionList = [self.SampleStats.FindBestSolutionLine(0,SampleSize-1)]
        IterationsList = list(range(NrOfIterations))

        # while True:
        for i in IterationsList:
            SummaryIntervals = list()
            NextSolutionList = list()
            for SolutionLine in CurrentSolutionList:
                if SolutionLine.isBisectable():
                    #add stats
                    SummaryIntervals.append(SolutionLine.MaximumDifference)
                    #bisect
                    [LeftPart, RightPart] = SolutionLine.Bisect()
                    for Part in [LeftPart, RightPart]:
                        Sol = self.SampleStats.FindBestSolutionLine(Part.SampleSet_Start,Part.SampleSet_End)
                        NextSolutionList.append(Sol)

            if i < 4:
                #print the basis z for iteration i
                Basis = [Sol.SampleSet_Start/SampleSize for Sol in CurrentSolutionList]
                Basis.append(1.0)
                # print('Iteration '+str(i)+' : '+str(Basis))
                print('Iteration '+str(i)+' : '+ ', '.join(["%.3f" % y for y in Basis]))

            CurrentSolutionList=NextSolutionList

            self.TreeCollection.append(self.ObtainErrorStatistics(SummaryIntervals, i))


    def ObtainErrorStatistics(self,Lista, Iteration):
        Summary = dict()
        Summary['Iteration'] = Iteration
        Summary['Maximum']   = max(Lista)
        Summary['Mean']      = np.mean(Lista)
        Summary['Length']    = len(Lista)
        return  Summary

    def print(self):
        for row in self.TreeCollection:
            print(str(row)+',')
        print('=============================================')
        # print([str(row) for row in self.TreeCollection])
