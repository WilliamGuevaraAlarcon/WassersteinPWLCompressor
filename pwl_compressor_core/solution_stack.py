__author__ = 'chuarph'

# import numpy as np
from .solution_line import SolutionLine

class SolutionStack:

    Stack = None

    def __init__(self):
        self.Stack = [] #initialize in the constructor, as mutable initialization in the class definition is dangerous

    def StackLength(self):
        return len(self.Stack)

    def isNotEmpty(self):
        if self.Stack:
            return True
        else:
            return False

    def append(self,ObjectToPush):
        if not(isinstance(ObjectToPush,SolutionLine)):
            raise Exception('pushed object must be of type SolutionLine')
        self.Stack.append(ObjectToPush)
        if self.isNotEmpty(): #if not empty
            self.Stack = sorted(self.Stack, key = lambda y : y.SampleSet_Start) #sort solutions according to y start point

    def GiveSolutionsIndicesWithNegativeIncrements(self):
        """
        return all indices of solutions i where
         a) solutions i and i+1 are adjacent ( (self.Stack[i].SampleSet_End + 1) == self.Stack[i+1].SampleSet_Start )
         b) the solution i+1 starts with a x-value lower than the end x-value of solution i (
        """
        Epsilon = 1e-9
        IndicesWithNegativeIncrements = []
        if self.isNotEmpty():
            StackLength = self.StackLength()
            for i in range(StackLength - 1):
                AbsoluteEpsilon = abs(self.Stack[i+1].Segment_Line_Start_X)*Epsilon
                if ( ( (self.Stack[i].SampleSet_End + 1)    == self.Stack[i+1].SampleSet_Start) & #end-y and start-y are equal, i.e. segments attach
                     (  self.Stack[i+1].Segment_Line_Start_X - self.Stack[i].Segment_Line_End_X < -AbsoluteEpsilon    ) ): #there is a negative jump in the x-coordinate, which is not allowed for cdf's
                    IndicesWithNegativeIncrements.append(i)
        return IndicesWithNegativeIncrements



    def CorrectOrPopNegativeIncrements(self):
        IndicesWithNegativeIncrements = self.GiveSolutionsIndicesWithNegativeIncrements()
        SolutionsToPop = set() # TODO: replace by set of references (i.e. pointers) instead of set of indices
        NewProblems = []
        for i in IndicesWithNegativeIncrements:
            # go through all solutions Sol = self.Stack[i] where  self.Stack[i].Segment_Line_End_X > self.Stack[i+1].Segment_Line_Start_X (which is not acceptable!)
            Sol = self.Stack[i]
            SolAbove = self.Stack[i+1]
            BoundFromLeft = max(Sol.Calculate_EndXFromDelta(Sol.Delta_LowerBound),SolAbove.Segment_Line_Start_X)
            BoundFromRight = min(SolAbove.Calculate_StartXFromDelta(SolAbove.Delta_LowerBound),Sol.Segment_Line_End_X)
            if BoundFromLeft <= BoundFromRight:
                #if BoundFromLeft <= BoundFromRight then the Delta parameters of Sol and SolAbove can be adjusted such
                # no negative increment exists any more
                ConnectingPoint = (BoundFromLeft + BoundFromRight)/2.0 #midpoint between left and right bound
                # set Delta parameter of Sol and SolAbove such that they connect at ConnectingPoint
                Sol.SetDelta(Sol.Calculate_DeltaFromEndX(ConnectingPoint))
                SolAbove.SetDelta(SolAbove.Calculate_DeltaFromStartX(ConnectingPoint))
                print('Connect Solutions to remove negative increment at y = '+str(Sol.Segment_Line_End_Y))
            else: #check which solution is Bisectiable. if both, take bigger
                SolIsBisectable = Sol.isBisectable()
                SolAboveIsBisectable = SolAbove.isBisectable()
                # go through the four possible states of SolIsBisectable x SolAboveIsBisectable = { (True,True),  (True,False), (False,True), (False,False) }
                if SolIsBisectable and SolAboveIsBisectable:
                    if Sol.Size() > SolAbove.Size(): #pop the bigger solution of both if both are bisectable
                        SolutionsToPop.add(i)
                    else:
                        SolutionsToPop.add(i+1)
                elif SolIsBisectable and not SolAboveIsBisectable: # if just one is bisectable, pop that one
                    SolutionsToPop.add(i)
                elif not SolIsBisectable and SolAboveIsBisectable: # if just one is bisectable, pop that one
                    SolutionsToPop.add(i+1)
                else: #this should never happen...
                    raise Exception('Adjacent solutions are incompatible, but neither is bisectable.')

        SolutionsToPop = sorted(SolutionsToPop, reverse = True)#sort from largest to smallest such that there is no indexshift when one object is removed
        for i in SolutionsToPop:
            # if self.Stack[i].isBisectable(): #take only those which can actually be bisected, which is not true if SampleSet_End == SampleSet_Start
            PoppedSolution = self.Stack.pop(i)
            NewProblems.extend(PoppedSolution.Bisect())
            print(('Cannot connect Solutions. Pop solution at y = '+str(PoppedSolution.Segment_Line_Start_Y)
                   +' to y = '+str(PoppedSolution.Segment_Line_End_Y)
                   +'. Bisect at '+str( (PoppedSolution.BestBisectionPoint+1)/PoppedSolution.SampleSize)
            ))

        return NewProblems


    def SmoothenSolutions(self):
        assert len(self.GiveSolutionsIndicesWithNegativeIncrements()) == 0
        StackLength = self.StackLength()

        PrecisionFactor = 1e-7 #TODO: unify all comparisons of coordinates

        for i in range(StackLength):
            Sol = self.Stack[i]
            CanBeConnectedBelow = False
            CanBeConnectedAbove = False
            IsAlreadyConnected  = False
            if i > 0: #if Sol is not the first solution, check compatibility with solution below
                SolBelow = self.Stack[i-1]
                EndPointBelow = SolBelow.Segment_Line_End_X
                PointEpsilon = PrecisionFactor*max(1,abs(EndPointBelow),abs(Sol.Segment_Line_Start_X))
                assert Sol.Segment_Line_Start_X - EndPointBelow  >= -PointEpsilon  #check that there is no negative increment
                if Sol.Segment_Line_Start_X - EndPointBelow  <= PointEpsilon:
                    IsAlreadyConnected = True
                else:
                    ConnectingDeltaFromBelow = Sol.Calculate_DeltaFromStartX(EndPointBelow)
                    CanBeConnectedBelow = (ConnectingDeltaFromBelow > Sol.Delta_LowerBound) and (ConnectingDeltaFromBelow < Sol.Delta_UpperBound)
            if i < StackLength - 1: #if Sol is not the last solution, check compatibility with solution above
                SolAbove = self.Stack[i+1]
                EndPointAbove = SolAbove.Segment_Line_Start_X
                PointEpsilon = PrecisionFactor*max(1,abs(EndPointAbove),abs(Sol.Segment_Line_End_X))
                assert EndPointAbove - Sol.Segment_Line_End_X >= -PointEpsilon #check that there is no negative increment
                if EndPointAbove - Sol.Segment_Line_End_X  <= PointEpsilon:
                    IsAlreadyConnected = True
                else:
                    SmoothingIntervalLeft = max(SolAbove.Calculate_StartXFromDelta(SolAbove.Delta_UpperBound),Sol.Segment_Line_End_X)
                    SmoothingIntervalRight = min(Sol.Calculate_EndXFromDelta(Sol.Delta_UpperBound),EndPointAbove)
                    CanBeConnectedAbove = (SmoothingIntervalLeft <= SmoothingIntervalRight)
                    DeltaImpliedFromAbove = Sol.Calculate_DeltaFromEndX( (SmoothingIntervalLeft+SmoothingIntervalRight)/2 )

            #test 3 cases
            if (not IsAlreadyConnected) and (CanBeConnectedBelow or CanBeConnectedAbove):
                if CanBeConnectedBelow and CanBeConnectedAbove:
                    NewDelta = min(ConnectingDeltaFromBelow,DeltaImpliedFromAbove)
                if CanBeConnectedBelow and not CanBeConnectedAbove:
                    NewDelta = ConnectingDeltaFromBelow
                if not CanBeConnectedBelow and CanBeConnectedAbove:
                    NewDelta = DeltaImpliedFromAbove

                print(('Make Solution flatter at y = '+'{:.4f}'.format(Sol.Segment_Line_Start_Y)
                       +' to y = '+'{:.4f}'.format(Sol.Segment_Line_End_Y)
                       +'. Old Delta = '+'{:.4f}'.format( Sol.Delta_Selected)
                       +'. New Delta = '+'{:.4f}'.format( NewDelta)
                       +'. Diff = '+'{:.4f}'.format( NewDelta-Sol.Delta_Selected)
                ))
                Sol.SetDelta(NewDelta)

    def CheckStrictAdmissibility(self, SampleStats, Accuracy):
        # Check for each solution in the solution stack whether the solution is strictly admissible. If not bisect.
        SolutionsToPop = set()
        NewProblems = []
        StackLength = self.StackLength()
        for i in range(StackLength):
            # go through all solutions Sol = self.Stack[i] and check whether its strictly admissible
            if not self.Stack[i].IsStrictyAdmissible(SampleStats, Accuracy):
                SolutionsToPop.add(i)

        SolutionsToPop = sorted(SolutionsToPop, reverse = True)#sort from largest to smallest such that there is no indexshift when one object is removed
        for i in SolutionsToPop:
            # if self.Stack[i].isBisectable(): #take only those which can actually be bisected, which is not true if SampleSet_End == SampleSet_Start
            PoppedSolution = self.Stack.pop(i)
            NewProblems.extend(PoppedSolution.Bisect())
            print(('ADMISSIBILITYPROBLEM: Solution not fully admissible. Pop solution at y = '+'{:.4f}'.format(PoppedSolution.Segment_Line_Start_Y)
                   +' to y = '+'{:.7f}'.format(PoppedSolution.Segment_Line_End_Y)
                   +'. Bisect at '+'{:.7f}'.format( (PoppedSolution.BestBisectionPoint+1)/PoppedSolution.SampleSize)
            ))

        if not NewProblems: print('Checked admissibilty: all segments are fine')
        return NewProblems


    def CheckCompletenessOfSolutionStack(self):
        """
        return true if the whole interval [0,1] is covered with adjacent solutions. false otherwise
        """
        StackLength = self.StackLength()
        if (self.Stack[0].Segment_Line_Start_Y != 0) or (self.Stack[StackLength-1].Segment_Line_End_Y != 1):
            return False
        for i in range(StackLength-1):
            if self.Stack[i].Segment_Line_End_Y != self.Stack[i+1].Segment_Line_Start_Y:
                return False
        return True
