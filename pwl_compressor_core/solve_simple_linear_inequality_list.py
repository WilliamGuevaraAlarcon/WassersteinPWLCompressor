__author__ = 'chuarph'

import numpy as np

def SolveSimpleLinearInequailityList(A, B, C, Epsilon):
    """
    for numpy arrays A, B, C of same size, solve the inequality problems
      A[i] <= x*B[i] <= C[i] for all i
    I.e., find the maximal and minimal x which satisfies all the inequalities above.
    Return values are SolutionExists (Bool), LowerBound (float), UpperBound (float)
    Meaning of return values are analogous to the return values of SolveSimpleConstrainedInequality

    Example:
    SolveSimpleLinearInequailityList(np.asarray([1,2]),np.asarray([2,2]),np.asarray([4,3]))) = (True, 1.0, 1.5)
    SolveSimpleLinearInequailityList(np.asarray([1,2,2]),np.asarray([-2,-2,0]),np.asarray([4,3,1]))) = (False, None, None)
    """

    assert A.ndim == B.ndim == C.ndim
    assert A.size > 1 #zero-dimension arrays can't be indexed....
    assert A.size == B.size == C.size

    #check first the rows with B==0
    B_EqualToZero = (B == 0)
    if np.any(B_EqualToZero) and ( np.any(A[B_EqualToZero] > Epsilon) or np.any(C[B_EqualToZero] < -Epsilon) ):
        return False, None, None
    #now we know that the inequalities with B==0 are satisfied
    B_BiggerThanZero = (B > 0)
    B_SmallerThanZero = (B < 0)

    if B.min() >= 0: #case where all B are positive
        LowerBound = (A[B_BiggerThanZero]/B[B_BiggerThanZero]).max()
        UpperBound = (C[B_BiggerThanZero]/B[B_BiggerThanZero]).min()
    elif B.max() <= 0: #case where all B are negative
        LowerBound = (C[B_SmallerThanZero]/B[B_SmallerThanZero]).max()
        UpperBound = (A[B_SmallerThanZero]/B[B_SmallerThanZero]).min()
    else: #general case: both B_BiggerThanZero and B_SmallerThanZero have some True's inside
        LowerBound = max( (A[B_BiggerThanZero]/B[B_BiggerThanZero]).max() , (C[ B_SmallerThanZero ]/B[ B_SmallerThanZero ]).max() )
        UpperBound = min( (C[B_BiggerThanZero]/B[B_BiggerThanZero]).min() , (A[ B_SmallerThanZero ]/B[ B_SmallerThanZero ]).min() )

    if (LowerBound <= UpperBound) and (UpperBound >= 0.0):
        LowerBound = max(0.0,LowerBound)
        return True, LowerBound, UpperBound
    else:
        return False, None, None



# runtestcases_SolveInequailityList()





