__author__ = 'chuarph'

import numpy as np
import matplotlib.pyplot as plt

class PiecewiseLinearDistribution:
    """
    PiecewiseLinearDistribution is a class that represents piecewise linear
    probability distributions.
    === PARAMETERS ===
    This univariate distribution is characterized by two vectors Xvalues
    and Fvalues of same size which satisfy
      -infty < Xvalues(1) <= Xvalues(2) <= ... <= Xvalues(end) < infty
           0 = Fvalues(1) <= Fvalues(2) <= ... <= Fvalues(end) = 1
    === MATHEMATICAL DEFINITION ===
    The probability distribution is defined such that the cumulative
    distribution function (cdf) is the interpolation between the points
    defined by the parameters. I.e., for Xvalues(i) < x < Xvalues(i+1), the
    cdf F(x) is given by
    F(x) = Fvalues(i) + (x-Xvalues(i))/(Xvalues(i+1)-Xvalues(i))*(Fvalues(i+1)-Fvalues(i))
    The distribution can have atoms, which are represented by two equal
    values for Xvalues and different values for Fvalues. In this case, the
    cdf evaluates to the higher Fvalue
    === METHODS ===
    The class implements various methods that are connected to
    distruibutions, such as cdf, pdf, quantile, rng, moments. complete
    descriptions of the methods are given below.
    === EXAMPLES ===
    Y~Bernoulli(0.5)
    myDist = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])

    properties:
      - Xvalues in [xmin, xmax]
      - Fvalues   in [0, 1]
    are column vectors with the same size.

    Methods:
    - PiecewiseLinearDistribution(X,P)
      constructs an object of PiecewiseLinearDistribution class, with Xvalues
      specified by X and Fvalues by P. Valid Arguments are
      - X real-valued vector, and P [0,1]-valued vector with the same
        size
      - X and P are monotonely increasing vectors
      - P(end)=1 and P(1)<=0.01.
      If 0<P(1)<=0.01 the constructor will extend the values of the CDF,
      i.e.: Fvalues=[0;P]; and Xvalues=[X(1);X];

    - plot or plot(optionalSettings)
      plots the cdf of the current entity and if speciefied, using the
      optionalSettings.

    - cdf(x)
      evaluates P(X<=x) by linear interpolation assuming that the
      cumulative distribution function is right-continuous with left
      limits. That is
          cdf(x)=sup{p(i) in [0,1] | cdf(x(i))<= p(i)}.
      Note that if
      x(i) < xmin, cdf(x(i))= 0 and x(i) > xmax, cdf(x(i)) = 1

    - quantile(p)
      evaluates the generalized inverse of p(i). that is
          inf{x(i) in R | cdf(x(i)) >= p(i)}

    - rnd, rnd(n) or rnd(n,m)
      generates a random number, n-dimensional column vector or an n * m matrix
      using the inverse method

    - shift(b)
      for scalar b, returns an object of PiecewiseLinearDistribution describing the
      CDF of  X + b.

    - scale(a)
      for scalar a, -inf < a < inf, returns an object of PiecewiseLinearDistribution
      describing the CDF of aX.

    - moment(k)
      computes the k-th moment of X. Allowed k are non-negative
      integers (for all distributions) and non-negative integers
      (for positive distributions).

    - expected, variance, stdev, skewness and kurtosis
      computes an approximation of the expected value, variance,
      standard deviation, skewness and kurtosis of X

    - TVaR(q,tailLabel)
      gives the expected shortfall of X at a level q in (0,1) at the tail
      specified by tailLabel in {'upper', 'lower'} by default
      tailLabel = 'lower'.
      The expected shortfall is computed by distorting the distribution
      of X.

    - distort(divFun)
      generates an entity of PiecewiseLinearDistribution with a cdf*(.)=divFun(cdf(.))
      where the mappping divFun: [0,1]->[0,1] is also a PiecewiseLinearDistribution.

    - getSupport
      returns the Support of X

    - pdf(x)
      evaluates the probability density function (pdf) of X at the points
      specified by the vector x.
    """

    Xvalues = None
    Fvalues = None

    def __init__(self, Xin, Pin):
        """
        Returns an object of PiecewiseLinearDistribution modelling a cumulative
        distribution functions specified by the values in X and
        probabilities in P.
        The vectors X and P should be omnotonely increasing and have
        the same size.
        Additonally P takes values between 0 and 1

        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          Y.plot
        """

        X = np.asarray(Xin, dtype = np.float)
        P = np.asarray(Pin, dtype = np.float)

        if X.ndim > 1 or P.ndim > 1 :
            raise Exception('PiecewiseLinearDistribution ist only defined for vector arguments')

        tol=1e-10
        #minimal distance between two nodes
        FStartEndTolerance=1e-3
        #Just a small Tolerance, in order that all treaties are accepted
        #xAct accurateness 1e-7
        #augment P and X in order to ensure that 0,1 are elements of P
        #If they are redundandt we take rid of them next

        #if np.any(P < 0) or np.any(P > 1):
        #    raise Exception('Error on CDF values: Probabilities should be in [0,1]')

        if (P[0] > FStartEndTolerance) or (P[-1] < 1-FStartEndTolerance): #P[-1] = P(end)
            raise Exception('Start and end value for Fvalues should be close to 0 and 1')

        #enlarge/append vectors if P is not starting at 0
        if P[0] > 0:
            P = np.insert(P,0,0.0)
            X = np.insert(X,0,X[0])

        #enlarge/append vectors if P is not ending at 1
        if P[-1] < 1:
            P = np.append(P,1)
            X = np.append(X,X[-1])

        #check increasingness of P
        dP = np.diff(P)
        if np.any(dP < 0): #if np.any(dP < -1e-10*P[1:]): #
            raise Exception('Error on CDF values: CDF is not monotonely increasing')

        #check increasingness of X
        dX = np.diff(X)
        if np.any(dX < 0):
            raise Exception('X is not monotonely increasing')

        #We remove first the redundant nodes
        keepPoints = (dX > tol) | (dP > tol) #has length n-1, type boolean
        keepPoints = np.insert(keepPoints,0,True)
        X = X[keepPoints]
        P = P[keepPoints]

        #new diffs...
        dP = np.diff(P)
        dX = np.diff(X)

        #and check if the nodes are colinear by testing if
        #the angle between line segments is near 0 (determinant)
        #It will not remove nodes if dP or dX are very large or inf.
        eps = 1e-13
        keepPoints = np.abs(dX[:len(dX)-1]*dP[1:] - dX[1:]*dP[:len(dP)-1]) > eps*np.abs(dX[:len(dX)-1]*dP[1:]) #size n-2
        keepPoints = np.insert(keepPoints,0,True)
        keepPoints = np.append(keepPoints,True)
        X = X[keepPoints]
        P = P[keepPoints]

        # assign object properties
        self.Xvalues = X
        self.Fvalues = P


    def plot(self,plotarguments = '.-'):
        # plots the current object of PiecewiseLinearDistribution
        # linspec is optional and specifies the plot settings
        # Example: Y~Bernoulli(0.5)
        #   Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
        #   Y.plot('-rO')
        plt.plot(self.Xvalues,self.Fvalues,plotarguments)
        plt.show()


    def cdf(self,X):
        """
        returns the cumulative probability
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          Y.cdf([0:0.1:1]')
        """
        return LinearInterpolation(X, self.Xvalues, self.Fvalues, DiscontinuitiesBehaviour = 'RightContinuous')


    def quantile(self,P):
        """
        return the quantile function
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          Y.quantile([0:0.1:1]')
        """
        return LinearInterpolation(P, self.Fvalues, self.Xvalues, DiscontinuitiesBehaviour = 'LeftContinuous')

    def inverse(self):

        return PiecewiseLinearDistribution(self.Fvalues, self.quantile(self.Fvalues))


    def rnd(self, n = 1):
        """
        Generates pseudo random numbers of the pwl distribution, size of returned sample is n
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          mean(Y.rnd(100000))
        """
        return self.quantile(np.random.rand(n))


    def shift(self,b):
        """
        return an object of PiecewiseLinearDistribution for X+b
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          mean(Y.shift(-0.5).rnd(100000))
        """
        if not(isinstance(b,(float,int))):
            raise Exception('Error in shift argument: it should be a finite scalar')

        return PiecewiseLinearDistribution(self.Xvalues+b,self.Fvalues)


    def scale(self,a):
        """
        return an object of PiecewiseLinearDistribution for aX
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          mean(Y.scale(-0.5).rnd(100000))
        """
        if not(isinstance(a,(float,int))):
            raise Exception('Error in scaling argument: it should be a finite scalar')

        if a > 0:
            scaledObj=PiecewiseLinearDistribution(a*self.Xvalues,self.Fvalues)
        elif a < 0:
            scaledObj=PiecewiseLinearDistribution(a*self.Xvalues[::-1], 1-self.Fvalues[::-1])
        else: #scale=0 and scaledObj=trivial
            scaledObj=PiecewiseLinearDistribution([0,0],[0,1])

        return scaledObj


    def moment(self,k):
        """
        Calculates the k-th moment E[X^k]. Allowed k are non-negative
        integers (for all distributions) and non-negative reals
        (for positive distributions).
        The formula is deduced from
        E[X^k] = \int_{-inf}^{+inf} x^k dF(x) = \int_0^1 (F^{-1}(p)9^k dp
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          Y.moment(1/2)
        """
        if not(isinstance(k,(float,int))):
            raise Exception('Error in argument: it should be a float or int')
        if k < 0:
            raise Exception('Error in argument: k must be non-negative')
        if (np.round(k) != k) and (self.Xvalues[0] < 0):
            raise Exception('for non-integer moments, distribution must be positive')

        weights = np.diff(self.Fvalues)
        dXVals  = np.diff(self.Xvalues)

        dXnonZero = (dXVals != 0)
        dXequalToZero = -dXnonZero #true -> false
        PartialIntegrals = np.zeros_like(dXVals)
        PartialIntegrals[dXnonZero]  = 1/(k+1)*np.diff(self.Xvalues**(k+1))[dXnonZero]/dXVals[dXnonZero] #correct for dX>0
        PartialIntegrals[dXequalToZero] = self.Xvalues[dXequalToZero]**k  #now, also correct for dX==0
        kthMoment = np.sum(PartialIntegrals*weights) #sum up weighted partial integrals
        return float(kthMoment)


    def expected(self):
        """
        calculates the expected value E[X]
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          Y.expected
        """
        return self.moment(1)


    def variance(self):
        """
        calculates the variance E[(X-E[X])^2]
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          Y.variance
        """
        mu = self.expected()
        return self.moment(2) - mu*mu


    def stdev(self):
        """
        calculates the standard deviation sqrt(E[(X-E[X])^2])
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          Y.stdev
        """
        return np.sqrt(self.variance())


    def skewness(self):
        """
        Calculates the Skewness sk = E[(X-EX)^3]/(Var(X))^(3/2)
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          Y.skewness
        """
        mu = self.expected()
        M2 = self.moment(2)
        ThirdCentralMoment = self.moment(3) - 3*M2*mu + 2*mu**3
        return ThirdCentralMoment/(M2-mu**2)**1.5


    def kurtosis(self):
        """
        Calculates the kurtosis kappa = E[(X-EX)^4]/(Var(x))^2
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          Y.kurtosis
        """
        mu = self.expected()
        M2 = self.moment(2)
        M3 = self.moment(3)
        FourthCentralMoment = self.moment(4) - 4*M3*mu + 6*M2*mu**2 - 3*mu**4
        return FourthCentralMoment/(M2-mu**2)**2


    def xTVaR(self, q, tailLabel = 'lower'):
        # unittests are missing!!!
        return abs(self.expected() - self.TVaR(q, tailLabel = tailLabel)) #abs because the taillabel switch is equivalent to absolute value

    def TVaR(self, q, tailLabel = 'lower'):
        """
        Computes the expected shortfall at q-level. The optional
        parameter tailLabel can be'lower' or 'upper'. Its default
        value is 'lower'
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          Y.TVaR(0.001,'upper')
        """
        if not(isinstance(q,(float,int))):
            raise Exception('Error in argument: it should be a float or int')
        if (q <= 0) or (q >= 1):
            raise Exception('Error in argument: threshold must be >0 and <1')
        if not(tailLabel in ('lower','upper')):
            raise Exception('Error in argument: tailLabel must be upper or lower')

        if tailLabel == 'lower':
            H = PiecewiseLinearDistribution([0,q,1],[0,1,1])
        else:
            H = PiecewiseLinearDistribution([0,q,1],[0,0,1])
        return self.distort(H).expected()


    def distort(self, divFun):
        """
        generates an entity of PiecewiseLinearDistribution with a cdf*(.)=divFun(cdf(.))
        where the mappping divFun: [0,1]->[0,1] is also a PiecewiseLinearDistribution.
        divFun is also a PiecewiseLinearDistribution mapping F to F*
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          Y.shift(-0.5).scale(10).distort(Y).plot
        """

        if not(isinstance(divFun,PiecewiseLinearDistribution)):
            raise Exception('Distort is defined for PiecewiseLinearDistribution arguments only')

        divSupport=divFun.getSupport()
        if (divSupport[0] < 0) or (divSupport[1] > 1):
            raise Exception('Diversification function should map [0,1] -> [0,1]')

        # The current PiecewiseLinearDistribution is exact on the partition Px1 defined
        # by self.Xvalues. Also the distorted function is exact on the
        # partition Px2 defined by divFun.Xvalues. Note that In the
        # augmented partition Px1 U Px2 the distorted probability is not
        # exact and the error is proportional to
        # max{abs(x_k-x_k-1)|x_k in P1}+ max{abs(x_k-x_k-1)|x_k in P2}.
        # We want to add required interpolation points to self, to that
        # end let us define the sets:
        # X_H := divFun.Xvalues, P_H :=divFun.Fvalues,
        # X_F := self.Xvalues, P_F :=self.Fvalues
        # and let rFi(p), lFi(p), be the right respectively left continuous
        #   versions of the inverse of F. In the same way
        # rH(x), lH(x) denote the right and left-continuous versions of
        # H, and let us define the set
        # (A,B):={(x_i,y_i)|x_i ith-element of A and y_i i-th element of B}
        # for A and B having the same cardinality.Then
        # Nodes = (rFi(X_H),H(X_H)) U (lFi(X_H),H(X_H)) U (X_F,rH(F(X_F))) U (X_F,lH(F(X_F)))}


        extended_Xvalues = np.concatenate( (
            LinearInterpolation(divFun.Xvalues,self.Fvalues,self.Xvalues,'LeftContinuous'),
            LinearInterpolation(divFun.Xvalues,self.Fvalues,self.Xvalues,'RightContinuous'),
            self.Xvalues, self.Xvalues) )
        distorted_Fvalues = np.concatenate( (
            divFun.Fvalues, divFun.Fvalues,
            LinearInterpolation(self.Fvalues,divFun.Xvalues,divFun.Fvalues,'LeftContinuous'),
            LinearInterpolation(self.Fvalues,divFun.Xvalues,divFun.Fvalues,'RightContinuous')) )
        XandFvaluesMatrix = np.concatenate(([extended_Xvalues],[distorted_Fvalues]),axis=0).T
        Nodes=LexicalSortedMatrixWithTwoColumns(XandFvaluesMatrix) #several rows will be duplicate, but filtered by constructor

        for i in range(1,Nodes.shape[0]): #there seem to be few roundoff issues in the function LexicalSortedMatrixWithTwoColumns... correct them here
            Nodes[i,1] = max(Nodes[i,1],Nodes[i-1,1])

        return PiecewiseLinearDistribution(Nodes[:,0],Nodes[:,1])


    def getSupport(self):
        """
        Returns a vector with the support of X
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          Y.getSupport()
        """
        return np.asarray([self.Xvalues[0], self.Xvalues[-1]], dtype = np.float)


    def pdf(self,evalPoints):
        """
        Evaluates the probability distribution function of X
        in the points specified by the vector evalPts.
        Example: Y~Bernoulli(0.5)
          Y = PiecewiseLinearDistribution([0;0;1;1],[0;0.5;0.5;1])
          Y.pdf([-1,0.25,0.7,0.75,1,1.25]')
        """
        dFVals = np.diff(self.Fvalues)
        dXVals = np.diff(self.Xvalues)

        pdfVals = np.zeros_like(dXVals)
        dXnonZero = (dXVals != 0)
        dXequalToZero = -dXnonZero # = (dXVals == 0)   .true -> false
        pdfVals[dXnonZero] = dFVals[dXnonZero]/dXVals[dXnonZero]
        pdfVals[dXequalToZero] = -1 #np.infty #infinity

        pdfValsForInterpolation = DuplicateVectorItems(pdfVals) # [2,4,3] -> [2,2,4,4,3,3]
        pdfValsForInterpolation = np.concatenate([[0],pdfValsForInterpolation,[0]]) # [2,2,4,4,3,3] -> [0,2,2,4,4,3,3,0]
        XValsForInterpolation = DuplicateVectorItems(self.Xvalues)

        # interpolate for all points which have noninf density
        pdfPoints = np.asarray(LinearInterpolation(evalPoints, XValsForInterpolation, pdfValsForInterpolation, DiscontinuitiesBehaviour = 'RightContinuous' ))

        # set the values to inf where evalPoints is on atoms
        AtomLoc = np.append(dXequalToZero,False)
        Atoms = self.Xvalues[AtomLoc]
        evalPointsOnAtoms = np.in1d(evalPoints,Atoms)
        pdfPoints[evalPointsOnAtoms] = np.infty
        return pdfPoints


def RunUnitTests():

    #test LexicalSortedMatrixWithTwoColumns
    TestData =  np.array([(3, 2), (6, 2), (3, 6), (3, 4), (5, 3)])
    assert np.all(LexicalSortedMatrixWithTwoColumns(TestData) == np.asarray([[3, 2],[3, 4],[3, 6],[5, 3],[6, 2]]))

    #test helper function IsApproximatelyEqual
    assert IsApproximatelyEqual(1.4,1.41,0.01001)
    assert IsApproximatelyEqual(0.01,0,0.02)
    assert not(IsApproximatelyEqual(0.01,0,0.001))

    #test helper function norm
    assert IsApproximatelyEqual(norm([1,2]),2.2360679774997898,0.0001)

    #test LinearInterpolation (more tests are done through cdf and icdf tests...
    assert IsApproximatelyEqual(norm(LinearInterpolation(1, [0,1,1,2], [2,3,4,5], DiscontinuitiesBehaviour = 'RightContinuous')-4),0,0)
    assert IsApproximatelyEqual(norm(LinearInterpolation(1, [0,1,1,2], [2,3,4,5], DiscontinuitiesBehaviour = 'LeftContinuous')-3),0,0)

    #test EmpiricalDistributionFromSample
    sample = [3,7,4,9]
    EmpDist = EmpiricalDistributionFromSample(sample)
    assert IsApproximatelyEqual(norm(EmpDist.Xvalues - [3,3,4,4,7,7,9,9]),0,0)
    assert IsApproximatelyEqual(norm(EmpDist.Fvalues - [0,0.25,0.25,0.5,0.5,0.75,0.75,1]),0,0)

    #test constructor
    Xt = [1, 1.5,  2,  2,   2,   2,    4,    4,  4]
    Pt = [0, 0.2, 0.4, 0.4, 0.4, 0.6,  0.6, 0.8, 1]
    Xx = [1, 2,   2,    4,   4]
    Px = [0, 0.4,0.6,  0.6,  1]
    myPWL  = PiecewiseLinearDistribution(Xt,Pt)

    assert IsApproximatelyEqual(norm(myPWL.Xvalues-Xx),0,0)
    assert IsApproximatelyEqual(norm(myPWL.Fvalues-Px),0,0)

    myPWL.plot('-b*')

    #test cdf and quantile
    Xi = [0, 1, 1.5, 1.75,  2,  2,   2,   2, 3,   4,    4,  4, 10]
    Pi = [0, 0.2, 0.4, 0.4, 0.4, 0.5, 0.6,  0.6, 0.8, 1 ]
    assert IsApproximatelyEqual(norm(myPWL.cdf(Xi)-[0, 0, 0.2, 0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 1, 1, 1, 1]),0,1e-10)
    assert IsApproximatelyEqual(norm(myPWL.quantile(Pi)-[1, 1.5, 2, 2, 2, 2, 2, 2, 4, 4]),0,1e-10)

    # Xi=rand(100000,1)*(myPWL.Xvalues(end)-myPWL.Xvalues(1))*2+myPWL.Xvalues(1);
    # Pi=rand(100000,1);
    # plot(Xi,myPWL.cdf(Xi),'g+');
    # plot(myPWL.quantile(Pi),Pi,'b*');

    #test shift and scale
    assert IsApproximatelyEqual(norm(myPWL.shift(3).Xvalues-[4,5,5,7,7]),0,1e-10)
    assert IsApproximatelyEqual(norm(myPWL.scale(1.5).Xvalues-[1.5,3,3,6,6]),0,1e-10)
    assert IsApproximatelyEqual(norm(myPWL.scale(0).Xvalues-[0,0]),0,1e-10)
    assert IsApproximatelyEqual(norm(myPWL.scale(-1).Xvalues-[-4,-4,-2,-2,-1]),0,1e-10)

    #test sampling
    N = 1000000
    Sample = myPWL.rnd(N)
    assert IsApproximatelyEqual(np.mean(Sample),myPWL.expected(),5*myPWL.stdev()/np.sqrt(N)) #5 sigma -> http://de.wikipedia.org/wiki/Six_Sigma
    assert IsApproximatelyEqual(np.std(Sample),myPWL.stdev(),10*myPWL.stdev()/np.sqrt(N)) #some guess... 10*myPWL.stdev()/np.sqrt(N) = 0.011718930554164626

    #test distort
    H1x=[0, 0.3, 0.5, 1]
    H1y=[0, 0.6, 0.6 , 1]
    H1=PiecewiseLinearDistribution(H1x,H1y)
    D1x=[1, 1.75, 2, 2, 4, 4]
    D1y=[0, 0.6, 0.6, 0.68, 0.68, 1]
    D1=myPWL.distort(H1)
    # D1.plot()
    H2x=[0, 0.1, 0.1, 1]
    H2y=[0, 0.1, 0.55, 1]
    D2x=[1, 1.25, 1.25, 2, 2, 4, 4]
    D2y=[0, 0.1, 0.55, 0.7, 0.8, 0.8, 1]
    H2=PiecewiseLinearDistribution(H2x,H2y)
    D2=myPWL.distort(H2)

    assert IsApproximatelyEqual(norm(D1.Xvalues-D1x),0,1e-10)
    assert IsApproximatelyEqual(norm(D1.Fvalues-D1y),0,1e-10)
    assert IsApproximatelyEqual(norm(D2.Xvalues-D2x),0,1e-10)
    assert IsApproximatelyEqual(norm(D2.Fvalues-D2y),0,1e-10)

    #Test Moments (expected,variance, stdev, skewness, kurtosis)
    Xt=[1,   2,   2,    4,  4]
    Pt=[0, 0.4, 0.6,  0.6,  1]
    D = PiecewiseLinearDistribution(Xt,Pt)
    assert IsApproximatelyEqual(D.moment(0),     1.0,1e-7)
    assert IsApproximatelyEqual(D.moment(1),     2.6,1e-7)
    assert IsApproximatelyEqual(D.moment(2),   8.133,1e-3)
    assert IsApproximatelyEqual(D.moment(3),  28.700,1e-3)
    assert IsApproximatelyEqual(D.moment(4), 108.080,1e-3)
    assert IsApproximatelyEqual(D.moment(5), 420.200,1e-3)
    assert IsApproximatelyEqual(D.moment(1.5), 4.511,1e-3)
    assert IsApproximatelyEqual(D.moment(2.5),15.110,1e-3)
    assert IsApproximatelyEqual(D.moment(3.5),55.385,1e-3)

    assert IsApproximatelyEqual(D.expected() ,D.moment(1),1e-7)
    assert IsApproximatelyEqual(D.expected() , 2.6,1e-3)
    assert IsApproximatelyEqual(D.variance() , 1.373,1e-3)
    assert IsApproximatelyEqual(D.stdev()    , 1.172,1e-3)
    assert IsApproximatelyEqual(D.skewness() , 0.256,1e-3)
    assert IsApproximatelyEqual(D.kurtosis() , 1.270,1e-3)

    #test pdf
    pdf2Eval=[ -2, -1, 0,  1, 1, 1.5, 2, 3, 3, 3.5, 4, 5, 6]
    pdfExact=[0, 0, 0, 0.4, 0.4, 0.4, np.inf, 0, 0, 0, np.inf, 0, 0]
    assert np.all(myPWL.pdf(pdf2Eval)==pdfExact)

    ### Tests von Jessica
    #with atoms:
    Xval = [-1,1,1,2]
    Fval = [0,0.2,0.5,1.0]
    dist = PiecewiseLinearDistribution(Xval, Fval)
    assert dist.pdf(1) == np.inf

    ### test case for non-injective Fvalues (here: the only
    ### "critical" function is icdf!)
    xval = [-1,0,1,2]
    Fval = [0,0.2,0.2,1.0]
    dist = PiecewiseLinearDistribution(xval, Fval)

    # test pdf
    assert IsApproximatelyEqual(norm(dist.pdf([-1.1,-1,-0.5,0,0.5,1,1.5,2,2.1])-[0,0.2,0.2,0,0,0.8,0.8,0,0]),0,1e-10)

    #moments test:
    assert IsApproximatelyEqual(dist.moment(1), 1.1,0)    #, 'first moment'
    assert IsApproximatelyEqual(dist.expected(), 1.1,0)    #, 'mean'
    assert IsApproximatelyEqual(dist.moment(2), 5.8/3, 1e-13)    #, 'second moment'
    assert IsApproximatelyEqual(dist.variance(), 5.8/3-1.1**2, 1e-13)    #, 'variance'
    assert IsApproximatelyEqual(dist.stdev(), np.sqrt(5.8/3-1.1**2),1e-13)    #, 'standard deviation'

    #TVaR tests
    assert IsApproximatelyEqual(dist.TVaR(0.1,'lower'), -0.75,0)    #,'TVaR failed')
    assert IsApproximatelyEqual(dist.TVaR(0.6,'upper'), 1.75,0)    #,'TVaR failed')


def LinearInterpolation(x, xp, fp, DiscontinuitiesBehaviour = 'RightContinuous'):
    """
    Linear interpolation of a function defined through points x and function values y, such that f(xp[i])=fp[i].
    the function is extended on left/right with first/last y value.
    For points where the function jumps (i.e., equal xp values), the DiscontinuitiesBehaviour parameter determines the
    evaluation strategy (continuous from left/right respectively)
    returns a vector of f which is given by f(x), where f is defined through xp and yp
    """

    if not(DiscontinuitiesBehaviour in ['RightContinuous','LeftContinuous']):
        raise Exception('Error in DiscontinuitiesBehaviour argument: must be RightContinuous or LeftContinuous')

    if not(np.all(np.diff(xp) >= 0)):
        raise Exception('input xp must be non-decreasing')

    #use numpy's interpolation function. make use of the fact that np.interp (seems to?) be implemented in a right
    #continuous manner
    if DiscontinuitiesBehaviour == 'RightContinuous':
        fx = np.interp( x, xp, fp, left=fp[0], right=fp[-1])
    else:  #DiscontinuitiesBehaviour == 'LeftContinuous':
        x  = np.asarray(x,  dtype = np.float) #make sure one can apply '-' operator...
        xp = np.asarray(xp, dtype = np.float) #make sure one can apply '-' operator...
        fp = np.asarray(fp, dtype = np.float)
        fx = np.interp(-x, -xp[::-1], fp[::-1], left=fp[-1], right=fp[0])
    if isinstance(fx,(float,int)):
        fx = np.asarray([fx])
    return fx


def IsApproximatelyEqual(A, B, epsilon):
    """
    returns true if the the difference between A and B is smaller  or equal to epsilon in absolute value.
    I.e., returns true if |A-B| <= epsilon
    INPUT: scalar real values A, B. nonnegative scalar epsilon
    """

    #input checks
    # if len(A) > 1 | len(B) > 1 | len(epsilon) > 1:
    if (not(isinstance(A,(float,int)))) or (not(isinstance(B,(float,int)))) or (not(isinstance(epsilon,(float,int)))):
        raise Exception('support only scalar inputs')

    #logic
    if abs(A-B) <= epsilon:
        return True
    else:
        return False

def norm(x):
    return float(np.sqrt(np.sum(np.square(x)))) #sqrt(sum(x_i^2))

def LexicalSortedMatrixWithTwoColumns(data):
    """
    http://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.lexsort.html
    """
    ind=np.lexsort((data[:,1],data[:,0]))
    return data[ind]

def DuplicateVectorItems(x):
    """
    duplicate items in a numpy array
    example: [2,4,3] -> [2,2,4,4,3,3]
    """
    return np.concatenate( [[x],[x]] ).T.flatten() # [2,4,3] -> [2,2,4,4,3,3]

def EmpiricalDistributionFromSample(sample):
    """
    turns a sample into its empirical cdf.
    EXAMPLE:
      sample = [3,4,7,9]
      EmpiricalDistributionFromSample(sample) =
         PiecewiseLinearDistribution([3,3,4,4,7,7,9,9],[0,0.25,0.25,0.5,0.5,0.75,0.75,1])
    """
    sample  = np.asarray(sample,  dtype = np.float) #make sure its numpy array
    sample = np.sort(sample)
    Xval = DuplicateVectorItems(sample) # [3,4,7,9] -> [3,3,4,4,7,7,9,9]
    SampleSize = sample.size
    Fval = DuplicateVectorItems( np.linspace(0,1,num = (SampleSize+1)) ) #4 -> [0,0,0.25,0.25,0.5,0.5,0.75,0.75,1,1]
    Fval = Fval[1:(2*SampleSize + 1)] # [0,0,0.25,0.25,0.5,0.5,0.75,0.75,1,1] -> [0,0.25,0.25,0.5,0.5,0.75,0.75,1]
    return PiecewiseLinearDistribution(Xval,Fval)

