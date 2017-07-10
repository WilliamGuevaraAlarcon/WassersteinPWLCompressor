# some methods to determine the admissible approximation error to be used in the
# pwl approximation. The admissible error is determined by estimating the
# wasserstein distance between the true distribution and the sample distribution.
# Alternatively, the user may specify the absolute value of the admissible error.

import numpy as np


def admissible_approximation_error(sample, admissibility_mode, admissibility_factor):
    assert(admissibility_mode in ['relative', 'absolute'])
    assert(isinstance(admissibility_factor, (float, int)))
    if admissibility_mode == 'relative':
        return admissibility_factor * ExpectedWassersteinError(sample)
    return admissibility_factor


def ExpectedWassersteinError(sample):
    # calculates sqrt(2/pi)*\int_{-inf}^{+inf} sqrt( F_n(x)*(1-F_n(x)) ) dx
    # for F_n empirical distribution of sample
    sample.sort()
    samplesize = len(sample)
    empdist = np.linspace(1/samplesize,1-1/samplesize,samplesize-1)
    dx = np.diff(sample)
    integrand = np.sqrt( empdist*(1-empdist) )
    integral = np.sum(integrand*dx)
    integral *= np.sqrt(2/np.pi)
    integral /= np.sqrt(samplesize)
    return integral


# def GenerateTestCase(sample):
#     import scipy.integrate as integrate
#     def integrand(x):
#         ssize = len(sample)
#         emp = sum(sample < x)/ssize
#         fun = np.sqrt(2/3.141592653589793/ssize*emp*(1-emp))
#         return fun
#     result = integrate.quad(integrand, min(sample), max(sample), limit = 300)[0]
#     print(result)


def test_admissible_approximation_error():
    assert(abs(admissible_approximation_error(None, 'absolute', 0.1) - 0.1) < 1e-6)

def test_ExpectedWassersteinError_1():
    sample = np.asarray([1,12,23,24,35])
    test_value = 5.237751431165672
    assert(abs(ExpectedWassersteinError(sample) - test_value) < 1e-7)
    assert(abs(admissible_approximation_error(sample, 'relative', 0.1) - 0.1*test_value) < 1e-6)

def test_ExpectedWassersteinError_2():
    sample = np.asarray([1,12,23,24,35,35,35])
    test_value = 4.450283650217102
    assert(abs(ExpectedWassersteinError(sample) - test_value) < 1e-7)
    assert(abs(admissible_approximation_error(sample, 'relative', 0.2) - 0.2*test_value) < 1e-6)

def test_ExpectedWassersteinError_3():
    sample = np.asarray([1])
    test_value = 0
    assert(abs(ExpectedWassersteinError(sample) - test_value) < 1e-7)
    assert(abs(admissible_approximation_error(sample, 'relative', 0.1) - 0.1*test_value) < 1e-6)

def test_ExpectedWassersteinError_4():
    sample = np.asarray([12,1,23,1,23,12,23,12,3,2354,345,234,12,12,31,23,213,12.23,22.22])
    test_value = 105.69764468283762
    assert(abs(ExpectedWassersteinError(sample) - test_value) < 1e-4)
    assert(abs(admissible_approximation_error(sample, 'relative', 10) - 10*test_value) < 1e-3)


if __name__ == "__main__":
    test_admissible_approximation_error()
    test_ExpectedWassersteinError_1()
    test_ExpectedWassersteinError_2()
    test_ExpectedWassersteinError_3()
    test_ExpectedWassersteinError_4()
