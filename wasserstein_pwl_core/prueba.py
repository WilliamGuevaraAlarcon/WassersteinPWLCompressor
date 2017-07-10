import numpy as np
from wasserstein_pwl_core.pwl_distribution import PiecewiseLinearDistribution, EmpiricalDistributionFromSample

Sample = [1, 1.6, 4.3, 4.6, 6, 7.1, 13, 13.4, 16, 18.8]



G = PiecewiseLinearDistribution([0.38, 7.82, 12.8, 13.6, 14.6, 20.2],
                                [0.0, 0.6, 0.6, 0.8, 0.8, 1.0])
F = EmpiricalDistributionFromSample(Sample)

epsilon = np.linspace(0.105, 0.114, 100)


x = 4.29999

result = [G.cdf(x-e)-e <= F.cdf(x) and F.cdf(x) <= G.cdf(x+e)+e for e in epsilon]

print(result)

print("hola")