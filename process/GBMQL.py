import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt


def GeneratePaths(process, maturity, nPaths, nSteps):

    generator = ql.UniformRandomGenerator()
    print(generator.next().value())
    sequenceGenerator = ql.UniformRandomSequenceGenerator(nSteps, generator)
    print(sequenceGenerator.dimension())
    gaussianSequenceGenerator = ql.GaussianRandomSequenceGenerator(sequenceGenerator)

    paths = np.zeros(shape=(nPaths, nSteps + 1))

    pathGenerator = ql.GaussianPathGenerator(process, maturity, nSteps, gaussianSequenceGenerator, False)

    for i in range(nPaths):

       path = pathGenerator.next().value()
       paths[i, :] = np.array([path[j] for j in range(nSteps + 1)])

    return paths


tradeDate = ql.Date(23, ql.November, 2018)
ql.Settings_instance().evaluationDate = tradeDate
dayCounter = ql.Actual360()
calendar = ql.UnitedStates()
settlementDate = calendar.advance(tradeDate, 2, ql.Days)

maturity = 3.0
nPaths = 50
nSteps = int(maturity * 365)
timeGrid = np.linspace(0.0, maturity, nSteps + 1)

# reversionSpeed = 0.05
# rateVolatility = 0.0099255
# r = ql.QuoteHandle(ql.SimpleQuote(0.01))
# curve = ql.RelinkableYieldTermStructureHandle(ql.FlatForward(settlementDate, r, dayCounter))
# HW1F = ql.HullWhiteProcess(curve, reversionSpeed, rateVolatility)
# hw1f_paths = GeneratePaths(HW1F, maturity, nPaths, nSteps)

initialValue = 1000
mue = 0.01
sigma = 0.0099255
GBM = ql.GeometricBrownianMotionProcess(initialValue, mue, sigma)
gbm_paths = GeneratePaths(GBM, maturity, nPaths, nSteps)

print(gbm_paths)

# for i in range(gbm_paths.shape[0]):
#     path = gbm_paths[i, :]
#     plt.plot(timeGrid, path)
#
# plt.show()






