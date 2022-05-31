import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt


class Stockprocess:
    def __init__(self, nPaths: int, nSteps: int):
        self.nPaths = nPaths
        self.nSteps = nSteps

    @staticmethod
    def GBMprocess(initialValues, mu, sigma):

        GBM = ql.GeometricBrownianMotionProcess(initialValues, mu, sigma)

        return GBM

    def GeneratePaths(self, process):

        # generating random seed
        # URG vs MTUR : seed = 0 제외하고 동일한 것으로 보임 --> 근데 MTUR은 sequenceGenerator에서 에러발생
        generator = ql.UniformRandomGenerator(0)

        # dimension, Random Number Generator(RNG)를 받음 --> tuple로 난수 리턴해줌
        sequenceGenerator = ql.UniformRandomSequenceGenerator(self.nSteps, generator)

        # 정규분포 따르는 난수로 변환
        gaussianSequenceGenerator = ql.GaussianRandomSequenceGenerator(sequenceGenerator)

        # 결과 저장용
        paths = np.zeros(shape=(self.nPaths, self.nSteps + 1))

        # process, 만기, 구간, 난수 종류, brownian bridge(T/F)
        pathGenerator = ql.GaussianPathGenerator(process,
                                                 3,
                                                 self.nSteps,
                                                 gaussianSequenceGenerator,
                                                 False
                                                 )

        for i in range(self.nPaths):

           path = pathGenerator.next().value()
           paths[i, :] = np.array([path[j] for j in range(self.nSteps + 1)])

        return paths


if __name__ == "__main__":
    ## 만기는 float도 가능, mue/sigma는 1년 기준 . 따라서 만기를 고정시키려면 mue, sigma를 다르게 해야함
    nPaths = 50
    nSteps = 3*365
    timeGrid = np.linspace(0.0, nSteps, nSteps + 1)

    initialValue = 1
    mue = 0.03
    sigma = 00

    stkpr = Stockprocess(nPaths, nSteps)

    gbm = stkpr.GBMprocess(initialValue, mue, sigma)

    process = stkpr.GeneratePaths(gbm)


    for i in range(process.shape[0]):
        path = process[i, :]
        plt.plot(timeGrid, path)

    plt.show()


# 아마 365 베이스인듯...
# e^(0.05) : 1.0512710963760241
# 252 :      1.0512658824402437
# 365 :      1.0512674964674609
# 360 :      1.0512674464734508
# 250 :      1.051265840734422

