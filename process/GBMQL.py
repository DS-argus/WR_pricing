import QuantLib as ql
import numpy as np
import pandas as pd


def GBMprocess(maturity:int,
               start_date:ql.Date,
               calendar:ql.Calendar,
               underlying: list,
               rf: float,
               sigma: dict,
               corr: list = None,
               spot: list = None) -> pd.DataFrame:

    # number of processes need to be generated
    nProcesses = len(underlying)

    if corr is None:
        corr = np.identity(nProcesses).tolist()

    # if there is no designated spot value, set as 1.0
    if spot is None:
        spot = [1.0 for _ in range(nProcesses)]

    # list containing each process
    process = []
    for i in range(nProcesses):
        process.append(ql.GeometricBrownianMotionProcess(spot[i], rf, sigma[underlying[i]]))

    # process = [ql.GeometricBrownianMotionProcess(spot[i], rf, sigma[underlying[i]]) for i in range(nProcesses)]

    # generate array of correlated 1-D stochastic processes
    processArray = ql.StochasticProcessArray(process, corr)

    # make schedule to get number of steps to be created
    schedule = ql.Schedule(start_date,
                           start_date + ql.Period(int(maturity), ql.Years),
                           ql.Period(1, ql.Days),
                           calendar,  # els.get_calendar() 로 수정
                           ql.Following,
                           ql.Following,
                           ql.DateGeneration.Forward,
                           False)

    schedule = [ql.Date.to_date(x) for x in list(schedule)]

    timeGrid = ql.TimeGrid(int(maturity), len(schedule) - 1)
    times = [t for t in timeGrid]
    nSteps = (len(times) - 1) * nProcesses

    # create random number
    generator = ql.UniformRandomGenerator()
    sequenceGenerator = ql.UniformRandomSequenceGenerator(nSteps, generator)
    gaussianSequenceGenerator = ql.GaussianRandomSequenceGenerator(sequenceGenerator)
    multiPathGenerator = ql.GaussianMultiPathGenerator(processArray, times, gaussianSequenceGenerator)

    multiPath = multiPathGenerator.next().value()

    df_path = pd.DataFrame(columns=underlying, index=schedule)

    for i in range(nProcesses):
        df_path.iloc[:, i] = [j for j in multiPath[i]]

    return df_path


if __name__ == "__main__":

    maturity = 3
    start_date = ql.Date.todaysDate()
    calendar = ql.JointCalendar(ql.SouthKorea(), ql.HongKong(), ql.UnitedStates())

    underlying = ['KOSPI200', 'HSCEI', 'S&P500']
    rf = 0.03
    sigma = {'KOSPI200': 0.2,
             'HSCEI': 0.22,
             'S&P500': 0.25}
    corr = [[1, 0.73, 0.17],
            [0.73, 1, 0.28],
            [0.17, 0.28, 1]]

    print(GBMprocess(maturity, start_date, calendar, underlying, rf, sigma, corr))
