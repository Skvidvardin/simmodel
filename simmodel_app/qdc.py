import numpy as np
import pandas as pd
import datetime


def next_c(t, k_knot_locations, k_servers_activity_indicators):
    try:
        l = np.min(np.argwhere(k_knot_locations > t))  # Find l such that x_l < t ≤ x_(l+1)
    except ValueError:
        l = len(k_knot_locations) - 1

    if k_servers_activity_indicators[l] == 0:
        return k_knot_locations[l]
    else:
        return t


def qdc_algo_3(arrival_serving_time, knot_locations, servers_activity_indicators):
    """

    :param arrival_serving_time:
    :param knot_locations:
    :param servers_activity_indicators:
    :return:
    """
    arrival_serving_time = np.insert(arrival_serving_time, 0, np.arange(len(arrival_serving_time)), axis=1)
    # Sort (a,s) in terms of a(ascending)
    # arrival_serving_time = arrival_serving_time[arrival_serving_time[:, 1].argsort()]

    for k in knot_locations:  # x_k,L_(k+1) ← oo
        k.append(np.inf)

    for k in servers_activity_indicators:  # y_k,L_(k+2) ← 1
        k.append(0)

    K = len(servers_activity_indicators)  # K ← length(x)

    b = np.zeros(K)  # vector of timestamps when server will be free
    c = np.zeros(K)  # vector of timestamps when server will be free at current step

    p = np.zeros(len(arrival_serving_time)).astype(int)  # vector of indexes of workers who served the model
    d = np.zeros(len(arrival_serving_time))  # vector of timestamps when model were served

    for i in range(len(arrival_serving_time)):
        for k in range(K):
            c[k] = next_c(max(arrival_serving_time[i, 1], b[k]), knot_locations[k], servers_activity_indicators[k])

        p[i] = np.argmin(b)
        b[p[i]] = c[p[i]] + arrival_serving_time[i, 2]
        d[i] = b[p[i]]

    arrival_serving_time = np.insert(arrival_serving_time, 3, np.array([d, p]), axis=1)
    arrival_serving_time = arrival_serving_time[arrival_serving_time[:, 0].argsort()]

    return arrival_serving_time


def main_basic(simulationsNum, nWorkers, modelsIncome, initialQueueSize, initialInprogressSize,
               avgWorkDaysPerModel, sdWorkDaysPerModel,
               startDate, endDate):

    inputData = pd.DataFrame(columns=['arrival_time', 'model_input_type'])

    if not startDate.weekday() < 5:
        while startDate.weekday() != 1:
            startDate = startDate + datetime.timedelta(days=1)

    if not endDate.weekday() < 5:
        while endDate.weekday() != 4:
            endDate = endDate - datetime.timedelta(days=1)

    dates = pd.bdate_range(start=startDate, end=endDate, normalize=True, closed='left').to_pydatetime().tolist()
    dates_straight_mapper = dict(zip(dates, list(range(len(dates)))))
    dates_inverse_mapper = dict(zip(list(range(len(dates))), dates))
    dates_unique_months = pd.DataFrame([[date.month, date.year] for date in dates],
                                       columns=['month', 'year']).drop_duplicates()
    dates_unique_months = dates_unique_months.sort_values(['year', 'month'], ascending=True)

    if initialInprogressSize > 0:

        if initialInprogressSize > nWorkers:
            return Exception(f'Initial in progress volume ({initialInprogressSize}) must be less than' +
                             f' number of workers ({nWorkers})')

        inputData = inputData.append(pd.DataFrame([[0, 'initialInprogress'] for _ in range(initialInprogressSize)],
                                                  columns=['arrival_time', 'model_input_type']))

    if initialQueueSize > 0:

        inputData = inputData.append(pd.DataFrame([[0, 'initialQueue'] for _ in range(initialInprogressSize)],
                                                  columns=['arrival_time', 'model_input_type']))

    if modelsIncome > 0:

        for index, row in dates_unique_months.iterrows():
            dates_selected = list(filter(lambda x: x.year == row.year and x.month == row.month, dates))
            modelsInput = pd.DataFrame(columns=['arrival_time', 'model_input_type'])
            modelsInput['arrival_time'] = list(map(dates_straight_mapper.get, dates_selected))
            modelsInput = \
                modelsInput.append([modelsInput] * (modelsIncome // len(dates_selected) - 1), ignore_index=True)
            randInts = np.random.choice(range(len(dates_selected)), modelsIncome % len(dates_selected), replace=False)
            for i in randInts:
                modelsInput = modelsInput.append(modelsInput.loc[i, :], ignore_index=True)

            modelsInput['model_input_type'] = 'income'

            inputData = inputData.append(modelsInput.sort_values('arrival_time', ignore_index=True), ignore_index=True)

    for sim in range(simulationsNum):
        inputData['service_time'] = np.random.gamma((avgWorkDaysPerModel ** 2) / (sdWorkDaysPerModel ** 2),
                                                    (sdWorkDaysPerModel ** 2) / (avgWorkDaysPerModel),
                                                    len(inputData))

        inputData['input_mult'] = 1
        inputData.loc[inputData['model_input_type'] == 'initialInprogress', 'input_mult'] = \
            np.random.uniform(0, 1, initialInprogressSize)

        inputData['service_time'] = inputData['service_time'] * inputData['input_mult']

        knot_locations = [[] for _ in range(nWorkers)]
        servers_activity_indicators = [[1] for _ in range(nWorkers)]

        outputData = \
            qdc_algo_3(inputData[['arrival_time', 'service_time']].to_numpy(dtype=float),
                       knot_locations, servers_activity_indicators)

        print(sim)

    print(1)


    return -1


def main_advanced():
    return -1


# arrival_serving_time = np.array([[2, 2], [1, 2], [3, 3]])
# knot_locations = [[1, 2], [0, 2]]
# servers_activity_indicators = [[0, 1, 1], [0, 0, 1]]
#
# result = qdc_main(arrival_serving_time, knot_locations, servers_activity_indicators)


main_basic(simulationsNum=1000,
           nWorkers=100,
           modelsIncome=100,
           initialQueueSize=200,
           initialInprogressSize=100,
           avgWorkDaysPerModel=14,
           sdWorkDaysPerModel=5,
           startDate=datetime.date(2017, 1, 1),
           endDate=datetime.date(2018, 1, 1))
