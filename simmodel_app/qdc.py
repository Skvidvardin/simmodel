import numpy as np
import pandas as pd
import datetime


def qdc_algo_2(arrival_serving_time, knot_locations, servers_activity_num):
    """

    :param arrival_serving_time:
    :param knot_locations:
    :param servers_activity_num:
    :return:
    """
    arrival_serving_time = np.insert(arrival_serving_time, 0, np.arange(len(arrival_serving_time)), axis=1)
    knot_locations.append(np.inf)
    servers_activity_num.append(1)

    b = np.full(np.max(servers_activity_num), np.inf)
    b[:servers_activity_num[0]] = 0

    p = np.zeros(len(arrival_serving_time)).astype(int)  # vector of indexes of workers who served the model
    d = np.zeros(len(arrival_serving_time))

    l = 0
    p[0] = 0

    for i in range(len(arrival_serving_time)):
        if (b > knot_locations[l + 1]).all() or arrival_serving_time[i, 1] > knot_locations[l + 1]:
            if knot_locations[l + 1] - knot_locations[l] > 0:
                for k in range(knot_locations[l] + 1, knot_locations[l + 1]):
                    b[k] = knot_locations[l + 1]
            if knot_locations[l + 1] - knot_locations[l] < 0:
                for k in range(knot_locations[l + 1] + 1, knot_locations[l]):
                    b[k] = np.inf
            l += 1

        p[i] = np.argmin(b)
        b[p[i]] = max(arrival_serving_time[i, 1], b[p[i]]) + arrival_serving_time[i, 2]
        d[i] = b[p[i]]

        if knot_locations[l] == 0:
            i -= 1

    arrival_serving_time = np.insert(arrival_serving_time, 3, np.array([d, p]), axis=1)
    arrival_serving_time = arrival_serving_time[arrival_serving_time[:, 0].argsort()]

    return arrival_serving_time


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
        k.append(1)  # 0 indicating that server is open, 1 opposite

    kk = len(servers_activity_indicators)  # K ← length(x)

    b = np.zeros(kk)  # vector of timestamps when server will be free
    c = np.zeros(kk)  # vector of timestamps when server will be free at current step

    p = np.zeros(len(arrival_serving_time)).astype(int)  # vector of indexes of workers who served the model
    d = np.zeros(len(arrival_serving_time))  # vector of timestamps when model were served

    for i in range(len(arrival_serving_time)):
        for k in range(kk):
            c[k] = next_c(max(arrival_serving_time[i, 1], b[k]), knot_locations[k], servers_activity_indicators[k])

        p[i] = np.argmin(b)
        b[p[i]] = c[p[i]] + arrival_serving_time[i, 2]
        d[i] = b[p[i]]

    arrival_serving_time = np.insert(arrival_serving_time, 3, np.array([d, p]), axis=1)
    arrival_serving_time = arrival_serving_time[arrival_serving_time[:, 0].argsort()]

    return arrival_serving_time


def sm_main(simulationsNum, nWorkers, modelsIncome, initialQueueSize, initialInprogressSize,
               avgWorkDaysPerModel, sdWorkDaysPerModel,
               startDate, endDate):

    inputData = pd.DataFrame(columns=['arrival_time', 'model_input_type'])
    resultData = pd.DataFrame(columns=['arrival_time', 'service_time', 'done_time', 'sim'])

    if not startDate.weekday() < 5:
        while startDate.weekday() != 1:
            startDate = startDate + datetime.timedelta(days=1)

    if not endDate.weekday() < 5:
        while endDate.weekday() != 4:
            endDate = endDate - datetime.timedelta(days=1)

    dates = pd.bdate_range(start=startDate, end=endDate, normalize=True, closed='left').to_pydatetime().tolist()
    dates_d = pd.bdate_range(start=startDate, end=endDate + datetime.timedelta(days=365),
                             normalize=True, closed='left').to_pydatetime().tolist()

    dates_straight_mapper = dict(zip(dates, list(range(len(dates)))))
    dates_inverse_mapper = dict(zip(list(range(len(dates_d))), dates_d))
    dates_unique_months = pd.DataFrame([[date.month, date.year] for date in dates],
                                       columns=['month', 'year']).drop_duplicates()
    dates_unique_months = dates_unique_months.sort_values(['year', 'month'], ascending=True)

    if initialInprogressSize > 0:

        if initialInprogressSize > nWorkers:
            raise Exception(f'Initial in progress volume ({initialInprogressSize}) must be less than' +
                            f' number of workers ({nWorkers})')

        inputData = inputData.append(pd.DataFrame([[0, 'initialInprogress'] for _ in range(initialInprogressSize)],
                                                  columns=['arrival_time', 'model_input_type']))

    if initialQueueSize > 0:
        inputData = inputData.append(pd.DataFrame([[0, 'initialQueue'] for _ in range(initialQueueSize)],
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

        knot_locations = [0]
        servers_activity_num = [nWorkers, nWorkers]

        outputData = \
            qdc_algo_2(inputData[['arrival_time', 'service_time']].to_numpy(dtype=float),
                       knot_locations, servers_activity_num)

        outputData = pd.DataFrame(outputData[:, 1:4], columns=['arrival_time', 'service_time', 'done_time'])
        outputData['sim'] = sim

        resultData = resultData.append(outputData)

    resultData['service_start'] = resultData['done_time'] - resultData['service_time']
    resultData['queue_time_range'] = resultData['service_start'] - resultData['arrival_time']
    resultData['done_time_range'] = resultData['done_time'] - resultData['arrival_time']

    for col in ['arrival_time', 'done_time', 'service_start']:

        resultData[col + '_int'] = (resultData[col] // 1)
        resultData[col + '_float'] = resultData[col] % 1

        resultData[col + '_int'] = resultData[col + '_int'].map(dates_inverse_mapper)
        resultData[col + '_float'] = pd.to_timedelta(9 + 9 * resultData[col + '_float'], 'h')
        resultData[col + '_realtime'] = resultData[col + '_int'] + resultData[col + '_float']
        del resultData[col + '_int'], resultData[col + '_float']

    dates_unique_months['day'] = 1
    dates_unique_months['timestamps'] = pd.to_datetime(dates_unique_months) + \
                                        pd.DateOffset(months=1) - \
                                        pd.DateOffset(seconds=1)

    # OUTPUT FORMATION
    resultData['arrival_time_year'] = resultData['arrival_time_realtime'].dt.year
    resultData['arrival_time_month'] = resultData['arrival_time_realtime'].dt.month

    resultData['service_start_year'] = resultData['service_start_realtime'].dt.year
    resultData['service_start_month'] = resultData['service_start_realtime'].dt.month

    resultData['done_time_year'] = resultData['done_time_realtime'].dt.year
    resultData['done_time_month'] = resultData['done_time_realtime'].dt.month

    values = dict()

    # calc incomeNum
    values['incomeNum'] = \
        resultData.groupby(['arrival_time_year', 'arrival_time_month', 'sim'], as_index=False) \
            .count()[['arrival_time_year', 'arrival_time_month', 'sim', 'arrival_time']] \
            .rename(columns={'arrival_time': 'incomeNum'})
    values['incomeNum'].loc[(values['incomeNum']['arrival_time_year'] == startDate.year) &
                            (values['incomeNum']['arrival_time_month'] == startDate.month), 'incomeNum'] -= \
        (initialQueueSize + initialInprogressSize)
    values['incomeNum']['date'] = (values['incomeNum']['arrival_time_month']).astype(str) + '-' + \
                                  (values['incomeNum']['arrival_time_year']).astype(str)
    del values['incomeNum']['arrival_time_month'], values['incomeNum']['arrival_time_year']

    # calc doneNum
    values['doneNum'] = \
        resultData.groupby(['done_time_year', 'done_time_month', 'sim'], as_index=False) \
            .count()[['done_time_year', 'done_time_month', 'sim', 'done_time']] \
            .rename(columns={'done_time': 'doneNum'})
    values['doneNum']['date'] = (values['doneNum']['done_time_month']).astype(str) + '-' + \
                                (values['doneNum']['done_time_year']).astype(str)
    del values['doneNum']['done_time_month'], values['doneNum']['done_time_year']

    # calc queueNum, inProgressNum, avgServingTime, avgWaitingTime, avgTime2Done
    for date in dates_unique_months['timestamps']:
        if date == dates_unique_months['timestamps'][0]:
            values['queueNum'] = resultData.loc[(resultData['service_start_realtime'] > date) &
                                                (resultData['arrival_time_realtime'] < date), :] \
                .groupby(['sim'], as_index=False).count()[['sim', 'service_start']] \
                .rename(columns={'service_start': 'queueNum'})
            values['queueNum']['date'] = str(date.month)+'-'+str(date.year)

            values['inProgressNum'] = resultData.loc[(resultData['done_time_realtime'] > date) &
                                                     (resultData['service_start_realtime'] < date), :] \
                .groupby(['sim'], as_index=False).count()[['sim', 'done_time']] \
                .rename(columns={'done_time': 'inProgressNum'})
            values['inProgressNum']['date'] = str(date.month)+'-'+str(date.year)

            values['avgTimes'] = resultData.loc[(resultData['done_time_year'] == date.year) &
                                                      (resultData['done_time_month'] == date.month), :] \
                .groupby(['sim'], as_index=False) \
                .mean()[['sim', 'service_time', 'queue_time_range', 'done_time_range']] \
                .rename(columns={'service_time': 'avgServingTime',
                                 'queue_time_range': 'avgWaitingTime',
                                 'done_time_range': 'avgTime2Done'})
            values['avgTimes']['date'] = str(date.month)+'-'+str(date.year)

        else:
            value = resultData.loc[(resultData['service_start_realtime'] > date) &
                                   (resultData['arrival_time_realtime'] < date), :] \
                .groupby(['sim'], as_index=False).count()[['sim', 'service_start']] \
                .rename(columns={'service_start': 'queueNum'})
            value['date'] = str(date.month)+'-'+str(date.year)
            values['queueNum'] = values['queueNum'].append(value)

            value = resultData.loc[(resultData['done_time_realtime'] > date) &
                                   (resultData['service_start_realtime'] < date), :] \
                .groupby(['sim'], as_index=False).count()[['sim', 'done_time']] \
                .rename(columns={'done_time': 'inProgressNum'})
            value['date'] = str(date.month)+'-'+str(date.year)
            values['inProgressNum'] = values['inProgressNum'].append(value)

            value = resultData.loc[(resultData['done_time_year'] == date.year) &
                                   (resultData['done_time_month'] == date.month), :] \
                .groupby(['sim'], as_index=False) \
                .mean()[['sim', 'service_time', 'queue_time_range', 'done_time_range']] \
                .rename(columns={'service_time': 'avgServingTime',
                                 'queue_time_range': 'avgWaitingTime',
                                 'done_time_range': 'avgTime2Done'})
            value['date'] = str(date.month)+'-'+str(date.year)
            values['avgTimes'] = values['avgTimes'].append(value)

    result = None
    for key in values.keys():
        if result is None:
            result = values[key]
        else:
            result = result.merge(values[key], 'left', on=['sim', 'date'])

    return result


# def main_advanced():
#     return -1


# main_basic(simulationsNum=100,
#            nWorkers=50,
#            modelsIncome=100,
#            initialQueueSize=200,
#            initialInprogressSize=50,
#            avgWorkDaysPerModel=14,
#            sdWorkDaysPerModel=5,
#            startDate=datetime.date(2017, 1, 1),
#            endDate=datetime.date(2018, 1, 1))
