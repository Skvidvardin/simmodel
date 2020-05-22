import numpy as np
import pandas as pd
import datetime


def qdc_algo_2(arrival_serving_time, knot_locations, active_servers_count):
    """

    :param arrival_serving_time:
    :param knot_locations:
    :param servers_activity_num:
    :return:
    """
    knot_locations.append(np.inf)
    active_servers_count.append(0)

    n_servers = np.max(active_servers_count)

    queue_times = np.full(n_servers, np.inf)  # vector of workers availability time (changes each moment)
    queue_times[:active_servers_count[0]] = 0

    n = len(arrival_serving_time)

    server_output = np.full(n, -1).astype(int)  # vector of indexes of workers who served the model
    output = np.full(n, np.inf)  # endings of models serving (timestamps)

    next_time = knot_locations[0]

    current_size = active_servers_count[0]
    next_size = active_servers_count[1]
    iter = 0

    for i in range(n):
        if (queue_times > next_time).all() or arrival_serving_time[i, 0] >= next_time:
            diff_size = next_size - current_size
            if diff_size > 0:
                for k in range(current_size, next_size):
                    queue_times[k] = next_time
            if diff_size < 0:
                for k in range(next_size, current_size):
                    queue_times[k] = np.inf

            current_size = next_size
            iter += 1
            next_size = active_servers_count[iter + 1]
            next_time = knot_locations[iter]

        queue = np.argmin(queue_times)
        queue_times[queue] = max(arrival_serving_time[i, 0], queue_times[queue]) + arrival_serving_time[i, 1]
        output[i] = queue_times[queue]
        server_output[i] = queue + 1

    arrival_serving_time = np.insert(arrival_serving_time, 2, np.array([output, server_output]), axis=1)

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


def qdc_algo_3(arrival_serving_time, knot_locations, active_servers_count):
    """

    :param arrival_serving_time:
    :param knot_locations:
    :param servers_activity_indicators:
    :return:
    """
    for k in knot_locations.keys():  # x_k,L_(k+1) ← oo
        knot_locations[k].append(np.inf)

    n_servers = dict()
    queue_times = dict()
    next_time = dict()
    current_size = dict()
    next_size = dict()
    iteration = dict()
    for k in active_servers_count.keys():
        active_servers_count[k].append(0)
        n_servers[k] = max(active_servers_count[k])
        queue_times[k] = np.full(n_servers[k], np.inf)  # vector of timestamps when server will be free
        queue_times[k][:active_servers_count[k][0]] = 0

        next_time[k] = knot_locations[k][0]
        current_size[k] = active_servers_count[k][0]
        next_size[k] = active_servers_count[k][1]

        iteration[k] = 0

    n = len(arrival_serving_time)

    server_output = np.full(n, 'None').astype(object)  # vector of indexes of workers who served the model
    output = np.full(n, np.inf)  # endings of models serving (timestamps)

    for i in range(n):

        queue_opt_index = dict()
        queue_opt_value = dict()
        for k in n_servers.keys():

            if (queue_times[k] > next_time[k]).all() or arrival_serving_time.loc[i, 'arrival_time'] >= next_time[k]:
                diff_size = next_size[k] - current_size[k]
                if diff_size > 0:
                    for j in range(current_size[k], next_size[k]):
                        queue_times[k][j] = next_time[k]
                if diff_size < 0:
                    for j in range(next_size[k], current_size[k]):
                        queue_times[k][j] = np.inf

                current_size[k] = next_size[k]
                iteration[k] += 1
                next_size[k] = active_servers_count[k][iteration[k] + 1]
                next_time[k] = knot_locations[k][iteration[k]]

            queue_opt_index[k] = np.argmin(queue_times[k])
            st = arrival_serving_time.loc[i, 'service_time_' + k]
            queue_opt_value[k] = max(arrival_serving_time.loc[i, 'arrival_time'], queue_times[k][queue_opt_index[k]]) + st

        opt_k = min(queue_opt_value, key=queue_opt_value.get)
        queue_times[opt_k][queue_opt_index[opt_k]] = queue_opt_value[opt_k]

        output[i] = queue_times[opt_k][queue_opt_index[opt_k]]
        server_output[i] = opt_k

    arrival_serving_time['done_time'] = output
    arrival_serving_time['serving_worker_type'] = server_output

    return arrival_serving_time


def base_postprocessing(resultData, dates_inverse_mapper, dates_unique_months, initialQueueSize, initialInprogressSize,
                        startDate):

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
    values['incomeNum']['date'] = (values['incomeNum']['arrival_time_month']).astype(int).astype(str) + '-' + \
                                  (values['incomeNum']['arrival_time_year']).astype(int).astype(str)
    del values['incomeNum']['arrival_time_month'], values['incomeNum']['arrival_time_year']

    # calc doneNum
    values['doneNum'] = \
        resultData.groupby(['done_time_year', 'done_time_month', 'sim'], as_index=False) \
            .count()[['done_time_year', 'done_time_month', 'sim', 'done_time']] \
            .rename(columns={'done_time': 'doneNum'})
    values['doneNum']['date'] = (values['doneNum']['done_time_month']).astype(int).astype(str) + '-' + \
                                (values['doneNum']['done_time_year']).astype(int).astype(str)
    del values['doneNum']['done_time_month'], values['doneNum']['done_time_year']

    # calc queueNum, inProgressNum, avgServingTime, avgWaitingTime, avgTime2Done
    for date in dates_unique_months['timestamps']:
        if date == dates_unique_months['timestamps'][0]:
            values['queueNum'] = resultData.loc[(resultData['service_start_realtime'] > date) &
                                                (resultData['arrival_time_realtime'] < date), :] \
                .groupby(['sim'], as_index=False).count()[['sim', 'service_start']] \
                .rename(columns={'service_start': 'queueNum'})
            values['queueNum']['date'] = str(date.month) + '-' + str(date.year)

            values['inProgressNum'] = resultData.loc[(resultData['done_time_realtime'] > date) &
                                                     (resultData['service_start_realtime'] < date), :] \
                .groupby(['sim'], as_index=False).count()[['sim', 'done_time']] \
                .rename(columns={'done_time': 'inProgressNum'})
            values['inProgressNum']['date'] = str(date.month) + '-' + str(date.year)

            resultData['queue_time_range'] = pd.to_numeric(resultData['queue_time_range'])
            resultData['done_time_range'] = pd.to_numeric(resultData['done_time_range'])
            resultData['service_time'] = pd.to_numeric(resultData['service_time'])
            values['avgTimes'] = resultData.loc[(resultData['done_time_year'] == date.year) &
                                                (resultData['done_time_month'] == date.month), :] \
                [['sim', 'service_time', 'queue_time_range', 'done_time_range']] \
                .groupby(['sim'], as_index=False)\
                .agg({'service_time': 'mean', 'queue_time_range': 'mean', 'done_time_range': 'mean'}) \
                .rename(columns={'service_time': 'avgServingTime',
                                 'queue_time_range': 'avgWaitingTime',
                                 'done_time_range': 'avgTime2Done'})
            values['avgTimes']['date'] = str(date.month) + '-' + str(date.year)

        else:
            value = resultData.loc[(resultData['service_start_realtime'] > date) &
                                   (resultData['arrival_time_realtime'] < date), :] \
                .groupby(['sim'], as_index=False).count()[['sim', 'service_start']] \
                .rename(columns={'service_start': 'queueNum'})
            value['date'] = str(date.month) + '-' + str(date.year)
            values['queueNum'] = values['queueNum'].append(value)

            value = resultData.loc[(resultData['done_time_realtime'] > date) &
                                   (resultData['service_start_realtime'] < date), :] \
                .groupby(['sim'], as_index=False).count()[['sim', 'done_time']] \
                .rename(columns={'done_time': 'inProgressNum'})
            value['date'] = str(date.month) + '-' + str(date.year)
            values['inProgressNum'] = values['inProgressNum'].append(value)

            value = resultData.loc[(resultData['done_time_year'] == date.year) &
                                   (resultData['done_time_month'] == date.month), :] \
                .groupby(['sim'], as_index=False) \
                .mean()[['sim', 'service_time', 'queue_time_range', 'done_time_range']] \
                .rename(columns={'service_time': 'avgServingTime',
                                 'queue_time_range': 'avgWaitingTime',
                                 'done_time_range': 'avgTime2Done'})
            value['date'] = str(date.month) + '-' + str(date.year)
            values['avgTimes'] = values['avgTimes'].append(value)

    result = None
    for key in values.keys():
        if result is None:
            result = values[key]
        else:
            result = result.merge(values[key], 'outer', on=['sim', 'date'])

    return result


def sm_main(simulationsNum, nWorkers, nWorkersDF, modelsIncome, modelsIncomeDF, initialQueueSize, initialInprogressSize,
            avgWorkDaysPerModel, sdWorkDaysPerModel,
            startDate, endDate):
    """

    :param simulationsNum:
    :param nWorkers:
    :param nWorkersDF:
    :param modelsIncome:
    :param modelsIncomeDF:
    :param initialQueueSize:
    :param initialInprogressSize:
    :param avgWorkDaysPerModel:
    :param sdWorkDaysPerModel:
    :param startDate:
    :param endDate:
    :return:
    """
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

    if modelsIncome > 0 and modelsIncomeDF is None:

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

    elif modelsIncomeDF is not None:

        modelsIncomeDF['arrival_time'] = pd.to_datetime(modelsIncomeDF['date'], format='%d/%m/%Y')
        modelsIncomeDF['arrival_time'] = modelsIncomeDF['arrival_time'].map(dates_straight_mapper)

        modelsIncomeDF = modelsIncomeDF.loc[modelsIncomeDF.index.repeat(modelsIncomeDF['volume'])]
        modelsIncomeDF['model_input_type'] = 'income'
        del modelsIncomeDF['volume'], modelsIncomeDF['date']
        inputData = inputData.append(modelsIncomeDF.sort_values('arrival_time', ignore_index=True), ignore_index=True)

    if nWorkersDF is not None:

        nWorkersDF['date'] = pd.to_datetime(nWorkersDF['date'], format='%d/%m/%Y')
        nWorkersDF['date'] = nWorkersDF['date'].map(dates_straight_mapper)

        nWorkersDF = pd.concat([pd.DataFrame({'date': [0], 'delta': [nWorkers]}), nWorkersDF]).reset_index(drop=True)

        max_range_workers_delta = np.inf
        for i in range(1, len(nWorkersDF)):
            max_range_workers_delta_check = int(nWorkersDF.loc[i, 'date']) - int(nWorkersDF.loc[i - 1, 'date'])
            if max_range_workers_delta_check < max_range_workers_delta:
                max_range_workers_delta = max_range_workers_delta_check
            nWorkersDF.loc[i, 'delta'] = nWorkersDF.loc[i - 1, 'delta'] + int(nWorkersDF.loc[i, 'delta'])
        del max_range_workers_delta_check

        knot_locations = nWorkersDF['date'].tolist()[1:]
        servers_activity_num = nWorkersDF['delta'].tolist()

    for sim in range(simulationsNum):
        inputData['service_time'] = np.random.gamma((avgWorkDaysPerModel ** 2) / (sdWorkDaysPerModel ** 2),
                                                    (sdWorkDaysPerModel ** 2) / (avgWorkDaysPerModel),
                                                    len(inputData))

        inputData['input_mult'] = 1
        inputData.loc[inputData['model_input_type'] == 'initialInprogress', 'input_mult'] = \
            np.random.uniform(0, 1, initialInprogressSize)

        inputData['service_time'] = inputData['service_time'] * inputData['input_mult']

        if nWorkersDF is None:
            knot_locations = [0]
            servers_activity_num = [nWorkers, nWorkers]
        elif nWorkersDF is not None:
            inputData['service_time_max'] = max_range_workers_delta
            inputData['service_time'] = inputData[['service_time', 'service_time_max']].min(axis=1)

        outputData = qdc_algo_2(inputData[['arrival_time', 'service_time']].to_numpy(dtype=float),
                                knot_locations, servers_activity_num)

        outputData = pd.DataFrame(outputData[:, 0:3], columns=['arrival_time', 'service_time', 'done_time'])
        outputData['sim'] = sim

        resultData = resultData.append(outputData)

    result = base_postprocessing(resultData, dates_inverse_mapper, dates_unique_months,
                                 initialQueueSize, initialInprogressSize, startDate)

    return result


def sm_advanced_main(startDate, endDate, simulationsNum,
                     avgModelMartix, stdModelMartix,
                     workersNumDF, modelsIncomeDF,
                     initialQueueInprogressMatrix):

    inputData = pd.DataFrame(columns=['arrival_time', 'model_input_type', 'model_type'])
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

    initialQueueInprogressMatrix = initialQueueInprogressMatrix.set_index('ModelType')

    # IN PROGRESS MODELS
    initialQueueInprogressMatrix['InProgress'] = initialQueueInprogressMatrix['InProgress'].astype(int)
    if (initialQueueInprogressMatrix['InProgress'] > 0).any():

        inputDataInProgress = pd.DataFrame(columns=['arrival_time', 'model_input_type', 'model_type'])

        for i in initialQueueInprogressMatrix.index:
            for _ in range(initialQueueInprogressMatrix.loc[i, 'InProgress']):

                inputDataInProgress = inputDataInProgress\
                    .append(pd.DataFrame([[0, 'initialInProgress', i]],
                                         columns=['arrival_time', 'model_input_type', 'model_type']),
                            ignore_index=True)

        inputDataInProgress = inputDataInProgress.sample(frac=1).reset_index(drop=True)
        inputData = inputData.append(inputDataInProgress, ignore_index=True)
        del inputDataInProgress

    # IN QUEUE MODELS
    initialQueueInprogressMatrix['InQueue'] = initialQueueInprogressMatrix['InQueue'].astype(int)
    if (initialQueueInprogressMatrix['InQueue'] > 0).any():

        inputDataInQueue = pd.DataFrame(columns=['arrival_time', 'model_input_type', 'model_type'])

        for i in initialQueueInprogressMatrix.index:
            for _ in range(initialQueueInprogressMatrix.loc[i, 'InQueue']):
                inputDataInQueue = inputDataInQueue \
                    .append(pd.DataFrame([[0, 'initialInQueue', i]],
                                         columns=['arrival_time', 'model_input_type', 'model_type']),
                            ignore_index=True)

        inputDataInQueue = inputDataInQueue.sample(frac=1).reset_index(drop=True)
        inputData = inputData.append(inputDataInQueue, ignore_index=True)
        del inputDataInQueue

    # MODELS INCOME
    if modelsIncomeDF is not None:

        inputDataModels = pd.DataFrame(columns=['arrival_time', 'model_input_type', 'model_type'])

        modelsIncomeDF = modelsIncomeDF.rename(columns={'date': 'arrival_time'})
        modelsIncomeDF['arrival_time'] = pd.to_datetime(modelsIncomeDF['arrival_time'], format='%d/%m/%Y')
        modelsIncomeDF['arrival_time'] = modelsIncomeDF['arrival_time'].map(dates_straight_mapper)

        for c in modelsIncomeDF.columns:
            if c != 'arrival_time':
                _modelsIncomeDF = modelsIncomeDF[['arrival_time', c]]\
                    .loc[modelsIncomeDF[['arrival_time', c]].index.repeat(modelsIncomeDF[c])]
                _modelsIncomeDF['model_input_type'] = 'Income'
                _modelsIncomeDF['model_type'] = c
                _modelsIncomeDF = _modelsIncomeDF[['arrival_time', 'model_input_type', 'model_type']]
                inputDataModels = inputDataModels.append(_modelsIncomeDF, ignore_index=True)

        inputDataModels = inputDataModels.sample(frac=1).sort_values('arrival_time', ignore_index=True)
        inputData = inputData.append(inputDataModels, ignore_index=True)
        del _modelsIncomeDF, inputDataModels

    # WORKERS PROCESSING
    workersNumDF['date'] = pd.to_datetime(workersNumDF['date'], format='%d/%m/%Y')
    workersNumDF['date'] = workersNumDF['date'].map(dates_straight_mapper)
    workersNumDF = workersNumDF.sort_values('date')

    if 0 not in workersNumDF['date']:
        workersNumDF = pd.concat([pd.DataFrame([0 for _ in range(len(workersNumDF.columns))],
                                               columns=workersNumDF.columns),
                                  workersNumDF]).reset_index(drop=True)

    knot_locations_by_type = {}
    servers_activity_num_by_type = {}

    for c in workersNumDF.columns:
        if c != 'date':
            knot_locations_by_type[c] = workersNumDF[['date', c]].drop_duplicates(c)['date'].tolist()[1:]
            servers_activity_num_by_type[c] = workersNumDF[['date', c]].drop_duplicates(c)[c].tolist()

    max_range_workers_delta = {}
    for k in knot_locations_by_type.keys():
        arr = knot_locations_by_type[k].copy()
        if 0 not in arr:
            arr.append(0)
        arr = sorted(arr)
        diff = np.inf
        for i in range(len(arr)):
            if arr[i] != arr[-1]:
                if arr[i+1] - arr[i] < diff:
                    diff = arr[i+1] - arr[i]
        max_range_workers_delta[k] = diff
    del arr, diff

    # SIMMULATIONS
    modelTypes = [c for c in avgModelMartix.columns if c != 'WorkerType']
    for c in modelTypes:
        avgModelMartix[c] = avgModelMartix[c].astype(float)
        stdModelMartix[c] = stdModelMartix[c].astype(float)

    for sim in range(simulationsNum):

        for wt in avgModelMartix['WorkerType'].unique():
            for mt in modelTypes:
                inputData.loc[inputData['model_type'] == mt, 'service_time_' + wt] = \
                    np.random.gamma((avgModelMartix.loc[avgModelMartix['WorkerType'] == wt, mt].values[0] ** 2) /
                                    (stdModelMartix.loc[stdModelMartix['WorkerType'] == wt, mt].values[0] ** 2),
                                    (stdModelMartix.loc[stdModelMartix['WorkerType'] == wt, mt].values[0] ** 2) /
                                    (avgModelMartix.loc[avgModelMartix['WorkerType'] == wt, mt].values[0]),
                                    len(inputData.loc[inputData['model_type'] == mt]))

            inputData['max_service_time_' + wt] = max_range_workers_delta[wt]
            inputData['service_time_' + wt] = inputData[['service_time_' + wt, 'max_service_time_' + wt]].min(axis=1)
            del inputData['max_service_time_' + wt]

        inputData['input_mult'] = 1
        inputData.loc[inputData['model_input_type'] == 'initialInProgress', 'input_mult'] = \
            np.random.uniform(0, 1, len(inputData.loc[inputData['model_input_type'] == 'initialInProgress']))

        for wt in avgModelMartix['WorkerType'].unique():
            inputData['service_time_' + wt] = inputData['service_time_' + wt] * inputData['input_mult']

        _inputData = inputData.drop(columns=['input_mult', 'model_input_type', 'model_type'])

        outputData = qdc_algo_3(_inputData, knot_locations_by_type, servers_activity_num_by_type)
        outputData['sim'] = sim

        outputData['service_time'] = np.inf
        for c in outputData.columns:
            if c.startswith('service_time'):
                outputData.loc[outputData['serving_worker_type'] == c[13:], 'service_time'] =\
                    outputData.loc[outputData['serving_worker_type'] == c[13:], c]

        outputData = outputData[['arrival_time', 'service_time', 'done_time', 'sim']]

        resultData = resultData.append(outputData)
        print(sim)

    result = base_postprocessing(resultData, dates_inverse_mapper, dates_unique_months,
                                 initialQueueInprogressMatrix['InQueue'].sum(),
                                 initialQueueInprogressMatrix['InProgress'].sum(),
                                 startDate)

    result['real_date'] = pd.to_datetime(result['date'], format='%m-%Y')
    result = result.sort_values('real_date', ignore_index=True)

    return result

# TEST
# startDate = datetime.datetime.strptime('20/05/2019', '%d/%m/%Y')
# endDate = datetime.datetime.strptime('20/05/2020', '%d/%m/%Y')
# simulationsNum = 20
# avgModelMartix = pd.DataFrame([{'WorkerType': 'BaseWorker', 'BaseModel': '14', 'BlackBoxModel': '30'},
#                                {'WorkerType': 'AdvancedWorker', 'BaseModel': '10', 'BlackBoxModel': '15'}])
# stdModelMartix = pd.DataFrame([{'WorkerType': 'BaseWorker', 'BaseModel': '2', 'BlackBoxModel': '5'},
#                                {'WorkerType': 'AdvancedWorker', 'BaseModel': '1', 'BlackBoxModel': '3'}])
# workersNumDF = pd.DataFrame([{'date': '20/05/2019', 'BaseWorker': 100, 'AdvancedWorker': 30},
#                              {'date': '19/07/2019', 'BaseWorker': 80, 'AdvancedWorker': 50}])
# modelsIncomeDF = pd.DataFrame([{'date': '20/06/2019', 'BaseModel': 600, 'BlackBoxModel': 150},
#                                {'date': '20/08/2019', 'BaseModel': 800, 'BlackBoxModel': 500}])
# initialQueueInprogressMatrix = pd.DataFrame([{'ModelType': 'BaseModel', 'InProgress': '60', 'InQueue': '100'},
#                                              {'ModelType': 'BlackBoxModel', 'InProgress': '10', 'InQueue': '50'}])
#
# a = sm_advanced_main(startDate, endDate, simulationsNum,
#                      avgModelMartix, stdModelMartix,
#                      workersNumDF, modelsIncomeDF,
#                      initialQueueInprogressMatrix)
