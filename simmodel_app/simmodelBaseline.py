import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd


class Model:

    @staticmethod
    def days_to_solve(_avg_work_days_per_model, _sd_work_days_per_model):
        return int(round(max(np.random.normal(_avg_work_days_per_model, _sd_work_days_per_model, 1)[0], 1)))

    @staticmethod
    def days_in_progress(_days_to_solve):
        return int(round(np.random.uniform(1, _days_to_solve, 1)[0]))

    def __init__(self, _avg_work_days_per_model, _sd_work_days_per_model, init_date, is_new=True, is_in_progress=False):
        self._days_to_solve = self.days_to_solve(_avg_work_days_per_model, _sd_work_days_per_model)

        if is_new is False and is_in_progress is True:
            self._days_in_progress = self.days_in_progress(self._days_to_solve)
            self._init_date = None
            self._start_serving_date = init_date
        elif is_new is False and is_in_progress is False:
            self._days_in_progress = 0
            self._init_date = None
            self._start_serving_date = None
        elif is_new is True:
            self._days_in_progress = 0
            self._init_date = init_date
            self._start_serving_date = None

        self._is_model_done = False
        self._end_date = None

    def add_progress(self, progress, date):
        self._days_in_progress += progress
        if self._start_serving_date is None:
            self._start_serving_date = date

        if self._days_to_solve <= self._days_in_progress:
            self._is_model_done = True
            self._end_date = date + relativedelta(days=1)
            return 1
        else:
            return 0


def initialize_models(model_class, n_models, _avg_work_days_per_model, _sd_work_days_per_model, is_new=True):
    models = []
    for _ in range(n_models):
        models.append(model_class(_avg_work_days_per_model, _sd_work_days_per_model, is_new))


def date_range(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def sm_main(simulationsNum, nWorkers,
            modelsIncome, initialQueueSize, initialInprogressSize,
            avgWorkDaysPerModel, sdWorkDaysPerModel,
            startDate, endDate):


    # simulationsNum = 100
    # nWorkers = 100
    #
    # modelsIncome = 100
    # initialQueueSize = 200
    # initialInprogressSize = 100
    #
    # avgWorkDaysPerModel = 14
    # sdWorkDaysPerModel = 5
    #
    # startDate = datetime.date(2017, 1, 1)
    # endDate = datetime.date(2018, 1, 1)


    results = {}

    for sim in range(simulationsNum):

        modelsQueue = []
        doneModelsList = []
        modelsInMonth = 0
        modelsIncrement = 0
        savedDays = None

        results[sim] = {}

        for date in date_range(startDate, endDate):

            if date == startDate:
                if initialInprogressSize != 0:
                    for _ in range(initialInprogressSize):
                        model = Model(avgWorkDaysPerModel, sdWorkDaysPerModel, date, is_new=False, is_in_progress=True)
                        modelsQueue.append(model)
                if initialQueueSize != 0:
                    for _ in range(initialQueueSize):
                        model = Model(avgWorkDaysPerModel, sdWorkDaysPerModel, date, is_new=False, is_in_progress=False)
                        modelsQueue.append(model)

            if date.day == 1 or date == startDate:
                savedDays = int((date + relativedelta(months=1) - date).days)
                modelsIncrement = modelsIncome / savedDays
                modelsInMonth = 0

            if modelsIncome >= 1:
                modelsIncrementCum = modelsIncrement + modelsInMonth
                if date.day == savedDays and modelsIncrementCum < modelsIncome:
                    incr = int(modelsIncome) - int(modelsInMonth)
                else:
                    incr = int(modelsIncrementCum) - int(modelsInMonth)
                modelsInMonth = modelsIncrementCum

                for _ in range(incr):
                    model = Model(avgWorkDaysPerModel, sdWorkDaysPerModel, date, is_new=True, is_in_progress=False)
                    modelsQueue.append(model)

            if len(modelsQueue) >= nWorkers > 0:
                for model in modelsQueue[:nWorkers]:
                    result = model.add_progress(1, date)
                    if result == 1:
                        doneModelsList.append(model)
                modelsQueue = [model for model in modelsQueue if model._is_model_done is False]

            elif nWorkers > len(modelsQueue) > 0:
                for model in modelsQueue:
                    result = model.add_progress(1, date)
                    if result == 1:
                        doneModelsList.append(model)
                modelsQueue = [model for model in modelsQueue if model._is_model_done is False]


            """
            calc end of the month metrics 
            """
            if (date + relativedelta(days=1)).month != date.month:
                results[sim][str(date.month)+'-'+str(date.year)] = {}

                incomeNum = 0
                queueNum = 0
                inProgressNum = 0
                doneNum = 0

                sumWaitingTime = 0
                nWaitingModels = 0

                sumServingTime = 0
                nServingModels = 0

                sumTime2Done = 0
                nTime2Done = 0

                for model in modelsQueue:
                    if model._start_serving_date is None:
                        queueNum += 1
                    elif model._start_serving_date is not None:
                        inProgressNum += 1

                    if model._init_date is not None and model._init_date.month == date.month:
                        incomeNum += 1

                for model in doneModelsList:
                    if model._init_date is not None and model._init_date.month == date.month:
                        incomeNum += 1
                    if model._end_date.month == date.month:
                        doneNum += 1

                    if model._init_date is not None and model._end_date.month == date.month and\
                            model._start_serving_date is not None:
                        sumWaitingTime += (model._start_serving_date - model._init_date).days
                        nWaitingModels += 1

                    if model._start_serving_date is not None and model._end_date.month == date.month and\
                            model._end_date is not None:
                        sumServingTime += (model._end_date - model._start_serving_date).days
                        nServingModels += 1

                    if model._end_date.month == date.month and\
                            model._init_date is not None:
                        sumTime2Done += (model._end_date - model._init_date).days
                        nTime2Done += 1

                results[sim][str(date.month)+'-'+str(date.year)]['incomeNum'] = incomeNum
                results[sim][str(date.month)+'-'+str(date.year)]['queueNum'] = queueNum
                results[sim][str(date.month)+'-'+str(date.year)]['inProgressNum'] = inProgressNum
                results[sim][str(date.month)+'-'+str(date.year)]['doneNum'] = doneNum

                avgWaitingTime = sumWaitingTime / nWaitingModels if nWaitingModels != 0 else None
                avgServingTime = sumServingTime / nServingModels if nServingModels != 0 else None
                avgTime2Done = sumTime2Done / nTime2Done if nTime2Done != 0 else None
                results[sim][str(date.month)+'-'+str(date.year)]['avgWaitingTime'] = avgWaitingTime
                results[sim][str(date.month)+'-'+str(date.year)]['avgServingTime'] = avgServingTime
                results[sim][str(date.month)+'-'+str(date.year)]['avgTime2Done'] = avgTime2Done


    # d = {}
    # for date in results[0].keys():
    #     d[date] = {}
    #     for sim in range(simulationsNum):
    #         d[date][sim] = results[sim][date]
    #     d[date] = pd.DataFrame.from_dict(d[date], orient='index')
    #
    # final_df = None
    # for key, value in d.items():
    #
    #     intermediate_df = pd.DataFrame()
    #     for i in value.columns:
    #         intermediate_df.loc[key, i + '_avg'] = value[i].mean()
    #         intermediate_df.loc[key, i + '_std'] = value[i].std()
    #
    #     if final_df is None:
    #         final_df = intermediate_df
    #     else:
    #         final_df = final_df.append(intermediate_df)

    final_df = None
    for sim in results.keys():

        intermediate_df = pd.DataFrame.from_dict(results[sim], orient='index')
        intermediate_df = intermediate_df.reset_index().rename(columns={'index': 'date'})
        intermediate_df['sim'] = sim

        if final_df is None:
            final_df = intermediate_df
        else:
            final_df = final_df.append(intermediate_df)

    return final_df


sm_main(simulationsNum=100,
        nWorkers=50,
        modelsIncome=100,
        initialQueueSize=200,
        initialInprogressSize=50,
        avgWorkDaysPerModel=14,
        sdWorkDaysPerModel=5,
        startDate=datetime.date(2017, 1, 1),
        endDate=datetime.date(2018, 1, 1))
