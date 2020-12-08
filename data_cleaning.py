import pandas as pd
from pandas import ExcelWriter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from base_classes import FileManagement
from rename_info import colnames, TTM_indicator_calculation
pd.set_option('mode.chained_assignment', None)

class DataCleaningPro(FileManagement):
    def __init__(self, fmClass):
        super(FileManagement, self).__init__()
        self.dataframes_Used = fmClass.dataframes_Used
        self.cate = ['predict', 'PPI', 'IPI']
        self.nan_info = {}
        self.secondary_dataframes = {}
        self.final_dataframes = {}
        self.tobeDelete_objs = {}

        self.ROEdata = fmClass.ROEdata
        self.yearPeriod = fmClass.yearPeriod
        self.standardize_Info = {}
        self.weightInfo = {}
        self.weight = pd.DataFrame()
        self.close = fmClass.close
        self.industry_datasets_raw = {}
        self.industry_datasets = {}
        self.buy_sell_signal = {}
        self.industry_indicators_score = {}
        self.complex_rawdata = pd.DataFrame()

        # Performance
        self.portfolio = {}
        self.indicator_performance = {}
        self.indicator_picker = pd.DataFrame()
        self.complex_ind = pd.DataFrame()

        self.PPI_ind = colnames.PPI_COLNAMES.value
        self.IPI_ind = colnames.IPI_COLNAMES.value
        self.end_date = "2020-10"
        self.startyear = 2009
        self.endyear = 2020
        self.endmonth = 10
        self.period = [str(i) for i in range(2010, 2021)]
        self.libor = {
            "2020": 0.0097,
            "2019": 0.0237,
            "2018": 0.0215,
            "2017": 0.0168,
            "2016": 0.0116,
            "2015": 0.0063,
            "2014": 0.0053,
            "2013": 0.0069,
            "2012": 0.0113,
            "2011": 0.0079,
            "2010": 0.0096,
            "2009": 0.0033
        }

    def rename_df_ENG(self, df, cate):
        if cate == "PPI":
            df = df.rename(columns=self.PPI_ind)
        elif cate == "IPI":
            df = df.rename(columns = self.IPI_ind)
        return df

    def weighted_calculation(self):
        '''
        To suppliment the data within an industry, between companies.
        The logic is to use the PE ratio to determine the company status within the industry, the use the corresponding level within the industry for each period.
        :param dataframeName: The target dataframe name.
        :return: No return, finishing the calculation of the weighted dataframe. Adjusted according to month by using PE.
        '''
        for ym in self.yearPeriod:
            ROE = self.ROEdata[ym]
            base = self.dataframes_Used['baseCompInfo']
            ROE = pd.merge(base, ROE, on=['stockCode', 'stockName'], how="inner")
            ROE = self.rename_dt_Columns(ROE, 3)
            dateCols = ROE.columns.values[3:]

            # To transfer the raw PE value into weighted value and store it into the weighted dataframe
            self.industry_max_min(ROE, "industry", dateCols)
            industry = set(list(base['industry']))
            ROE = self.std_data_groupby_cate(ROE, industry, 'industry', dateCols)
            sumdict = self.industry_sum(ROE, 'industry', dateCols)
            self.calculated_weight(ROE, sumdict, industry, 'industry', dateCols, ym)

        for e, i in enumerate(self.weightInfo):
            if e==0:
                weight = self.weightInfo[i]
            else:
                data = self.weightInfo[i]
                weight = pd.merge(weight, data, on=["stockCode", "industry", "stockName"], how="inner")
        self.weight = weight


    def standardize_rawDF(self, rawDF, col, year, period, groupby_cate, groupby_cateName):
        '''
        To standardize the raw data by using the max-min scaler.
        :param rawDF: Raw data with data larger and smaller than 0.
        :param col: One columns of data that needs to be dealt.
        :param year: All of the elements (none duplicate) of the year feature.
        :param period: All of the elements (none duplicate) of the period feature.
        :param groupby_cate: All of the elements (none duplicate) of the group by feature.
        :param groupby_cateName: The groupby feature name.
        :return: The standardized dataframe (just one target column) of the raw dataframe.
        '''
        subdf = []
        Fcol = ["stockCode", "stockName", "industry"] # The columns that is used as indexes.
        for y in year:
            for rp in period:
                subdata = rawDF[Fcol + [col]][(rawDF['year'] == y) & (rawDF['reportPeriod'] == rp)]
                subdata = subdata.rename(columns={col: y+"-"+rp})
                self.industry_max_min(subdata, "industry", [y+"-"+rp])
                if len(subdata)==0:
                    continue
                else:
                    subdata = self.std_data_groupby_cate(subdata, groupby_cate, groupby_cateName, [y+"-"+rp])
                    subdata['year'] = y
                    subdata['reportPeriod'] = rp
                    subdata = subdata.rename(columns={y+"-"+rp: col})
                    subdf.append(subdata)
        subdf = pd.concat(subdf)
        rawDF = rawDF.drop(col, axis=1)
        rawDF = pd.merge(rawDF, subdf, on=['stockCode', 'stockName', 'year', 'reportPeriod', 'industry'], how="inner")
        return rawDF

    def PPI_data_calcul(self):
        '''
        To combine the periodic performance indicators data.
        :return: Calculated dataframe with just a single variable for each category.
        '''
        data = self.dataframes_Used['PPI']
        data = data.rename(columns=colnames.PPI_COLNAMES.value)
        data['reportPeriod'] = data['reportPeriod'].apply(self.change_name)
        cols_cr = TTM_indicator_calculation.ppi_cal_cr.value
        cols_yoy = TTM_indicator_calculation.ppi_cal_yoy.value
        cols = ['stockCode', 'stockName', 'year', 'reportPeriod', 'industry']
        cols.extend(cols_cr)
        cols.extend(cols_yoy)
        data = data[cols]

        base = self.dataframes_Used['baseCompInfo']
        industry = set(list(base['industry']))
        year = list(set(list(data['year'])))
        year.sort()
        reportPeriod = list(set(list(data['reportPeriod'])))
        reportPeriod.sort()
        for c in cols_cr:
            self.secondary_dataframes['PPI'] = data
            self.NaN_suppliment_company('PPI', [c])
            data = self.cr_growthRate_compute(data, c)
            self.industry_datasets_raw["PPI"] = data
            data = self.standardize_rawDF(data, c, year, reportPeriod, industry, 'industry')
            data[c + "_checkNaN"] = data[c].apply(lambda x: 1 if np.isnan(x) else 0)
        for c in cols_yoy:
            self.secondary_dataframes['PPI'] = data
            self.NaN_suppliment_company('PPI', [c])
            data = self.yoy_growthRate_compute(data, c)
            self.industry_datasets_raw["PPI"] = data
            data = self.standardize_rawDF(data, c, year, reportPeriod, industry, 'industry')
            data[c + "_checkNaN"] = data[c].apply(lambda x: 1 if np.isnan(x) else 0)
        self.secondary_dataframes['PPI'] = data

    def IPI_data_calcul(self):
        '''
        To combine the instant data together.
        The data from the express sheet and advanced notice sheet is combined into the financial data sheet.
        :return: Calculated dataframe with just a single variable for each category.
        '''
        data = self.dataframes_Used['IPI']
        data = data.rename(columns=colnames.IPI_COLNAMES.value)
        data['reportPeriod'] = data['reportPeriod'].apply(self.change_name)
        cols = TTM_indicator_calculation.ipi_cols.value
        base = self.dataframes_Used['baseCompInfo']
        industry = set(list(base['industry']))
        year = list(set(list(data['year'])))
        year.sort()
        reportPeriod = list(set(list(data['reportPeriod'])))
        reportPeriod.sort()

        for c in cols:
            if c == "GrowthNPtoShareholder":
                # Combine the Growth Net Profit to Shareholder data
                tarCols = [c, 'Express_' + c, 'ADnotice_Up_' + c, 'ADnotice_Down_' + c]
                # The new dataframe columns should be the same with the previous one when using apply function, or receieve "NaN".
                data[tarCols] = data[tarCols].apply(self.combine_variables, axis=1)
                del data['Express_' + c]
                del data['ADnotice_Up_' + c]
                del data['ADnotice_Down_' + c]
                data = self.standardize_rawDF(data, c, year, reportPeriod, industry, 'industry')
                self.secondary_dataframes['IPI'] = data
                self.NaN_suppliment_company('IPI', [c])
                data = self.yoy_growthRate_compute(data, c)
                self.industry_datasets_raw["IPI"] = data
                data = self.standardize_rawDF(data, c, year, reportPeriod, industry, 'industry')
            else:
                tarCols = [c, 'Express_' + c]
                data[tarCols] = data[tarCols].apply(self.combine_variables, axis=1)
                del data['Express_' + c]
                self.secondary_dataframes['IPI'] = data
                self.NaN_suppliment_company('IPI', [c])
                data = self.yoy_growthRate_compute(data, c)
                self.industry_datasets_raw["IPI"] = data
                data = self.standardize_rawDF(data, c, year, reportPeriod, industry, 'industry')
        self.secondary_dataframes['IPI'] = data

    def PPI_nan_deal(self):
        self.count_nan("PPI")
        data = self.secondary_dataframes['PPI']
        todelete, tmp = self.tobeDelete_objs['PPI']
        del tmp
        for i in todelete:
            data = data.drop(data[data['stockCode']==i].index, axis=0)
        tarcols = TTM_indicator_calculation.ppi_cols.value
        self.NaN_suppliment_company('PPI', tarcols)

    def IPI_nan_deal(self):
        self.count_nan("IPI")
        data = self.secondary_dataframes['IPI']
        todelete, tmp = self.tobeDelete_objs["IPI"]
        del tmp
        for i in todelete:
            data = data.drop(data[data['stockCode']==i].index, axis=0)
        tarcols = TTM_indicator_calculation.ipi_cols.value
        self.NaN_suppliment_company("IPI", tarcols)

    def predict_nan_deal(self):
        '''
        To deal with the NaN data.
        Fill the NaN data with the former value, if there is no former value, then fillin with 0
        :return: No return
        '''
        data = self.dataframes_Used['predict']
        base = self.dataframes_Used['baseCompInfo'][['stockCode', 'stockName', 'industry']]
        data = pd.merge(base, data, on=['stockCode', 'stockName'], how="inner")
        data['reportPeriod'] = data['month'].apply(lambda x: "{:0>2}".format(x))
        del data['month']
        self.secondary_dataframes['predict'] = data

        self.count_nan('predict')
        todelete, tmp = self.tobeDelete_objs["predict"]
        del tmp
        for i in todelete:
            data = data.drop(data[data['stockCode']==i].index, axis=0)
        self.industry_datasets_raw["predict"] = data
        tarcols = TTM_indicator_calculation.predict_cols.value

        industry = set(list(base['industry']))
        year = list(set(list(data['year'])))
        year.sort()
        reportPeriod = list(set(list(data['reportPeriod'])))
        reportPeriod.sort()

        for c in tarcols:
            data = self.standardize_rawDF(data, c, year, reportPeriod, industry, 'industry')
        self.secondary_dataframes['predict'] = data
        self.NaN_suppliment_company("predict", tarcols)

    def NaN_suppliment_company(self, dataframeName, tarcols):
        '''
        To suppliment the data within a company. Just within a company, not industry.
        The logic is to use the last period of data to fulfil the missing data after the last period.
        :param dataframeName: The target dataframe name.
        :param tarcols: The target columns that need to be deal with.
        :return:
        '''
        data = self.secondary_dataframes[dataframeName]
        company = np.array(data['stockCode'].drop_duplicates(keep='first'))
        for c in tarcols:
            newval = []
            rec = 0
            for i in company:
                deal = data[data['stockCode']==i]
                list = np.array(deal[c])
                for r in list:
                    if not (np.isnan(r) or np.isinf(r)):
                        rec = r
                    else:
                        rec = rec # + randint(0,9) * 0.001 * rec # Add in random disturbance
                    newval.append(rec)
            del data[c]
            pos = len(data.columns.values)
            data.insert(pos, c, newval)
        newcols = ['stockCode', 'stockName', 'year', 'reportPeriod', 'industry']
        newcols.extend(tarcols)
        data = data[newcols]
        self.secondary_dataframes[dataframeName] = data

    def weighted_close(self):
        # Weighted data
        years = list(self.weightInfo.keys())
        years.sort()
        weights = []
        for y in years:
            w = self.weightInfo[y].set_index(['stockCode', 'industry', 'stockName'])
            weights.append(w)
        weights = pd.DataFrame(pd.concat(weights, axis=1, join="inner")).reset_index()

        close = self.close
        weights = weights.sort_values(by=['stockCode', 'industry']).fillna(0)
        close = close.sort_values(by=['stockCode', 'industry']).fillna(0)

        industry = list(set(weights['industry']))
        datelist = list(weights.columns.values)[3:]
        indClose = []
        for i in industry:
            subClose = [i]
            for d in datelist:
                closeArr = np.array(close[d][close['industry']==i]).reshape(1, -1)
                weightArr = np.array(weights[d][weights['industry']==i]).reshape(-1, 1)
                subClose.append(np.dot(closeArr, weightArr)[0][0])
            indClose.append(pd.DataFrame(subClose).T)
        indClose = pd.concat(indClose)
        indClose.columns = ['industry'] + datelist
        self.close = indClose # The close index of the trading day at the end of each month

    def count_nan(self, dataframeName):
        data = self.dataframes_Used[dataframeName]
        statNaN_df = pd.DataFrame(set(list(data['stockCode'])))
        statNaN_df.columns = ['stockCode']
        cols = [i for i in data.columns.values if "_checkNaN" in i]
        for c in cols:
            tmp = data[['stockCode', c]]
            stat = tmp.groupby('stockCode').sum()
            unit = len(data)/len(stat)
            stat[c[:-9] + "_percnt"] = stat[c].apply(lambda x: round(x * 100 / unit, 2))
            stat = stat.reset_index()
            statNaN_df = pd.merge(statNaN_df, stat[['stockCode', c[:-9] + "_percnt"]], on='stockCode', how="inner")
        statNaN_df['NaNcols'] = statNaN_df.iloc[:, 1:].apply(lambda x: np.sum(x>80), axis=1) # Record the columns that the missing values are larger than 80%
        statNaN_df['NaN_percnt'] = statNaN_df['NaNcols'].apply(lambda x: x / (len(statNaN_df.columns.values)-2)) # Counted the missing value for each of the stocks
        tobeDelete = list(statNaN_df[statNaN_df['NaN_percnt'] > 0.3]['stockCode']) # Delete those missing stocks that have missing values over 30% in all of the features
        self.tobeDelete_objs[dataframeName] = [tobeDelete, statNaN_df]

    def change_name(self, val):
        '''
        To change the report period into month number.
        :param val: the single value of the dataframe elements
        :return: the corresponding value of the input dataframe element
        '''
        if val == '一季': val = '03'
        if val == '中报': val = '06'
        if val == '三季': val = '09'
        if val == '年报': val = '12'
        return val

    def combine_variables(self, colVals):
        '''
        To combine the Express data and Advance Notice to the financial data.
        :param raw_value: The raw value of the data.
        :return: The new value of the data.
        '''
        mom_colVal = colVals[0]
        child_colVal = colVals[1:]
        if np.isnan(mom_colVal):
            if np.isnan(child_colVal[0]) and len(child_colVal) > 2:
                if not np.isnan(child_colVal[1]) and np.isnan(child_colVal[2]):
                    colVals[0] = (child_colVal[1] + child_colVal[0]) / 2
                elif np.isnan(child_colVal[1]) or np.isnan(child_colVal[2]):
                    colVals[0] = child_colVal[1] if np.isnan(child_colVal[1]) else child_colVal[2]
                else:
                    colVals[0] = np.NaN
            else:
                colVals[0] = child_colVal[0]
        return colVals

    def cr_growthRate_compute(self, df, varName):
        '''
        To calculate the Chain Ratio growth rate of the target variable.
        If the raw value of certain period is NULL, then there will be two NULL values in the growth rate list.
        Because there is no compare objects.
        :param df: The dataframe that includes the variables needed to be computed.
        :param varName: The target variable name.
        :return: New dataframe with calculated growth rate, the original target variable will be replace by the new value.
        '''
        np.seterr(divide='ignore', invalid='ignore')
        array = np.array(df[varName])[::-1]
        array = list((array[:-1] - array[1:]) / array[1:])[::-1]
        array.append(np.NaN)
        del df[varName]
        pos = len(df.columns.values)
        df.insert(pos, varName, array)
        return df

    def yoy_growthRate_compute(self, df, varName):
        '''
        To calculate the YoY growth rate of the target variable.
        If the raw value of certain period is NULL, then there will be two NULL values in the growth rate list.
        Because there is no compare objects.
        :param df: The dataframe that includes the variables needed to be computed.
        :param varName: The target variable name.
        :return: New dataframe with calculated growth rate, the original target variable will be replace by the new value.
        '''
        np.seterr(divide='ignore', invalid='ignore')
        array = np.array(df[varName])[::-1]
        array = list((array[:-4] - array[4:]) / array[4:])[::-1]
        for i in range(4):
            array.append(np.NaN)
        del df[varName]
        pos = len(df.columns.values)
        df.insert(pos, varName, array)
        return df

    def rename_dt_Columns(self, data, startInd):
        '''
        Rename the columns that are written in datetime format.
        :param data: The dataframe that needed to be renamed.
        :param startInd: The start renamed position.
        :return: Renamed dataframe
        '''
        org_cols = list(data.columns.values[:startInd])
        cols = data.columns.values[startInd:]
        cols = [i for i in map(lambda x: str(x.year) + "-" + "{:0>2}".format(str(x.month)), cols)]
        org_cols.extend(cols)
        data.columns = org_cols
        return data

    def industry_max_min(self, data, groupby_Var, cols):
        '''
        To calculate the max min value of the dataframe.
        :param data: The target dataframe to be calculated.
        :param groupby_Var: The group by feature name, string type.
        :param cols: The target columns within the dataframe to be calculated. Date columns, formatting in "yyyy-mm".
        :return: No return, the result of each month was stored in a dictionary. The key is 'yyyy-mm', the value is the dataframe.
        '''
        for i in cols:
            tmp = data[[groupby_Var, i]]
            res = tmp.groupby(by=groupby_Var).max()
            res2 = tmp.groupby(by=groupby_Var).min()
            res = pd.merge(res, res2, left_index=True, right_index=True).reset_index()
            res.columns = [groupby_Var, "max", "min"]
            self.standardize_Info[i] = res

    def industry_sum(self, data, groupby_Var, cols):
        '''
        To calculate the summary of the dataframe.
        :param data: The target dataframe to be calculated.
        :param groupby_Var: The group by feature name, string type.
        :param cols: The target columns within the dataframe to be calculated.
        :return: Return the summary dictionary, the key is the datetime "yyyy-mm", the value is the corresponding dataframe.
        '''
        sum_dict = {}
        for i in cols:
            tmp = data[[groupby_Var, i]]
            res = tmp.groupby(by=groupby_Var).sum().reset_index()
            res.columns = [groupby_Var, "sum"]
            sum_dict[i] = res
        return sum_dict

    def std_data_groupby_cate(self, data, groupby_cate, groupby_Var, cols):
        '''
        To standardized the raw data, for the later calculation.
        :param data: The target raw data dataframe.
        :param groupby_cate: The groupby feature, a list that consists of all the categories within the feature.
        :param groupby_Var: The group by feature name, string type.
        :param cols: The target columns within the dataframe to be calculated. Another groupby features, can be date "yyyy-mm".
        :return: A standardized dataframe.
        '''
        stdList = []
        for i in groupby_cate:
            tardf = data[data[groupby_Var]==i]
            for j in cols:
                stdDf = self.standardize_Info[j]
                max = stdDf[stdDf[groupby_Var]==i]['max'].values[0]
                min = stdDf[stdDf[groupby_Var]==i]['min'].values[0]
                tardf[j] = tardf[j].apply(lambda x: round(float((x - min) / (max - min)), 3))
            stdList.append(tardf)
        data = pd.concat(stdList)
        return data

    def calculated_weight(self, data, sumdict, groupby_cate, groupby_Var, cols, year):
        '''
        To standardized the raw data, for the later calculation.
        :param data: The target raw data dataframe.
        :param sumdict: The summary dictionary group by the groupby_cate, using the datetime as key, and the summary dataframe as value.
        :param groupby_cate: The groupby feature, a list that consists of all the categories within the feature.
        :param groupby_Var: The group by feature name, string type.
        :param cols: The target columns within the dataframe to be calculated.
        :param year: Year variable, formated in "yyyy".
        :return: No return, the result is stored in the weighted dataframe.
        '''
        res = []
        # Here is grouped by industry.
        for i in groupby_cate:
            tardf = data[data[groupby_Var]==i]
            # Here is datetime "yyyy-mm"
            for j in cols:
                stdDf = sumdict[j]
                summary = stdDf[stdDf[groupby_Var]==i]['sum'].values[0] # Sum up the data group by industry, corresponding to datetime "yyyy-mm"
                tardf[j] = tardf[j].apply(lambda x: round(x / summary, 3))
            res.append(tardf)
        self.weightInfo[year] = pd.concat(res).fillna(0)

    def industry_indicators(self):
        '''
        To weighted the stocks indicators within each industry.
        :return: No return, the dataset is stored in the variable. The dataset is grouped by according to industry, date ('yyyy-mm')
        '''
        # Temporary
        sheetname = ["IPI", "PPI", "predict"]
        dfDict = {}
        for s in sheetname:
            data = self.secondary_dataframes[s]
            dfDict[s] = data
        weight = self.weight

        colsdict = {
            'IPI': TTM_indicator_calculation.ipi_cols.value,
            'PPI': TTM_indicator_calculation.ppi_cols.value,
            'predict': TTM_indicator_calculation.predict_cols.value
        }
        baseCol = TTM_indicator_calculation.weight_cols.value
        newCol = ['industry', 'ym']

        self.weighted_close()

        for s in sheetname:
            data = dfDict[s]
            data["ym"] = data.apply(lambda x: str(x['year']) + "-" + "{:0>2}".format(x['reportPeriod']), axis=1)
            col = colsdict[s]
            yearmon = list(set(list(data['ym'])))
            yearmon.sort()

            # Multiply the weight to each stock indicators group by industry
            newdata = []
            for ym in yearmon:
                if (int(ym[:4]) >= self.startyear and int(ym[:4]) < self.endyear) or (int(ym[:4]) == self.endyear and int(ym[5:]) < self.endmonth):
                    cal_df = data[baseCol + col + ['ym']][data['ym']==ym]
                    weight_df = weight[baseCol + [ym]]
                    cal_df = pd.merge(cal_df, weight_df, on=baseCol, how="inner")
                    for c in col:
                        cal_df[c] = cal_df.apply(lambda x: x[c] * x[ym], axis=1)
                    del cal_df[ym]
                    newdata.append(cal_df)
            newdata = pd.concat(newdata)

            industry_data = newdata[newCol].drop_duplicates()
            for c in col:
                tarcol = newCol + [c]
                df = newdata[tarcol].groupby(by=newCol).sum()
                industry_data = pd.merge(industry_data, df, on=newCol, how="inner")
            self.industry_datasets[s] = industry_data

    def buysell_indicator(self, thres):
        '''
        To change the industry score into long short signals.
        :param thres: The threshold to control on what level the signal is long / short.
        :return: No return, the data is stored in two datasets. The score and the long short signal datasets.
        '''
        for s in self.cate:
            data = self.industry_datasets[s]
            indicatorsScoreCols = data.columns.values[2:]
            longShortCols = []
            for c in indicatorsScoreCols:
                avg = data[["industry", c]].groupby(by=["industry"]).mean().reset_index().rename(columns={c: "avg"})
                data = pd.merge(data, avg, on="industry", how="left")
                # The buy / sell signal occurs when the indicator is larger than the enlarge the average / reduce the average
                # Since the A stock market performs bad after the indicators shows a good signal, we upside down the long short accordingly
                data[c + "_buy"] = data.apply(lambda x: (x[c] < x["avg"] * (1 - thres)) * 1, axis=1)
                data[c + "_sell"] = data.apply(lambda x: (x[c] > x["avg"] * (1 + thres)) * 1, axis=1)
                del data['avg']
                longShortCols.append(c + "_buy")
                longShortCols.append(c + "_sell")
            longShortCols = list(data.columns.values[:2]) + longShortCols
            indicatorsScoreCols = list(data.columns.values[:2]) + list(indicatorsScoreCols)
            self.buy_sell_signal[s] = data[longShortCols]
            self.industry_indicators_score[s] = data[indicatorsScoreCols]

    def single_buysell_perf(self, writeExcel=False):
        '''
        To turn the long short signals into investing period. And measure the portfolio value according to the close index of the industry.
        :param writeExcel: Whether to write into an Excel.
        :return: No return, directly to calculate the portfolio performance.
        '''
        close = self.close
        col_dict = {
            "IPI": TTM_indicator_calculation.ipi_cols.value,
            "PPI": TTM_indicator_calculation.ppi_cols.value,
            "predict": TTM_indicator_calculation.predict_cols.value
        }

        idx_ind_score = {}
        financial_data = []
        for s in self.cate:
            idx_ind_subscore = []
            data = self.buy_sell_signal[s]

            industry = list(data['industry'])
            industry = sorted(set(industry), key=industry.index)
            col = col_dict[s]

            # To calculate each indicators one by one
            for c in col:
                indScore = []
                for i in industry:
                    subdf = data[["industry", "ym", c + "_buy", c + "_sell"]][data['industry']==i]
                    # Buy decision - money cost
                    buySig = np.array(subdf[c + "_buy"])
                    # Sell decision - money gain
                    sellSig = np.array(subdf[c + "_sell"])
                    tmp = buySig - sellSig
                    buyrec, sellrec, pos = 0, 0, 0
                    tradeSig = []
                    for t in tmp:
                        if buyrec==0 and sellrec==0 and t > 0:
                            tradeSig.append(1)
                            buyrec = 1
                        elif buyrec==1 and sellrec==0 and t < 0:
                            tradeSig.append(-1)
                            buyrec = 0
                            sellrec = 1
                        elif buyrec==0 and sellrec==1 and t > 0:
                            tradeSig.append(1)
                            buyrec = 1
                            sellrec = 0
                        elif buyrec==1 and sellrec==0 and t >= 0:
                            tradeSig.append(1)
                        else:
                            tradeSig.append(0)
                    subdf.insert(4, c, tradeSig)
                    subclose = close[close["industry"]==i].T.reset_index().iloc[1:, :]
                    subclose.columns = ["ym", c + "_close"]
                    subdf = pd.merge(subdf[["industry", "ym", c]], subclose, on="ym", how="inner")
                    subdf[s+c+'_res'] = subdf.apply(lambda x: np.abs(x[c] * x[c + "_close"]), axis=1)
                    subdf = subdf.rename(columns={c: s+c})

                    indScore.append(subdf[['industry', 'ym', s+c, s+c+'_res']])

                indScore = pd.concat(indScore).set_index(['industry', 'ym']).fillna(0)
                idx_ind_subscore.append(indScore)

            if s == "predict":
                idx_ind_score["predict"] = pd.concat(idx_ind_subscore, axis=1).reset_index()
            else:
                financial_data.append(pd.concat(idx_ind_subscore, axis=1))

        idx_ind_score["financial"] = pd.concat(financial_data, axis=1).reset_index()
        if writeExcel:
            write = ExcelWriter("industry_score.xlsx")
            for k in ['predict', 'financial']:
                idx_ind_score[k].to_excel(write, k, index=False)
                self.portfolio_performance(idx_ind_score[k], "portfolio_" + k, k, writeExcel)
            write.save()
        else:
            for k in ['predict', 'financial']:
                self.portfolio_performance(idx_ind_score[k], "portfolio_" + k, k, writeExcel)


    def portfolio_performance(self, data, excelName, dataCate, writeExcel=False):
        '''
        To calculate the value of the portfolio. Assume each industry accounts for a unit value.
        :param data: The long/short and close price data.
        :param excelName: The excel name of the stored portfolio value time series data.
        :param dataCate: The category of data, either 'predict' or 'financial'.
        :return: No return, stored the calculated portfolio value dataframe.
        '''
        allCols = list(data.columns.values)
        cols = allCols[:2] + [i for i in allCols[2:] if '_res' in i]
        idxcols = [i for i in allCols[2:] if '_res' not in i]
        industry = list(set(list(data['industry'])))
        industry.sort()
        df = []
        for i in industry:
            subdf = data[data['industry']==i]

            # The "cols" is indicators, the rows are dates' value
            for l, c in enumerate(cols[2:]):
                pos = cols.index(c)
                price = list(subdf[c])
                idx = list(subdf[idxcols[l]])
                val = []
                long = 0 # The amount of stocks hold
                pval = 1 # Current portfolio value
                rival = 1 # Raw investment portfolio value
                diff = 0
                for e, p in enumerate(price):
                    if int(idx[e]) >= 0 and float(p) > 0:
                        if long == 0 and idx[e] > 0:
                            long = rival / p # Assume all of the money can change into stocks, not rests
                            pval = rival + diff
                        else:
                            pval = long * p + diff
                    elif int(idx[e]) == -1:
                        pval = long * p + diff
                        long = 0
                        diff = pval - rival
                        rival = 1
                    val.append(pval)
                subdf.insert(pos, c + "_val", val)

            df.append(subdf)
        df = pd.concat(df)
        self.portfolio[dataCate] = df

        if writeExcel:
            write = ExcelWriter(excelName + ".xlsx")
            df.to_excel(write, "data", index=False)
            write.save()


    def performance_calcul(self, data, dataCate=None, plotReq=False, startDate="2009-03", endDate="2020-09"):
        '''
        To calculate the performance of the portfoilo.
        :param data: The value of the portfolio.
        :param dataCate: The kind of data, financial indicators or predicted ones.
        :param plotReq: Whether to plot a picture and store it.
        :param startDate: The start date of the observation.
        :param endDate: The end date of the observation.
        :return: No return, the performance of the portfolio.
        '''
        Cols = [i for i in data.columns.values if "_res_val" in i]
        rawCapital = len(data['industry'].drop_duplicates())

        datePeriod = self.datetime_func(startDate, endDate)
        subdf = []
        for i in datePeriod:
            subdf.append(data[data['ym']==i])
        data = pd.concat(subdf)

        df = []
        for c in Cols:
            subdata = data[['ym', c]]
            subdata = subdata.groupby(['ym']).sum()
            df.append(subdata)
        df = pd.concat(df, axis=1)

        # Portfolio performance plot
        if plotReq:
            dt = list(df.index)
            dt.sort()
            fig = plt.figure(figsize=(20, 12))
            ax = fig.add_subplot(title="{} indicators portfolio performance".format(dataCate))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            for e, c in enumerate(Cols):
                ax.axis = c
                if len(dt) > 4 * 10:
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
                ax.plot(dt, np.array(df[c]), label=c)
            ax.legend(loc='best', frameon=False)
            plt.savefig("{} portfolio performance.png".format(dataCate))

        # Portfolio return indicators calculation
        newCols = []
        for e, c in enumerate(Cols):
            df[c[:-8] + "_returnRate"] = df.apply(lambda x: (x[c] - rawCapital) / rawCapital, axis=1)
            newCols.append(c[:-8] + "_returnRate")
        newdf = df[newCols].T
        newdf = self.performance_5_scores(newdf, df, Cols)

        resDF = newdf[['AnnRR', 'SigRR', 'SharpR', 'WLR', 'MDD']]
        print("resDF:\n", resDF)

        if dataCate: self.indicator_performance[dataCate] = newdf

    def performance_5_scores(self, newdf, portfolioValDF, portfolioValCols):
        '''
        To calculate the five scores of the portfolio.
        :param self:
        :param newdf: The dataframe that indicates the return rate of the portfolio in every period.
        :param portfolioValDF: The portfolio value of all the indicators in a period of time.
                               The columns are the indicators. The indexes are the industry and the date.
        :param portfolioValCols: The indicators that used to construct the portfolio.
        :return: The calculated dataframe with the performance result of the portfolio within a period of time.
        '''
        dateCols = newdf.columns.values
        datediff = int(dateCols[1][-2:]) - int(dateCols[0][-2:])

        # 1. Annual Rate of Return
        for e, d in enumerate(dateCols):
            # Annualize the return in each period. The first period does not count.????
            if e != 0:
                newdf[d+"_RR"] = newdf[d].apply(lambda x: x * (12 / datediff) / (int(e / (12 / datediff)) + 1))
        newdf["AnnRR"] = newdf.iloc[:, -1] # Calculated according to the last trading date's portfolio value.

        # 2. Annual Rate of Volatility
        newdf["AvgRR"] = newdf.iloc[:, 1:-1].apply(lambda x: np.mean(x), axis=1)

        SigCols = []
        RRCols = []
        for d in dateCols[1:]:  # The first period has zero return rate
            newdf["ER_" + d] = newdf.apply(lambda x: x[d] - x["AvgRR"], axis=1)
            RRCols.append("ER_" + d)
            SigCols.append("SigR_" + d)
            if len(SigCols) > 1:
                newdf["SigR_" + d] = newdf.apply(lambda x: np.sqrt(sum(np.square(x[RRCols])) / (len(SigCols) - 1)),
                                                 axis=1)
        SigCols.pop(0)
        newdf["SigRR"] = newdf[SigCols].apply(lambda x: np.sqrt(np.nansum(x) / (len(SigCols) - 1)),
                                              axis=1)  # The first period is not included inside
        # 3. Sharpe Ratio
        ShpCols = []
        for d in dateCols[3:]:
            rf = self.libor[d[:4]]
            newdf['SharpR_' + d] = newdf.apply(lambda x: (x[d] - rf) / x["SigR_" + d], axis=1)
            ShpCols.append('SharpR_' + d)
        newdf["SharpR"] = newdf[ShpCols].apply(lambda x: np.mean(x), axis=1)
        # 4. Win Loss Ratio
        WLRCols = []
        for d in dateCols[1:]:
            newdf["WLR_" + d] = newdf.apply(lambda x: 1 if x[d] > 0 else 0, axis=1)
            WLRCols.append("WLR_" + d)
        newdf["WLR"] = newdf.apply(lambda x: sum(x[WLRCols]) / len(WLRCols), axis=1)
        # 5. Max Drawdown
        maxP = portfolioValDF[portfolioValCols].T.cummax(axis=1)
        mdd = []
        for i in range(len(maxP)):
            cummaxP = np.array(maxP.iloc[i, :])
            rawP = np.array(portfolioValDF[portfolioValCols].T.iloc[i, :])
            maxDrawDown = rawP / cummaxP - 1
            mdd.append(maxDrawDown)
        mdd = pd.DataFrame(mdd)
        mddCols = ["MDD_" + i for i in dateCols]
        mdd.columns = mddCols
        newdf = newdf.reset_index()
        newdf = pd.merge(newdf, mdd, left_index=True, right_index=True)
        newdf['MDD'] = newdf[mddCols].apply(lambda x: min(x), axis=1)
        newdf = newdf.set_index("index").sort_values(by="AnnRR", ascending=False)
        return newdf

    def check_correlation_vars(self):
        '''
        Check the correlation of the dataset.
        :return: No return, store the correlation of the dataset, get ready for the selection of the indicators.
        '''
        rawdata = []
        for s in ['financial', 'predict']:
            df = self.indicator_performance[s]
            rawdata.append(df)
        rawdata = pd.concat(rawdata, sort=False)
        ERCols = [i for i in rawdata.columns.values if "ER_" in i]
        ERCols.sort()
        data = rawdata[["AnnRR"] + ERCols].sort_values(by="AnnRR", ascending=False)

        # Suppliment the missing data -> financial indicators
        newdata = []
        tobedelete = 0
        trigger = True
        for i in range(len(data)):
            line = data.iloc[i, 1:]
            newline = [line[0]]
            former = line[0]
            for l in line[1:]:
                if np.isnan(l):
                    newline.append(former)
                    if np.isnan(former) and trigger: tobedelete += 1
                else:
                    newline.append(l)
                    former = l
            newdata.append(pd.DataFrame(newline).T)
            if list(data.index)[i][:7] != "predict": trigger = False
        newdata = pd.concat(newdata)
        newdata.index = data.index
        newdata.columns = ERCols
        newdata = newdata.iloc[:, tobedelete:].T

        corrDf = newdata.corr()
        corrInfo = []
        for e, i in enumerate(corrDf.columns.values):
            arr = np.mean(list(corrDf[i])[:e] + list(corrDf[i])[e:])
            corrInfo.append(round(float(arr), 2))
        corrInfo = pd.DataFrame(corrInfo)
        corrInfo.set_index(keys=corrDf.columns.values, inplace=True)
        corrInfo.columns = ["corrInfo"]
        self.indicator_picker = pd.merge(rawdata[['AnnRR', 'SigRR', 'SharpR', 'WLR', 'MDD']], corrInfo,
                                    left_index=True, right_index=True).sort_values(by="AnnRR", ascending=False)

    def select_indicators(self, minimunThres):
        '''
        Select the indicators that used to take part in the complex indicators.
        :param minimunThres: The minimum threshold of the filter standard.
        :return: Filtered dataset.
        '''
        data = self.indicator_picker
        data = data[(data['corrInfo'] <= 0.9) & (data['AnnRR'] > min(minimunThres, np.mean(data['AnnRR'])))]
        self.indicator_picker = [i.split("_")[0] for i in list(data.index)]

    def complex_indicator_rawLS(self):
        '''
        To combine the indicator data together and suppliment the missing data.
        Filter the variables of the dataframe.
        :return: A filtered variable dataframe and the raw data of the filtered variables without missing value.
        '''
        data = []
        for s in self.cate:
            df = self.buy_sell_signal[s].set_index(["industry", "ym"])
            longCols = [i for i in df.columns.values if "_buy" in i]
            shortCols = [i for i in df.columns.values if "_sell" in i]
            dfCols = [s + i[:-4] for i in df.columns.values if "_buy" in i]
            df[shortCols] = df[shortCols].apply(lambda x: x * (-1), axis=1)
            for e, c in enumerate(dfCols):
                df[c] = df.apply(lambda x: x[longCols[e]] + x[shortCols[e]], axis=1)
                del df[longCols[e]]
                del df[shortCols[e]]
            data.append(df)
        data = pd.concat(data, axis=1, sort=False).reset_index()

        # Suppliment the nan data
        industry = list(set(list(data['industry'])))
        dataCols = data.columns.values
        newdata = []
        for i in industry:
            newsubdf = data.iloc[:, :2][data['industry']==i].reset_index(drop=True) # Get the "industry" and "ym" columns
            subdf = data[data['industry']==i]
            updateData = []
            predictCols = []
            cols = []
            tobedelete = 0
            trigger = True
            for c in dataCols:
                if c[:3]=="IPI" or c[:3]=="PPI":
                    cols.append(c)
                    line = list(subdf[c])
                    former = line[0]
                    newline = []
                    for l in line:
                        if np.isnan(l):
                            newline.append(former)
                            if np.isnan(former) and trigger: tobedelete += 1
                        else:
                            former = l
                            newline.append(l)
                            trigger = False
                    newline = newline[tobedelete:]
                    updateData.append(newline)
                else:
                    predictCols.append(c)
            updateData = pd.DataFrame(updateData).T
            updateData.columns = cols
            updateData.index = [i for i in range(tobedelete, len(newsubdf))]
            newsubdf = pd.merge(newsubdf, updateData, left_index=True, right_index=True)
            predDF = subdf[predictCols].iloc[tobedelete:, :]
            newsubdf = pd.merge(newsubdf, predDF, on=['industry', 'ym'], how='inner')
            newdata.append(newsubdf)

        newdata = pd.concat(newdata, sort=False).reset_index(drop=True)
        # Select indicators to take part in the complex indicator
        newdata = newdata[['industry', 'ym'] + self.indicator_picker]

        self.complex_rawdata = newdata

    def complex_indicator(self):
        '''
        TO form a dataframe of the complex indicators.
        :return: The dataframe that used to decide which industry to invest in.
        '''
        data = self.complex_rawdata
        data['complexIDX'] = data.iloc[:, 2:].apply(lambda x: sum(x), axis=1)
        data = data[["industry", "ym", "complexIDX"]].set_index(["industry", "ym"])

        self.complex_ind = data.unstack(level=-1)['complexIDX']

        write = ExcelWriter("complex_indicators.xlsx")
        self.complex_ind.to_excel(write, "data")
        write.save()

    def complex_ind_performance(self, thres, startDate, endDate, excelwriter=None):
        '''
        The calculation of the portfolio performance by using complex indicator.
        :param thres: The threshold used to judge long short signals. Ranges from (0, 0.5).
        :param startDate: The start date of the calculation. "yyyy-mm" format.
        :param endDate: The end date of the calculation. "yyyy-mm" format.
        :return: The performance of the Portfolio by using the complex indicator.
        '''
        dateCols = self.datetime_func(startDate, endDate)
        ind_close = self.close.set_index('industry')
        ind_close = ind_close[dateCols]
        data = self.complex_ind[dateCols]

        scaler = MinMaxScaler()
        cols = data.columns.values
        for c in cols:
            arr = np.array(data[c]).reshape(-1, 1)
            arr = scaler.fit_transform(arr)
            del data[c]
            data.insert(0, c, arr)
            # Decide which industry to invest and which to quit
            data[c] = data[c].apply(lambda x: -1 if x <= 0.5 + thres else x)
            data[c] = data[c].apply(lambda x: 1 if x >= 0.5 + thres else x)

        # # Portfolio value
        # 1. Trading Signal
        rows = list(data.index)
        dataSig = []
        for r in rows:
            line = list(data.loc[r])
            signal = []
            trading = False
            for l in line:
                if l == 1:
                    signal.append(1)
                    trading = True
                elif l == -1 and trading:
                    signal.append(1)
                    trading = False
                elif trading:
                    signal.append(1)
                else:
                    signal.append(0)
            dataSig.append(signal)
        dataSig = pd.DataFrame(dataSig, index=rows, columns=cols)

        # 2. Value
        PortfolioVal = []
        for r in rows:
            line = list(ind_close.loc[r])
            sig = list(dataSig.loc[r])
            val = []
            rawVal = 0
            for n in range(len(line)):
                signal, price = sig[n], line[n]
                if (signal == 1 and n==0) or (rawVal == 0 and signal == 1):
                    val.append(1)
                    rawVal = price
                elif signal == 1:
                    # Assume the raw investment in each industry is 1 unit.
                    val.append(price / rawVal)
                elif signal == 0 and n==0:
                    val.append(1)
                else:
                    val.append(val[-1])
            PortfolioVal.append(val)
        PortfolioVal = pd.DataFrame(PortfolioVal, columns=cols, index=rows).unstack().reset_index()
        PortfolioVal.columns = ['ym', 'industry', 'complex']

        write = ExcelWriter("PortfolioVal.xlsx")
        PortfolioVal.to_excel(write, "data")
        write.save()

        # # Portfolio performance
        rawCapital = len(PortfolioVal['industry'].drop_duplicates())
        PortfolioVal = PortfolioVal.groupby('ym').sum()

        PortfolioVal['complex_returnRate'] = PortfolioVal.apply(lambda x: (x - rawCapital) / rawCapital, axis=1)
        newdf = pd.DataFrame(PortfolioVal['complex_returnRate']).T
        newdfCols = newdf.columns.values

        res = self.performance_5_scores(newdf, PortfolioVal, ['complex'])
        print("complex indicators performance:\n", res[['AnnRR', 'SigRR', 'SharpR', 'WLR', 'MDD']])
        years = list(set([i.split("-")[0] for i in newdfCols]))
        months = list(set([i.split("-")[1] for i in newdfCols]))
        years.sort()
        months.sort()
        data = []
        for i in years:
            line = []
            for j in newdfCols:
                if j.split("-")[0]==i:
                    line.append(newdf[j].values[0])
            data.append(line)
        data = pd.DataFrame(data, columns=months, index=years)

        data.to_excel(excelwriter, startDate[:4])


    def run_dataClean(self, startDate, endDate):
        self.IPI_data_calcul()
        self.IPI_nan_deal()
        self.PPI_data_calcul()
        self.PPI_nan_deal()
        self.predict_nan_deal()

        self.weighted_calculation()

        self.industry_indicators()

        self.buysell_indicator(0.1)
        self.single_buysell_perf()

        for c in ['predict', 'financial']:
            data = self.portfolio[c]
            self.performance_calcul(data, c, True, "2016-01", "2020-09")

        self.check_correlation_vars()
        self.select_indicators(0.05)
        self.complex_indicator_rawLS()
        self.complex_indicator()

        write = ExcelWriter("final_result.xlsx")
        self.complex_ind_performance(0.1, startDate, endDate, write)
        write.save()