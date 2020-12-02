import pandas as pd
from pandas import ExcelWriter
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from base_classes import FileManagement
from random import randint
import os
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
        self.performance_LongShort = {} # Store the dataframe that long/short the industry
        self.indicatorRes = {}
        self.indicatorScore = {}

        self.PPI_ind = colnames.PPI_COLNAMES.value
        self.IPI_ind = colnames.IPI_COLNAMES.value
        self.end_date = "2020-10"
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
            "2010": 0.0096
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
        self.close = indClose

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
                if int(ym[:4]) == 2019 or (int(ym[:4]) == 2020 and int(ym[5:]) < 10): # Temporary
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
        for s in self.cate:
            data = self.industry_datasets[s]
            for c in data.columns.values[2:]:
                avg = data[["industry", c]].groupby(by=["industry"]).mean().reset_index().rename(columns={c: "avg"})
                data = pd.merge(data, avg, on="industry", how="left")
                # The buy / sell signal occurs when the indicator is larger than the enlarge the average / reduce the average
                # Since the A stock market performs bad after the indicators shows a good signal, we upside down the long short accordingly
                data[c + "_buy"] = data.apply(lambda x: (x[c] < x["avg"] * (1 - thres)) * 1, axis=1)
                data[c + "_sell"] = data.apply(lambda x: (x[c] > x["avg"] * (1 + thres)) * 1, axis=1)
                del data['avg']
                del data[c]
            self.buy_sell_signal[s] = data

    def single_buysell_perf(self, writeExcel=False):
        close = self.close
        col_dict = {
            "IPI": TTM_indicator_calculation.ipi_cols.value,
            "PPI": TTM_indicator_calculation.ppi_cols.value,
            "predict": TTM_indicator_calculation.predict_cols.value
        }

        write = ExcelWriter("result.xlsx")
        idx_ind_score = []
        for s in self.cate:
            data = self.buy_sell_signal[s]

            industry = list(data['industry'])
            industry = sorted(set(industry), key=industry.index)
            col = col_dict[s]

            # To calculate each indicators one by one
            indicatorRes = {}
            for c in col:
                indRes = {}
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
                    for e, t in enumerate(tmp):
                        if buyrec==0 and sellrec==0 and t > 0:
                            tradeSig.append(1)
                            buyrec = 1
                            pos = e
                        elif buyrec==1 and sellrec==0 and t < 0:
                            tradeSig.append(-1)
                            buyrec = 0
                            sellrec = 1
                        elif buyrec==0 and sellrec==1 and t > 0:
                            tradeSig.append(1)
                            buyrec = 1
                            sellrec = 0
                            pos = e
                        else:
                            tradeSig.append(0)
                    if sum(tradeSig) != 0: tradeSig[pos] = 0
                    subdf.insert(4, c, tradeSig)
                    subclose = close[close["industry"]==i].T.reset_index().iloc[1:, :]
                    subclose.columns = ["ym", c + "_close"]
                    subdf = pd.merge(subdf[["industry", "ym", c]], subclose, on="ym", how="inner")
                    # Long give money out, Short get money back
                    subdf[c+'_res'] = subdf.apply(lambda x: round(x[c] * x[c + "_close"] * (-1), 2), axis=1)

                    tradeRec = subdf[c+'_res']
                    datelist = subdf['ym']
                    closeP = subdf[c+'_close']
                    rf = np.array([self.libor[i] for i in self.libor]).mean()
                    indRes[i] = self.performance_calcul(tradeRec, datelist, closeP, rf)
                    indScore.append(subdf[['industry', 'ym', c]])

                indRes = pd.DataFrame(indRes)
                # Aggregate all the ratios and average them to get a ratio table represents the performance of each indicator.
                indRes['avg'] = indRes.apply(lambda x: np.nanmean(x), axis=1)
                indicatorRes[c] = list(indRes['avg'])

                indScore = pd.concat(indScore).set_index(['industry', 'ym'])
                idx_ind_score.append(indScore)

            indicatorRes = pd.DataFrame(indicatorRes).T.reset_index()
            indicatorRes.columns = ['indicator', 'AnnualProfRate', 'AnnualVol', 'SharpeRatio', 'MaxDrawdown', 'WinLossRatio']
            indicatorRes.to_excel(write, s, index=False)
        if writeExcel: write.save()

        idx_ind_score = pd.concat(idx_ind_score, axis=1)
        idx_ind_score = idx_ind_score.reset_index()
        write = ExcelWriter("industry_score.xlsx")
        for dt in set(list(idx_ind_score['ym'])):
            idx_ind_score.to_excel(write, dt, index=False)
        write.save()

    def performance_calcul(self, tradeRec, datelist, close, rf):
        '''
        To calculate the performance of the portfolio.
        :param tradeRec: The trading record of the transactions, should be in array, pd.Series, or list format.
                         Just shows the amount of money of the long and short, long position is negative, short position is positive.
        :param datelist: The datelist of the transaction date. It is full amount data, including all of the transaction date.
        :param close: The close price of the transactions. Used to calculate the max drawdown.
        :param rf: The risk free rate of the market, recorded by year, by using LIBOR.
        :return:
        '''
        trade = np.array(tradeRec) != 0
        tradeDate = np.array(datelist)[trade]
        tradeRec = np.array(tradeRec)[trade]
        i = 0
        Prof = {"annualProf": [], "win": 0, "trade": 0}
        while i < len(tradeDate):
            longDate = datetime.strptime(tradeDate[i], "%Y-%m")
            shortDate = datetime.strptime(tradeDate[i + 1], "%Y-%m")
            DateDelta = shortDate - longDate
            annualProf = (tradeRec[i + 1] + tradeRec[i]) * 365.0 / (np.abs(tradeRec[i]) * DateDelta.days)
            Prof['annualProf'].append(annualProf)
            if tradeRec[i + 1] > np.abs(tradeRec[i]): Prof['win'] += 1
            Prof['trade'] += 1
            i += 2

        if len(Prof['annualProf'])==0: return [np.NaN] * 5
        # 1. Annual Rate of Return
        annualProfMean = round(sum(Prof['annualProf']) / len(Prof['annualProf']), 3)
        # 2. Annual Rate of Volatility
        profU = sum(np.square(np.array(Prof['annualProf']) - annualProfMean)) / (len(Prof['annualProf']) - 1) if len(Prof['annualProf']) > 1 else np.NaN
        yearVol = round(np.sqrt(profU), 3) if len(Prof['annualProf']) > 1 else np.NaN
        # 3. Sharpe Ratio
        SharpeR = (annualProfMean - rf) / yearVol if len(Prof['annualProf']) > 1 else np.NaN
        # 4. Win Loss Ratio
        wlr = round(Prof['win'] * 1.0 / Prof['trade'], 3)
        # 5. Max Drawdown
        rec = False
        closeTradeP = []
        ind = 1
        closeWin = []
        for e, i in enumerate(trade):
            if i: ind += 1
            if (i or rec) and ind % 2 == 0:
                closeTradeP.append(close[e])
                rec = True
            elif i and rec and ind % 2 == 1:
                closeTradeP.append(close[e])
                rec = False
                closeWin.append(closeTradeP)
                closeTradeP = []
        maxDB = []
        for i in closeWin:
            data = pd.Series(i)
            rollMax = data.cummax()
            drawdown = np.array(data) / np.array(rollMax) - 1
            maxDB.append(round(min(drawdown), 3))
        res = [annualProfMean, yearVol, SharpeR, min(maxDB), wlr]
        return res

    def run_dataClean(self):
        self.IPI_data_calcul()
        self.IPI_nan_deal()
        self.PPI_data_calcul()
        self.PPI_nan_deal()
        self.predict_nan_deal()

        self.weighted_calculation()

        self.industry_indicators()

        self.buysell_indicator(0.1)
        self.single_buysell_perf(True)

