import pandas as pd
import numpy as np
from datetime import datetime
from base_classes import FileManagement


class PredictData(FileManagement):
    def __init__(self, fmClass):
        super(FileManagement, self).__init__()
        self.dataframes = fmClass.dataframes
        self.columnName = fmClass.columnName
        self.dataframes_Used = fmClass.dataframes_Used
        self.predict_data_cate = fmClass.predict_data_cate
        self.temp_Data = {} # Store the raw data read from different excels
        self.raw_analyze_data = {}

    def dataframe_transform(self):
        '''
        To transform the datetime columns into a datetime column.
        :return: A dataframe with fewer columns.
        '''
        for i in self.predict_data_cate:
            for j in self.predict_data_cate[i]:
                data = self.dataframes[j]
                frames = []
                for d in data:
                    d = d.set_index(keys=['stockCode', 'stockName'])
                    newdata = pd.DataFrame(d.stack(dropna=False))
                    newdata = newdata.reset_index()
                    newdata = newdata.rename(columns={'level_2':"date", 0:j})
                    newdata['year'] = newdata['date'].apply(lambda x: str(x)[:4])
                    newdata['month'] = newdata['date'].apply(lambda x: str(x)[5:7])
                    del newdata['date']
                    frames.append(newdata)
                self.temp_Data[j] = pd.concat(frames).reset_index(drop=True)

    def data_deal(self):
        '''
        Calculate the recent 12 months financial data according to the FY1 & FY2 data.
        :return: The recent 12 months financial data.
        '''
        self.dataframe_transform() # Transform the data before dealing with them.
        for i in self.predict_data_cate:
            if len(self.predict_data_cate[i])==2:
                fy1_data = self.temp_Data[self.predict_data_cate[i][0]] if "FY1" in self.predict_data_cate[i][0] else self.temp_Data[self.predict_data_cate[i][1]]
                fy2_data = self.temp_Data[self.predict_data_cate[i][0]] if "FY2" in self.predict_data_cate[i][0] else self.temp_Data[self.predict_data_cate[i][1]]

                dt = []
                for y in set(list(fy1_data['year'])):
                    for m in set(list(fy1_data['month'])):
                        dt.append([y, m]) # Store the year and month seperately

                data = fy1_data[['stockCode', 'stockName', 'year', 'month']]
                frame = []
                for d in dt:
                    fy1 = fy1_data[["stockCode", "year", "month", i + "_FY1"]][(fy1_data['year']==d[0]) & (fy1_data['month']==d[1])]
                    fy2 = fy2_data[["stockCode", "year", "month", i + "_FY2"]][(fy2_data['year'] == d[0]) & (fy2_data['month'] == d[1])]
                    df = pd.merge(fy1, fy2, on=["stockCode", "year", "month"], how="inner")

                    newdata = []
                    for v in range(len(df)):
                        fy1_val = df[i + "_FY1"][v]
                        fy2_val = df[i + "_FY2"][v]
                        if np.isnan(fy2_val) and np.isnan(fy1_val):
                            newdata.append(np.NaN) # If they are both null, then just keep the null value for the new data.
                            continue
                        if np.isnan(fy1_val) and not np.isnan(fy2_val): fy1_val = 0 # If there is no prediction in fy1 but has prediction in fy2, then we regard the company is still growing, no judgement is sufficient.
                        if not np.isnan(fy1_val) and np.isnan(fy2_val):
                            newdata.append(fy1_val) # If there is a predition for fy1 but none for fy2, then we still use the same value for the fy2
                            continue
                        val = (1 - (int(d[1]) - 1) / 12) * fy1_val + (int(d[1]) - 1) / 12 * fy2_val
                        newdata.append(val)
                    df.insert(len(df.columns.values), i, newdata)
                    del df[i + "_FY1"]
                    del df[i + "_FY2"]
                    frame.append(df)
                frame = pd.concat(frame)
                data = pd.merge(data, frame, on=["stockCode", "year", "month"], how="inner")
            else:
                data = self.temp_Data[self.predict_data_cate[i][0]]
            self.raw_analyze_data[i] = data
        self.temp_Data.clear()

    def derive_indicators(self, target_var, newColname):
        data = self.raw_analyze_data[target_var]
        close = self.raw_analyze_data['close']
        data[newColname] = close['close'] / data[target_var]
        self.raw_analyze_data[target_var] = data

    def calculate_PE_G(self):
        data = self.raw_analyze_data['EPS']
        data['PEG'] = data['PE'] / self.raw_analyze_data['profitGrowth']['profitGrowth']

    def dataframe_combine(self):
        '''
        Combine all the sub-dataframes.
        :return: A total raw prediction dataframe
        '''
        data = self.raw_analyze_data['close'][['stockCode', 'stockName', 'year', 'month']]
        for i in self.raw_analyze_data:
            if 'close'==i: continue
            subdata = self.raw_analyze_data[i]
            data = pd.merge(data, subdata, on=['stockCode', 'stockName', 'year', 'month'], how='inner')
        data = data[['stockCode', 'stockName', 'year', 'month', 'EPS', 'profitGrowth', 'netProfit', 'PE', 'PEG',
                     'netProfitComp', 'ROE', 'BPS', 'PB']]

        cols = data.columns.values[4:]
        for c in cols:
            data[c + "_checkNaN"] = data[c].apply(lambda x: 1 if np.isnan(x) else 0)

        self.dataframes_Used['predict'] = data

    def derive_industry_indicator(self):
        pass


    def run_predict(self):
        self.data_deal()
        # Calculate certain indicators that does not provide by Wind
        self.derive_indicators('EPS', 'PE')
        self.calculate_PE_G()
        self.derive_indicators('BPS', 'PB')
        self.dataframe_combine()
