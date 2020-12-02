import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class FileManagement:
    def __init__(self, file_path):
        self.dataframes = {}
        self.columnName = {}
        self.dataframes_Used = {}
        self.file_names = []
        self.file_path = file_path
        self.IPI_data_cate = {}
        self.predict_data_cate = {}
        self.ROEdata = {}
        self.close = pd.DataFrame()
        self.yearPeriod = ["20" + "{:0>2}".format(i) for i in range(8, 21)]


    def read_files(self, file_name, dataframeName, multi=False):
        entire_path = self.file_path + file_name
        if not multi:
            data = pd.read_excel(entire_path, sheet_name="data")
            data = data[:len(data) - 2]
        else:
            data = []
            for y in self.yearPeriod:
                subdata = pd.read_excel(entire_path, sheet_name=y)
                subdata = subdata[:len(subdata) - 2]
                colname = list(subdata.columns.values)
                newcolname = self.excel_int_date(colname[2:])
                newcolname.insert(0, colname[0])
                newcolname.insert(1, colname[1])
                subdata.columns = newcolname
                subdata['stockCode'] = subdata['stockCode'].apply(lambda x: x.split(".")[0])
                data.append(subdata)
            if dataframeName=="close":
                base = data[0]
                base['stockCode'] = base['stockCode'].apply(lambda x: x.split(".")[0])
                for df in data[1:]:
                    df['stockCode'] = base['stockCode'].apply(lambda x: x.split(".")[0])
                    base = pd.merge(base, df, on=['stockCode', 'stockName'], how='inner')
                self.close = base
                self.close_data_manage()
        self.dataframes[dataframeName] = data
        self.file_names.append(dataframeName)

    def close_data_manage(self):
        base = self.dataframes['baseCompInfo'][["证券代码", "证券简称", "行业名称"]]
        base.columns = ["stockCode", "stockName", "industry"]
        close = self.close
        close = pd.merge(base, close, on=['stockCode', 'stockName'], how='inner')
        closecol = list(close.columns.values)
        closecol = [closecol[:3] + [str(i.year) + "-" + "{:0>2}".format(i.month) for i in closecol[3:]]][0]
        close.columns = closecol
        self.close = close

    def column_name_baseDF(self, dataframeName):
        dfColumn = self.dataframes[dataframeName].columns.values
        dfColumn = [i.split('\r')[0] for i in dfColumn]
        self.columnName[dataframeName] = dfColumn

    def column_name_PeriodDF(self, dataframeName):
        dfColumn = self.dataframes[dataframeName].columns.values
        dfColumn = [i.split(' ') for i in dfColumn]
        new_dfColumn = []
        for i in dfColumn:
            if len(i) > 1:
                tmp = []
                for j in i:
                    tmp.append(j.split("\r")[0])
                new_dfColumn.append("_".join(tmp))
            else:
                new_dfColumn.append(i[0].strip())
        self.columnName[dataframeName] = new_dfColumn

    def column_name_YearDF(self, dataframeName):
        pass

    def rename_DF(self, dataframeName, base_period):
        '''
        Rename the dataframe columns name accordingly.
        :param dataframeName: The target dataframe name
        :param base_period: "base" for "b", "period" for "p"
        :return: A dataframe with new name
        '''
        if base_period=="p":
            self.column_name_PeriodDF(dataframeName)
            self.dataframes[dataframeName].columns = self.columnName[dataframeName]
        elif base_period=="b":
            self.column_name_baseDF(dataframeName)
            self.dataframes[dataframeName].columns = self.columnName[dataframeName]
        elif base_period=="y":
            self.column_name_YearDF(dataframeName)
            self.dataframes[dataframeName].columns = self.columnName[dataframeName]
        else:
            print("Error, please check your input")

    def initialized_data(self, file_name, dataframeName, base_period="p", multi=False):
        self.read_files(file_name, dataframeName, multi)
        if multi: return
        self.rename_DF(dataframeName, base_period)

    def store_yoy_growth_dataInfo(self, file_name, dataframeName):
        keyName = file_name.split('_')[0].split('/')[-1]
        if keyName not in self.IPI_data_cate:
            self.IPI_data_cate[keyName] = [dataframeName]
        else:
            self.IPI_data_cate[keyName].append(dataframeName)

    def store_predict_dataInfo(self, dataframeName, cate):
        if cate not in self.predict_data_cate:
            self.predict_data_cate[cate] = [dataframeName]
        else:
            self.predict_data_cate[cate].append(dataframeName)

    def excel_int_date(self, datelist):
        '''
        Alter the input excel datetime (int format) into the date format "yyyy-mm-dd".
        :param datelist: The list that contains the excel int format data.
        :return: The right date format data, in string style. ["yyyy-mm-dd"]
        '''
        new_list = []
        for dt in datelist:
            if type(dt)==datetime:
                return datelist
            dt = timedelta(days=int(dt))
            dateStr = datetime.strftime(datetime.strptime("1899-12-30", "%Y-%m-%d") + dt, "%Y-%m-%d")
            new_list.append(dateStr)
        return new_list

    def OREdf(self):
        ORE = self.dataframes['ROE']
        for i in ORE:
            subROE = i
            cols = subROE.columns.values
            year = str(cols[-1].year)
            self.ROEdata[year] = subROE

    def datetime_func(self, s, e):
        '''
        Iterate a datetime series, date delta is month. Format: yyyy-mm
        :param s: start date, format: yyyy-mm
        :param e: end date, format: yyyy-mm
        :return: A list of datetime, string style.
        '''
        res = [s]
        s = datetime.strptime(s, "%Y-%m")
        e = datetime.strptime(e, "%Y-%m")
        while s < e:
            if s.month in [1, 3, 5, 7, 8, 10, 12]:
                s = s + timedelta(days=31)
            elif s.month in [4, 6, 9, 11]:
                s = s + timedelta(days=30)
            else:
                if s.year % 4 == 0 and s.year % 400 != 0:
                    s = s + timedelta(days=29)
                else:
                    s = s + timedelta(days=28)
            res.append(s.strftime("%Y-%m"))
        return res


class BaseData(FileManagement):
    def __init__(self, fmClass):
        super(FileManagement, self).__init__()
        self.dataframes = fmClass.dataframes
        self.columnName = fmClass.columnName
        self.dataframes_Used = fmClass.dataframes_Used

    def explore_data(self, dataframeName):
        colname = self.columnName[dataframeName]
        data = self.dataframes[dataframeName]
        data = data[data['股票种类'] == 'A股'].reset_index(drop=True)
        data['成立日期'] = data['成立日期'][data['成立日期']!=np.NaN].apply(lambda i: datetime.date(datetime.strptime(str(int(i)), "%Y%m%d")))

        # To view the distribution of each category - decide which stocks to be picked
        for i in ['交易所', '行业名称', '是否属于风险警示板', '公司属性', '企业规模', '所属行政区划', '城市']:
            df = pd.merge(data[['证券代码', '证券简称']], pd.DataFrame(data[i]),
                          left_index=True, right_index=True)
            df['counter'] = 1
            pivot = pd.pivot_table(df, values='counter', index=i, aggfunc=np.sum)
            # print('pivot:\n', pivot)
        return data

    def select_data(self, dataframeName, scale=10):
        data = self.explore_data(dataframeName)
        # To pick out 10 stocks from each indeustry
        # 1. Seperate important index combination
        df1 = data[data['是否属于重要指数成份']=='是']
        df2 = data[data['是否属于重要指数成份']=='否']
        # 2. Rank according to capital, group by industry and company types
        df = []
        for i in set(list(data['行业名称'])):
            for j in set(list(data['公司属性'])):
                df.append(df2[(df2['行业名称']==i) & (df2['公司属性']==j)].sort_values(by='注册资本', ascending=False).iloc[:scale])
        df.append(df1)
        df = pd.concat(df).reset_index(drop=True)
        df['counter'] = 1
        df_base = df[['证券代码','行业名称', '证券简称']].rename(columns={'证券代码':'stockCode', '行业名称':'industry', '证券简称':'stockName'})
        pivot = pd.pivot_table(df[['行业名称', 'counter']], index='行业名称', values='counter', aggfunc=np.sum).T
        self.dataframes_Used[dataframeName + "_whole"] = df
        self.dataframes_Used[dataframeName] = df_base

