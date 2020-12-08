import pandas as pd
import numpy as np
from datetime import datetime
from base_classes import FileManagement


class ImmediateData(FileManagement):
    def __init__(self, fmClass):
        super(FileManagement, self).__init__()
        self.dataframes = fmClass.dataframes
        self.columnName = fmClass.columnName
        self.dataframes_Used = fmClass.dataframes_Used
        self.IPI_data_cate = fmClass.IPI_data_cate

    def two_kinds_data(self, dataframeNameList, storeName, baseDfName):
        # Rearrange the dataframe sequence - for the convenience of building the dataframe
        re_arrange_dfname = [""] * 2
        for i in dataframeNameList:
            if "yoy_" in i: re_arrange_dfname[0] = i
            if "_yoy" in i: re_arrange_dfname[1] = i

        # Deal with the dataframe
        new_df = pd.DataFrame()
        for d in re_arrange_dfname:
            # Columns
            Columns = self.columnName[d]
            category = list(set([i.split("_")[0] for i in Columns[2:]]))

            # Dataframe
            df = self.dataframes[d]
            df = df.set_index(keys=['证券代码', '证券简称'])
            df = pd.DataFrame(df.stack(dropna=False))
            df = df.reset_index()
            df = df.rename(columns={'证券代码':'stockCode', '证券简称':'stockName', 0: category[0]})
            df['year'] = df['level_2'].apply(lambda x: x.split("_")[1][:4])
            df['reportPeriod'] = df['level_2'].apply(lambda x: x.split("_")[1][4:])
            del df['level_2']
            df['stockCode'] = df['stockCode'].apply(lambda x: x.split(".")[0])
            df_cols = list(df.columns.values)
            df_cols.remove(category[0])
            df_cols.append(category[0])
            df = df[df_cols]
            if len(new_df)>0:
                new_df = pd.merge(new_df, df, on=['stockCode', 'stockName', 'year', 'reportPeriod'], how='left')
            else:
                new_df = df
        new_df = pd.merge(new_df, self.dataframes_Used[baseDfName], on=['stockCode', 'stockName'], how='right')
        self.dataframes_Used[storeName] = new_df

    def three_kinds_data(self, dataframeNameList, storeName, baseDfName):
        # Rearrange the dataframe sequence - for the convenience of building the dataframe
        re_arrange_dfname = [""] * 2
        for i in dataframeNameList:
            if "yoy_" in i: re_arrange_dfname[0] = i
            if "_yoy" in i: re_arrange_dfname[1] = i

        # Deal with the dataframe
        new_df = pd.DataFrame()
        for d in re_arrange_dfname:
            # Columns
            Columns = self.columnName[d]
            category = list(set([i.split("_")[0] for i in Columns[2:]]))

            # Dataframe
            df = self.dataframes[d]
            df = df.set_index(keys=['证券代码', '证券简称'])
            if len(category)==1:
                new_df = pd.DataFrame(df.stack(dropna=False))
                new_df = new_df.reset_index()
                new_df = new_df.rename(columns={'证券代码':'stockCode', '证券简称':'stockName', 0: category[0]})
                new_df['year'] = new_df['level_2'].apply(lambda x: x.split("_")[1][:4])
                new_df['reportPeriod'] = new_df['level_2'].apply(lambda x: x.split("_")[1][4:])
                del new_df['level_2']
                new_df['stockCode'] = new_df['stockCode'].apply(lambda x: x.split(".")[0])
                df_cols = list(new_df.columns.values)
                df_cols.remove(category[0])
                df_cols.append(category[0])
                new_df = new_df[df_cols]
            else:
                for c in category:
                    c_list = [k for k in df.columns.values if c in k]
                    sub_df = pd.DataFrame(df[c_list].stack(dropna=False))
                    sub_df = sub_df.reset_index()
                    sub_df = sub_df.rename(columns={'证券代码':'stockCode', '证券简称':'stockName', 0: c})
                    sub_df['year'] = sub_df['level_2'].apply(lambda x: x.split("_")[1][:4])
                    sub_df['reportPeriod'] = sub_df['level_2'].apply(lambda x: x.split("_")[1][4:])
                    del sub_df['level_2']
                    sub_df['stockCode'] = sub_df['stockCode'].apply(lambda x: x.split(".")[0])
                    new_df = pd.merge(new_df, sub_df, on=['stockCode', 'stockName', 'year', 'reportPeriod'], how="left")
        new_df = pd.merge(new_df, self.dataframes_Used[baseDfName], on=['stockCode', 'stockName'], how='right')
        self.dataframes_Used[storeName] = new_df

    def label_NaN_value(self):
        for i in self.IPI_data_cate:
            data = self.dataframes_Used[i]
            colnames = data.columns.values
            colnames = colnames[4:-1]
            for c in colnames:
                data[c + '_checkNaN'] = data[c].apply(lambda x: np.isnan(x) * 1.0)
            self.dataframes_Used[i] = data

    def combine_all_df(self):
        new_df = pd.DataFrame()
        for e, i in enumerate(self.IPI_data_cate):
            if e==0:
                baseDf = self.dataframes_Used['baseCompInfo']
                new_df = pd.merge(baseDf, self.dataframes_Used[i], on=['stockCode', 'industry', 'stockName'], how='left')
            else:
                new_df = pd.merge(new_df, self.dataframes_Used[i], on=['stockCode', 'industry', 'stockName', 'year', 'reportPeriod'], how='left')
        new_df = new_df.drop(new_df[(new_df['year'] == '2020') & (new_df['reportPeriod'] == '三季')].index, axis=0)
        new_df = new_df.drop(new_df[(new_df['year'] == '2020') & (new_df['reportPeriod'] == '年报')].index, axis=0)
        new_df = new_df.reset_index(drop=True)
        self.dataframes_Used['IPI'] = new_df # IPI stands for Immediate Performance Indicators

    def run_ipi(self, baseDfName):
        for i in self.IPI_data_cate:
            if i=="np":
                self.three_kinds_data(self.IPI_data_cate[i], i, baseDfName)
            else:
                self.two_kinds_data(self.IPI_data_cate[i], i, baseDfName)
        self.label_NaN_value()
        self.combine_all_df()
