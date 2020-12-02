import pandas as pd
import numpy as np
from base_classes import FileManagement


class PeriodData(FileManagement):
    def __init__(self, fmClass):
        super(FileManagement, self).__init__()
        self.dataframes = fmClass.dataframes
        self.columnName = fmClass.columnName
        self.dataframes_Used = fmClass.dataframes_Used

    def singleVar_data_transfer(self, dataframeName, baseDfName):
        baseData = self.dataframes_Used[baseDfName][['stockCode', 'industry', 'stockName']]
        data = self.dataframes[dataframeName]
        data['证券代码'] = data['证券代码'].apply(lambda x: x.split('.')[0])
        self.dataframes[dataframeName] = data

        # To deal with the column name
        datacols = self.columnName[dataframeName]
        category = [j for j in set([i.split('_')[0] for i in datacols[2:]])][0]

        # Transform the dataframe
        data = data.set_index(keys=['证券代码', '证券简称'])
        new_df = pd.DataFrame(data.stack(dropna=False))
        new_df.columns = [category]
        new_df = new_df.reset_index()
        new_df['year'] = new_df['level_2'].apply(lambda x: x.split("_")[1][:4])
        new_df['reportPeriod'] = new_df['level_2'].apply(lambda x: x.split("_")[1][4:])
        del new_df['level_2']
        new_df = new_df.rename(columns={'证券代码':'stockCode', '证券简称':'stockName'})
        new_df = pd.merge(new_df, baseData, on=['stockCode', 'stockName'], how='right')
        new_df_cols = list(new_df.columns.values)
        new_df_cols.remove(category)
        new_df_cols.append(category)
        new_df = new_df[new_df_cols]
        # Check the null value
        new_df[category + '_checkNaN'] = new_df[category].apply(lambda x: np.isnan(x) * 1.0)
        self.dataframes_Used[dataframeName] = new_df

    def duplicatedVar_data_transfer(self, dataframeName, baseDfName):
        baseData = self.dataframes_Used[baseDfName][['stockCode', 'industry', 'stockName']]
        data = self.dataframes[dataframeName]
        data['证券代码'] = data['证券代码'].apply(lambda x: x.split('.')[0])
        self.dataframes[dataframeName] = data

        # To deal with the column name
        datacols = self.columnName[dataframeName]
        category = [j for j in set([i.split('_')[0] for i in datacols[2:]])]
        new_df = data[['证券代码', '证券简称']]
        data = data.set_index(keys=['证券代码', '证券简称'])
        for e, c in enumerate(category):
            tarcol = [i for i in data.columns.values if c==i.split("_")[0]]
            part_data = data[tarcol]
            df = pd.DataFrame(part_data.stack(dropna=False))
            df = df.reset_index()
            df['year'] = df['level_2'].apply(lambda x: x.split("_")[1][:4])
            df['reportPeriod'] = df['level_2'].apply(lambda x: x.split("_")[1][4:])
            del df['level_2']
            df = df.rename(columns={0:c})
            dfcol = list(df.columns.values)
            dfcol.remove(c)
            dfcol.append(c)
            df = df[dfcol]
            if len(new_df.columns.values)==2:
                new_df = pd.merge(new_df, df, on=['证券代码', '证券简称'], how='right')
            else:
                new_df = pd.merge(new_df, df, on=['证券代码', '证券简称', 'year', 'reportPeriod'], how='right')
            # Check the null value
            new_df[c + '_checkNaN'] = new_df[c].apply(lambda x: np.isnan(x) * 1.0)
        new_df = new_df.rename(columns={'证券代码':'stockCode', '证券简称':'stockName'})
        new_df = pd.merge(new_df, baseData, on=['stockCode', 'stockName'], how='right')
        self.dataframes_Used[dataframeName] = new_df

    def combine_all_df(self):
        comb_df = pd.DataFrame()
        ind = True
        for i in self.dataframes_Used:
            if "baseCompInfo" not in i:
                if ind:
                    comb_df = self.dataframes_Used[i]
                    ind = False
                else:
                    comb_df = pd.merge(comb_df, self.dataframes_Used[i], on=['stockCode', 'industry', 'stockName', 'year', 'reportPeriod'], how='outer')
        comb_df = comb_df.drop(comb_df[(comb_df['year'] == '2020') & (comb_df['reportPeriod'] == '三季')].index, axis=0)
        comb_df = comb_df.drop(comb_df[(comb_df['year'] == '2020') & (comb_df['reportPeriod'] == '年报')].index, axis=0)
        comb_df = comb_df.reset_index(drop=True)
        self.dataframes_Used['PPI'] = comb_df # PPI stands for periodic performance indicators

    def run_ipi(self):
        self.combine_all_df()