from base_classes import FileManagement, BaseData
from PPI_data import PeriodData
from IPI_data import ImmediateData
from data_cleaning import DataCleaningPro
from predict_data import PredictData
import os


if __name__=="__main__":
    file_path = "../DATA/"
    files = os.listdir(file_path)

    # # Initialize base class
    fm: FileManagement = FileManagement(file_path)
    fm.initialized_data("basicInfo.xlsx", "baseCompInfo", "b")
    # Initialize base company infomation and needed stocks (companies)
    bd = BaseData(fm)
    bd.select_data("baseCompInfo")


    # Periodic Performance Indicators Data
    print("PPI Data")
    # With single feature
    fm.initialized_data("operationAbility.xlsx", "OperationAbility", "p")
    fm.initialized_data("assetStructure.xlsx", "AssetStructure", "p")
    # With duplicated features
    fm.initialized_data("profitability.xlsx", "Profitability", "p")
    fm.initialized_data("growthAbility.xlsx", "growthAbility", "p")

    ppi = PeriodData(fm)
    # With single feature
    ppi.singleVar_data_transfer("OperationAbility", "baseCompInfo")
    ppi.singleVar_data_transfer("AssetStructure", "baseCompInfo")
    # With duplicated features
    ppi.duplicatedVar_data_transfer("Profitability", "baseCompInfo")
    ppi.duplicatedVar_data_transfer("growthAbility", "baseCompInfo")

    ppi.run_ipi()


    # Immediate Performance Indicators Data
    print("IPI Data")
    ipi = ImmediateData(fm)
    target_folders = [i for i in files if "_growth" in i]
    for t in target_folders:
        tar_file_path = file_path + t + "/"
        # print('tar_file_path:', tar_file_path)
        subfiles = os.listdir(tar_file_path)
        for s in subfiles:
            if ".xlsx" in s and "~$" not in s:
                # print('    ', s)
                fm.initialized_data(t + "/" + s, s.split('.')[0], "p")
                fm.store_yoy_growth_dataInfo(t + "/" + s, s.split('.')[0])
    ipi.run_ipi("baseCompInfo")

    # Prediction data
    print("Predict Data")
    pdd = PredictData(fm)
    tar_file_path = file_path + 'yuce/std/'
    subfiles = os.listdir(tar_file_path)
    predict_categories = ["EPS", "profitGrowth", "netProfit", "netProfitComp", "ROE", "BPS", "close"]
    for i in predict_categories:
        for j in subfiles:
            if j.split("_")[0]==i or j.split(".")[0]==i:
                fm.initialized_data('yuce/std/' + j, j.split(".xlsx")[0], "p", True)
                fm.store_predict_dataInfo(j.split(".xlsx")[0], i)
    pdd.run_predict()

    fm.initialized_data("Industry/ROE.xlsx", "ROE", base_period="p", multi=True)
    fm.OREdf() # Read the ROE data
    print("Data clean process")
    dcp = DataCleaningPro(fm)
    dcp.run_dataClean('2016-01', '2020-09')
