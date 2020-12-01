from enum import Enum

class colnames(Enum):
    PPI_COLNAMES = {
        "资产负债率": "DebtToAsset", "资产负债率_checkNaN": "DebtToAsset_checkNaN",
        "总资产周转率(TTM)": "AssetSturn", "总资产周转率(TTM)_checkNaN": "AssetSturn_checkNaN",
        "销售毛利率(TTM)": "GrossProfitMargin", "销售毛利率(TTM)_checkNaN": "GrossProfitMargin_checkNaN",
        "净资产收益率(TTM)": "ROE", "净资产收益率(TTM)_checkNaN": "ROE_checkNaN",
        "总资产报酬率(TTM)": "ROA", "总资产报酬率(TTM)_checkNaN": "ROA_checkNaN",
        "销售净利率(TTM)": "NetProfitMargin", "销售净利率(TTM)_checkNaN": "NetProfitMargin_checkNaN",
        "归属母公司股东的净利润(同比增长率)": "GrowthNPtoShareholder", "归属母公司股东的净利润(同比增长率)_checkNaN": "GrowthNPtoShareholder_checkNaN",
        "利润总额(同比增长率)": "totalProfit", "利润总额(同比增长率)_checkNaN": "totalProfit_checkNaN",
        "营业收入(同比增长率)": "OperRev", "营业收入(同比增长率)_checkNaN": "OperRev_checkNaN",
        "净利润(同比增长率)": "NetProfit", "净利润(同比增长率)_checkNaN": "NetProfit_checkNaN",
        "经营活动产生的现金流量净额(同比增长率)": "GrowthOperNetCF", "经营活动产生的现金流量净额(同比增长率)_checkNaN": "GrowthOperNetCF_checkNaN",
        "EBITDA/营业总收入": "EBITDA", "EBITDA/营业总收入_checkNaN": "EBITDA_checkNaN",
        "归属母公司股东的净利润-扣除非经常损益(同比增长率)": "NPtoShareholder", "归属母公司股东的净利润-扣除非经常损益(同比增长率)_checkNaN": "NPtoShareholder_checkNaN",
    }

    IPI_COLNAMES = {
        "归属母公司股东的净利润": "GrowthNPtoShareholder", "业绩快报.同比增长率:归属母公司股东的净利润": "Express_GrowthNPtoShareholder",
        "预告净利润同比增长上限": "ADnotice_Up_GrowthNPtoShareholder", "预告净利润同比增长下限": "ADnotice_Down_GrowthNPtoShareholder",
        "归属母公司股东的净利润_checkNaN": "GrowthNPtoShareholder_checkNaN", "业绩快报.同比增长率:归属母公司股东的净利润_checkNaN": "Express_GrowthNPtoShareholder_checkNaN",
        "预告净利润同比增长上限_checkNaN": "ADnotice_Up_GrowthNPtoShareholder_checkNaN", "预告净利润同比增长下限_checkNaN": "ADnotice_Down_GrowthNPtoShareholder_checkNaN",
        "每股收益EPS-基本": "EPS", "业绩快报.同比增长率:基本每股收益": "Express_EPS",
        "每股收益EPS-基本_checkNaN": "EPS_checkNaN", "业绩快报.同比增长率:基本每股收益_checkNaN": "Express_EPS_checkNaN",
        "利润总额": "totalProfit", "业绩快报.利润总额": "Express_totalProfit",
        "利润总额_checkNaN": "totalProfit_checkNaN", "业绩快报.利润总额_checkNaN": "Express_totalProfit_checkNaN",
        "营业利润": "OperProfit", "业绩快报.同比增长率:营业利润": "Express_OperProfit",
        "营业利润_checkNaN": "OperProfit_checkNaN", "业绩快报.同比增长率:营业利润_checkNaN": "Express_OperProfit_checkNaN",
        "净资产收益率ROE": "ROE", "业绩快报.同比增减:加权平均净资产收益率": "Express_ROE",
        "净资产收益率ROE_checkNaN": "ROE_checkNaN", "业绩快报.同比增减:加权平均净资产收益率_checkNaN": "Express_ROE_checkNaN",
        "营业收入": "OperRev", "业绩快报.同比增长率:营业收入": "Express_OperRev",
        "营业收入_checkNaN": "OperRev_checkNaN", "业绩快报.同比增长率:营业收入_checkNaN": "Express_OperRev_checkNaN"
    }

class TTM_indicator_calculation(Enum):
    ppi_cal_yoy = ["OperRev", "GrowthNPtoShareholder", "NetProfit", "totalProfit", "DebtToAsset"] # Need to calculated the YoY growth rate
    ppi_cal_cr = ["NetProfitMargin", "GrossProfitMargin", "ROE", "ROA", "AssetSturn"] # Need to calculate the Chain Ratio (环比)
    ppi_cols = ["OperRev", "GrowthNPtoShareholder", "NetProfit", "totalProfit", "DebtToAsset",
                "NetProfitMargin", "GrossProfitMargin", "ROE", "ROA", "AssetSturn"]

    ipi_cols = ["GrowthNPtoShareholder", "EPS", "totalProfit", "OperProfit", "ROE", "OperRev"]

    predict_cols = ['EPS', 'profitGrowth', 'netProfit', 'PE', 'PEG', 'netProfitComp', 'ROE', 'BPS', 'PB']

    weight_cols = ['stockCode', 'industry', 'stockName']
