# -*- coding: utf-8 -*-
"""
1. Preprocessing
2. Feature engineering
"""

# -----------------------------------------------------------------------------
### Initial Setup
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os 
from collections import defaultdict
import pickle
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from swifter import swiftapply

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import rcParams
rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'

raw_data_path = "..."
new_data_path = "..."

seed = 12345

# -----------------------------------------------------------------------------
### load data
# -----------------------------------------------------------------------------
df_train = pd.read_csv(os.path.join(raw_data_path, "training-set.csv"))
df_test = pd.read_csv(os.path.join(raw_data_path, "testing-set.csv"))
df_claim = pd.read_csv(os.path.join(raw_data_path, "claim_0702.csv"))
df_policy = pd.read_csv(os.path.join(raw_data_path, "policy_0702.csv"))

df_insurance_type_info = pd.read_excel(os.path.join(raw_data_path, "VariableExplanationV2.xlsx"), "險種分類及自負額說明")

# -----------------------------------------------------------------------------
### Feature Engineering (without claims)
# -----------------------------------------------------------------------------

one_hot_transformer = dict()
tf_idf_transformer = dict()

df_policy["Insurance_Coverage_header_number"] = swiftapply(df_policy["Insurance_Coverage"], lambda x: x[:2])
df_policy["Insurance_Coverage_tail_symbol"] = swiftapply(df_policy["Insurance_Coverage"], lambda x: x[2])
df_policy["Insurance_Coverage_tail_symbol"] = "123" + df_policy["Insurance_Coverage_tail_symbol"]
df_policy["Sum_Insured_Amount"] = df_policy["Insured_Amount1"] + df_policy["Insured_Amount2"] + df_policy["Insured_Amount3"]
df_policy["fsex_Insured's_ID"] = df_policy["Insured's_ID"].apply(lambda x: mapping_dict_tmp[x])
df_policy["fmarriage_Insured's_ID"] = df_policy["Insured's_ID"].apply(lambda x: mapping_dict_tmp[x])
df_policy["ibirth_Insured's_ID"] = df_policy["Insured's_ID"].apply(lambda x: mapping_dict_tmp[x])

df_policy_number_identifier = pd.DataFrame([list(set(df_train["Policy_Number"])) + list(set(df_test["Policy_Number"])),
           [1, ]*len(list(set(df_train["Policy_Number"]))) + [0, ]*len(list(set(df_test["Policy_Number"])))])
df_policy_number_identifier = df_policy_number_identifier.transpose()
df_policy_number_identifier.columns = ["Policy_Number", "Train"]
df_policy = df_policy.merge(df_policy_number_identifier, on = ["Policy_Number"], how = "left")

df_train["Churn"] = 0
df_train["Churn"][df_train["Next_Premium"] == 0] = 1

# df_train
# -----------------------------------------------------------------------------

### Use df_policy

## Feature: Policy_Number出現數量
feature_col = "Policy_Number"
df_train = df_train.merge(df_policy.groupby([feature_col])["Insured's_ID"].agg(lambda x: len(x)).reset_index(),
               on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_" + feature_col, ]

## Feature: Policy_Number對應的Insured's_ID出現在不同的獨立Policy_Number次數
feature_col = "Insured's_ID"
mapping_dict_tmp = (df_policy.groupby([feature_col])["Policy_Number"].agg(lambda x: len(set(x))))
df_policy["Count_" + feature_col + "_in_Unique_Policy_Number"] = list(swiftapply(df_policy[feature_col], lambda x: mapping_dict_tmp[x]))

df_train = df_train.merge(df_policy.groupby(["Policy_Number"])["Count_" + feature_col + "_in_Unique_Policy_Number"].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_" + feature_col + "_in_Unique_Policy_Number", ]

## Feature: Policy_Number對應的Insured's_ID出現的次數
feature_col = "Insured's_ID"
mapping_dict_tmp = (df_policy.groupby([feature_col])["Policy_Number"].agg(lambda x: len(x)))
df_policy["Count_" + feature_col + "_in_Total_Policy_Number"] = list(swiftapply(df_policy[feature_col], lambda x: mapping_dict_tmp[x]))

df_train = df_train.merge(df_policy.groupby(["Policy_Number"])["Count_" + feature_col + "_in_Total_Policy_Number"].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_" + feature_col + "_in_Total_Policy_Number", ]

## Feature: 有無Prior_Policy_Number
feature_col = "Prior_Policy_Number"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(\
            lambda x: 1 if (~pd.Series(x).isnull()).sum() > 0 else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Binary_Not_NA_" + feature_col, ]

## Feature: Cancellation是否為"Y" (看起來和"有無Prior_Policy_Number"是一樣的，不過有一筆不一樣)
feature_col = "Cancellation"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(\
                          lambda x: 1 if list(x)[0] == "Y" else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Binary_Y_" + feature_col, ]

## Feature: Policy_Number對應的Insured's_ID有幾種不同的非NA Vehicle_identifier
feature_col = "Vehicle_identifier"
mapping_dict_tmp = df_policy.groupby(["Insured's_ID"])[feature_col].agg(lambda x:  len(set(x[~pd.Series(x).isnull()])))
df_policy["Count_Insured's_ID_in_Unique_Not_NA_" + feature_col] = list(swiftapply(df_policy["Insured's_ID"], lambda x: mapping_dict_tmp[x]))
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])["Count_Insured's_ID_in_Unique_Not_NA_" + feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_Insured's_ID_in_Unique_Not_NA_" + feature_col, ]

## Feature: Policy_Number對應的Insured's_ID有幾個非NA Vehicle_identifier
feature_col = "Vehicle_identifier"
mapping_dict_tmp = df_policy.groupby(["Insured's_ID"])[feature_col].agg(lambda x:  np.sum((~pd.Series(x).isnull())))
df_policy["Count_Insured's_ID_in_Not_NA_" + feature_col] = list(swiftapply(df_policy["Insured's_ID"], lambda x: mapping_dict_tmp[x]))
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])["Count_Insured's_ID_in_Not_NA_" + feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_Insured's_ID_in_Not_NA_" + feature_col, ]

## Feature: 有無Vehicle_identifier
feature_col = "Vehicle_identifier"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(\
            lambda x: 1 if (~pd.Series(x).isnull()).sum() > 0 else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Binary_Not_NA_" + feature_col, ]

## Feature: Manafactured_Year_and_Month值
feature_col = "Manafactured_Year_and_Month"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: Engine_Displacement_(Cubic_Centimeter)值
feature_col = "Engine_Displacement_(Cubic_Centimeter)"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: Insured's_ID值
feature_col = "Insured's_ID"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: Vehicle_Make_and_Model1值
feature_col = "Vehicle_Make_and_Model1"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: Vehicle_Make_and_Model2值
feature_col = "Vehicle_Make_and_Model2"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: Imported_or_Domestic_Car值
feature_col = "Imported_or_Domestic_Car"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: Imported_or_Domestic_Car值(One Hot Encoding)
feature_col = "Imported_or_Domestic_Car"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
le = LabelEncoder()
oe = OneHotEncoder(sparse=False, n_values=len(set(df_policy[feature_col])))
le.fit(df_policy[feature_col])
oe.fit(le.transform(df_train[feature_col]).reshape([-1, 1]))
one_hot_transformer[feature_col] = [le, oe]

df_train = pd.concat([df_train, pd.DataFrame(one_hot_transformer[feature_col][1].transform(one_hot_transformer[feature_col][0].transform(df_train[feature_col]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_" + str(x) + "_" + feature_col for x in one_hot_transformer[feature_col][0].classes_])], axis = 1)
df_train.drop([feature_col], axis = 1, inplace = True)

## Feature: Coding_of_Vehicle_Branding_&_Type值
feature_col = "Coding_of_Vehicle_Branding_&_Type"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: qpt值
feature_col = "qpt"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: 各個Main_Insurance_Coverage_Group數量
feature_col = "Main_Insurance_Coverage_Group"
mapping_df_tmp = df_policy.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_train = df_train.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-3]) + ["Count_" + x + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Main_Insurance_Coverage_Group比例
feature_col = "Main_Insurance_Coverage_Group"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_train = df_train.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-3]) + ["Percentage_" + x + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: Policy_Number對應的Main_Insurance_Coverage_Group有幾種不同的值
feature_col = "Main_Insurance_Coverage_Group"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: len(set(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_Unique_" + feature_col, ]

## Feature: 各個Insurance_Coverage數量
feature_col = "Insurance_Coverage"
mapping_df_tmp = df_policy.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_train = df_train.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-60]) + ["Count_" + x + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Insurance_Coverage比例
feature_col = "Insurance_Coverage"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_train = df_train.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-60]) + ["Percentage_" + x + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: Policy_Number對應的Insurance_Coverage有幾種不同的值
feature_col = "Insurance_Coverage"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: len(set(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_Unique_" + feature_col, ]

## Feature: Policy_Number對應的Main_Insurance_Coverage_Group有幾種不同的值對上Policy_Number出現次數的比率
df_train["Ratio_Count_Unique_Main_Insurance_Coverage_Group_to_Count_Policy_Number"] = df_train["Count_Unique_Main_Insurance_Coverage_Group"]/df_train["Count_Policy_Number"]

## Feature: Policy_Number對應的Insurance_Coverage有幾種不同的值對上Policy_Number出現次數的比率
df_train["Ratio_Count_Unique_Insurance_Coverage_to_Count_Policy_Number"] = df_train["Count_Unique_Insurance_Coverage"]/df_train["Count_Policy_Number"]

## Feature: Premium總和
feature_col = "Premium"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Sum_" + feature_col, ]

## Feature: Replacement_cost_of_insured_vehicle值
feature_col = "Replacement_cost_of_insured_vehicle"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: Replacement_cost_of_insured_vehicle值對上Premium總和的比率
df_train["Ratio_Value_Replacement_cost_of_insured_vehicle_to_Sum_Premium"] = df_train["Value_Replacement_cost_of_insured_vehicle"]/df_train["Sum_Premium"]

## Multiple_Products_with_TmNewa_(Yes_or_No?)值
feature_col = "Multiple_Products_with_TmNewa_(Yes_or_No?)"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Value_" + feature_col, ]

## Multiple_Products_with_TmNewa_(Yes_or_No?)值對上Policy_Number出現次數的比率
df_train["Ratio_Value_Multiple_Products_with_TmNewa_(Yes_or_No?)_to_Count_Policy_Number"] = df_train["Value_Multiple_Products_with_TmNewa_(Yes_or_No?)"]/df_train["Count_Policy_Number"]

## Feature: lia_class值
feature_col = "lia_class"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: plia_acc值
feature_col = "plia_acc"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: pdmg_acc值
feature_col = "pdmg_acc"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: fassured值(One Hot Encoding)
feature_col = "fassured"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
le = LabelEncoder()
oe = OneHotEncoder(sparse=False, n_values=len(set(df_policy[feature_col])))
le.fit(df_policy[feature_col])
oe.fit(le.transform(df_train[feature_col]).reshape([-1, 1]))
one_hot_transformer[feature_col] = [le, oe]

df_train = pd.concat([df_train, pd.DataFrame(one_hot_transformer[feature_col][1].transform(one_hot_transformer[feature_col][0].transform(df_train[feature_col]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_" + str(x) + "_" + feature_col for x in one_hot_transformer[feature_col][0].classes_])], axis = 1)
df_train.drop([feature_col], axis = 1, inplace = True)

## Feature: fassured值
feature_col = "fassured"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: Premium, Mean, Median, Max, Min, Std, 非0數量, 非0比例
feature_col = "Premium"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Mean_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Median_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Max_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Min_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Std_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: Insured_Amount1, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Insured_Amount1"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Sum_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Mean_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Median_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Max_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Min_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Std_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: Insured_Amount2, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Insured_Amount2"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Sum_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Mean_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Median_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Max_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Min_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Std_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: Insured_Amount3, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Insured_Amount3"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Sum_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Mean_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Median_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Max_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Min_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Std_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: Insurance_Coverage TF-IDF
feature_col = "Insurance_Coverage"
tf_idf_vectorizer = TfidfVectorizer(lowercase = False)
tf_idf_vectorizer.fit(list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).values))
tf_idf_transformer[feature_col] = tf_idf_vectorizer

df_tmp = pd.DataFrame(tf_idf_transformer[feature_col].transform(list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).values)).todense())
df_tmp["Policy_Number"] = list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).index)
df_tmp.columns = ["TF_IDF_" + str(x) + "@" + "_" + feature_col if len(x) == 2 else \
                 "TF_IDF_" + str(x) + "_" + feature_col for x in
                 [x[0] for x in sorted(tf_idf_transformer[feature_col].vocabulary_.items(), key = lambda x: x[1])]] + ["Policy_Number", ]

df_train = df_train.merge(df_tmp, on = ["Policy_Number"], how = "left")

## Feature: Insurance_Coverage_header_number TF-IDF
df_policy["Insurance_Coverage_header_number"] = swiftapply(df_policy["Insurance_Coverage"], lambda x: x[:2])
feature_col = "Insurance_Coverage_header_number"
tf_idf_vectorizer = TfidfVectorizer(lowercase = False)
tf_idf_vectorizer.fit(list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).values))
tf_idf_transformer[feature_col] = tf_idf_vectorizer

df_tmp = pd.DataFrame(tf_idf_transformer[feature_col].transform(list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).values)).todense())
df_tmp["Policy_Number"] = list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).index)
df_tmp.columns = ["TF_IDF_" + str(x) + "_" + feature_col for x in
                 [x[0] for x in sorted(tf_idf_transformer[feature_col].vocabulary_.items(), key = lambda x: x[1])]] + ["Policy_Number", ]

df_train = df_train.merge(df_tmp, on = ["Policy_Number"], how = "left")

## Feature: Insurance_Coverage_tail_symbol TF-IDF
feature_col = "Insurance_Coverage_tail_symbol"
tf_idf_vectorizer = TfidfVectorizer(lowercase = False)
tf_idf_vectorizer.fit(list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).values))
tf_idf_transformer[feature_col] = tf_idf_vectorizer

df_tmp = pd.DataFrame(tf_idf_transformer[feature_col].transform(list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).values)).todense())
df_tmp["Policy_Number"] = list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).index)
df_tmp.columns = ["TF_IDF_" + "@" + "_" + feature_col if len(x) == 3 else \
                 "TF_IDF_" + str(x)[-1] + "_" + feature_col for x in
                 [x[0] for x in sorted(tf_idf_transformer[feature_col].vocabulary_.items(), key = lambda x: x[1])]] + ["Policy_Number", ]

df_train = df_train.merge(df_tmp, on = ["Policy_Number"], how = "left")

## Feature: 各個Insurance_Coverage_header_number數量
feature_col = "Insurance_Coverage_header_number"
mapping_df_tmp = df_policy.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_train = df_train.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-44]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Insurance_Coverage_header_number比例
feature_col = "Insurance_Coverage_header_number"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_train = df_train.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-44]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Insurance_Coverage_tail_symbol數量
feature_col = "Insurance_Coverage_tail_symbol"
mapping_df_tmp = df_policy.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_train = df_train.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-17]) + ["Count_" + str(x)[3] + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Insurance_Coverage_tail_symbol比例
feature_col = "Insurance_Coverage_tail_symbol"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_train = df_train.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-17]) + ["Percentage_" + str(x)[3] + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: Sum_Insured_Amount, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Sum_Insured_Amount"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Sum_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Mean_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Median_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Max_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Min_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Std_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: Distribution_Channel值
feature_col = "Distribution_Channel"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: aassured_zip值
feature_col = "aassured_zip"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: iply_area值(One Hot Encoding)
feature_col = "iply_area"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
le = LabelEncoder()
oe = OneHotEncoder(sparse=False, n_values=len(set(df_policy[feature_col])))
le.fit(df_policy[feature_col])
oe.fit(le.transform(df_train[feature_col]).reshape([-1, 1]))
one_hot_transformer[feature_col] = [le, oe]

df_train = pd.concat([df_train, pd.DataFrame(one_hot_transformer[feature_col][1].transform(one_hot_transformer[feature_col][0].transform(df_train[feature_col]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_" + str(x) + "_" + feature_col for x in one_hot_transformer[feature_col][0].classes_])], axis = 1)
df_train.drop([feature_col], axis = 1, inplace = True)

## Feature: iply_area值
feature_col = "iply_area"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: fsex值(One Hot Encoding)
# 用Insured's_ID回推，找到fsex
feature_col = "fsex"
mapping_dict_tmp = df_policy.groupby(["Insured's_ID"])[feature_col].agg(lambda x: list(set(x)))
mapping_dict_tmp = mapping_dict_tmp.apply(lambda x: x[1] if len(x) == 2 else x[0])
df_policy[feature_col + "_From_Insured's_ID"] = df_policy["Insured's_ID"].apply(lambda x: mapping_dict_tmp[x])
mapping_dict_tmp = df_policy.groupby(["Policy_Number"])[feature_col + "_From_Insured's_ID"].agg(lambda x: list(x)[0])
mapping_dict_tmp[mapping_dict_tmp.isnull()] = '0'
df_train["Value_" + feature_col + "_From_Insured's_ID"] = list(swiftapply(df_train["Policy_Number"], lambda x: mapping_dict_tmp[x]))

le = LabelEncoder()
oe = OneHotEncoder(sparse=False, n_values=len(set(df_train["Value_" + feature_col + "_From_Insured's_ID"])))
le.fit(df_train["Value_" + feature_col + "_From_Insured's_ID"])
oe.fit(le.transform(df_train["Value_" + feature_col + "_From_Insured's_ID"]).reshape([-1, 1]))
one_hot_transformer[feature_col + "_From_Insured's_ID"] = [le, oe]

df_train = pd.concat([df_train, pd.DataFrame(one_hot_transformer[feature_col + "_From_Insured's_ID"][1].transform(one_hot_transformer[feature_col + "_From_Insured's_ID"][0].transform(df_train["Value_" + feature_col + "_From_Insured's_ID"]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_" + str(x) + "_" + feature_col + "_From_Insured's_ID" for x in one_hot_transformer[feature_col + "_From_Insured's_ID"][0].classes_])], axis = 1)

## Feature: fsex值
feature_col = "fsex"
df_train["Category_" + feature_col + "_From_Insured's_ID"] = df_train["Value_" + feature_col + "_From_Insured's_ID"]
df_train.drop(["Value_" + feature_col + "_From_Insured's_ID"], axis = 1, inplace = True)

## Feature: fmarriage值(One Hot Encoding)
# 用Insured's_ID回推，找到fmarriage
feature_col = "fmarriage"
mapping_dict_tmp = df_policy.groupby(["Insured's_ID"])[feature_col].agg(lambda x: list(set(x)))
mapping_dict_tmp = mapping_dict_tmp.apply(lambda x: x[1] if len(x) == 2 else (x[2] if len(x) == 3 else x[0]))
df_policy[feature_col + "_From_Insured's_ID"] = df_policy["Insured's_ID"].apply(lambda x: mapping_dict_tmp[x])
mapping_dict_tmp = df_policy.groupby(["Policy_Number"])[feature_col + "_From_Insured's_ID"].agg(lambda x: list(x)[0])
mapping_dict_tmp[mapping_dict_tmp.isnull()] = '0'
df_train["Value_" + feature_col + "_From_Insured's_ID"] = list(swiftapply(df_train["Policy_Number"], lambda x: mapping_dict_tmp[x]))

le = LabelEncoder()
oe = OneHotEncoder(sparse=False, n_values=len(set(df_train["Value_" + feature_col + "_From_Insured's_ID"])))
le.fit(df_train["Value_" + feature_col + "_From_Insured's_ID"])
oe.fit(le.transform(df_train["Value_" + feature_col + "_From_Insured's_ID"]).reshape([-1, 1]))
one_hot_transformer[feature_col + "_From_Insured's_ID"] = [le, oe]

df_train = pd.concat([df_train, pd.DataFrame(one_hot_transformer[feature_col + "_From_Insured's_ID"][1].transform(one_hot_transformer[feature_col + "_From_Insured's_ID"][0].transform(df_train["Value_" + feature_col + "_From_Insured's_ID"]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_" + str(x) + "_" + feature_col + "_From_Insured's_ID" for x in one_hot_transformer[feature_col + "_From_Insured's_ID"][0].classes_])], axis = 1)

## Feature: fmarriage值
feature_col = "fmarriage"
df_train["Category_" + feature_col + "_From_Insured's_ID"] = df_train["Value_" + feature_col + "_From_Insured's_ID"]
df_train.drop(["Value_" + feature_col + "_From_Insured's_ID"], axis = 1, inplace = True)

## Feature: ibirth值(有NA值)
end_date = "2016/12"
feature_col = "ibirth"
#df_policy.groupby(["Insured's_ID"])[feature_col].agg(lambda x: len(set(x))).max()
#a = df_policy.groupby(["Insured's_ID"])[feature_col].value_counts().unstack()
#a_na = a.apply(lambda x: np.sum(~np.isnan(x)), axis = 1)
#a_na.min() # 1
#a_na.max() # 2 
# 有不少例子會有多個生日，以Policy_Number對應到的ibirth的為主
df_policy["Age_Year_" + feature_col] = (pd.to_datetime(end_date) - pd.to_datetime(df_policy[feature_col])).astype('timedelta64[Y]')
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Value_" + "Age_Year_" + feature_col, ]

## Feature: dbirth值(有NA值)
feature_col = "dbirth"
df_policy["Age_Year_" + feature_col] = (pd.to_datetime(end_date) - pd.to_datetime(df_policy[feature_col], errors = "coerce")).astype('timedelta64[Y]')
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Value_" + "Age_Year_" + feature_col, ]

## Feature: Value_plia_acc, Value_pdmg_acc總和
df_train["Sum_Value_plia_acc_and_Value_pdmg_acc"] = df_train["Value_plia_acc"] + df_train["Value_pdmg_acc"]

## Feature: Sum_Insured_Amount1對Sum_Premium的比例
df_train["Ratio_Sum_Insured_Amount1_to_Sum_Premium"] = df_train["Sum_Insured_Amount1"]/df_train["Sum_Premium"]

## Feature: Sum_Insured_Amount2對Sum_Premium的比例
df_train["Ratio_Sum_Insured_Amount2_to_Sum_Premium"] = df_train["Sum_Insured_Amount2"]/df_train["Sum_Premium"]

## Feature: Sum_Insured_Amount3對Sum_Premium的比例
df_train["Ratio_Sum_Insured_Amount3_to_Sum_Premium"] = df_train["Sum_Insured_Amount3"]/df_train["Sum_Premium"]

## Feature: Sum_Insured_Amount1對Value_Replacement_cost_of_insured_vehicle的比例
df_train["Ratio_Amount1_to_Sum_Premium_to_Value_Replacement_cost_of_insured_vehicle"] = df_train["Sum_Insured_Amount1"]/df_train["Value_Replacement_cost_of_insured_vehicle"]

## Feature: Sum_Insured_Amount2對Value_Replacement_cost_of_insured_vehicle的比例
df_train["Ratio_Amount2_to_Sum_Premium_to_Value_Replacement_cost_of_insured_vehicle"] = df_train["Sum_Insured_Amount2"]/df_train["Value_Replacement_cost_of_insured_vehicle"]

## Feature: Sum_Insured_Amount3對Value_Replacement_cost_of_insured_vehicle的比例
df_train["Ratio_Amount3_to_Sum_Premium_to_Value_Replacement_cost_of_insured_vehicle"] = df_train["Sum_Insured_Amount3"]/df_train["Value_Replacement_cost_of_insured_vehicle"]

### 將categorical variable篩選至df_train和df_test裡面皆有，且數量夠多的category，不要的category視為rare
feature_category = ["Category_Insured's_ID",
 'Category_Vehicle_Make_and_Model1',
 'Category_Vehicle_Make_and_Model2',
 'Category_Imported_or_Domestic_Car',
 'Category_Coding_of_Vehicle_Branding_&_Type',
 'Category_fassured',
 'Category_Distribution_Channel',
 'Category_aassured_zip',
 'Category_iply_area',
 "Category_fsex_From_Insured's_ID",
 "Category_fmarriage_From_Insured's_ID"]

for i in feature_category:
    feature = "_".join(i.split("_")[1:])
    print([feature, len(set(df_policy[feature])), len(set(df_policy[feature][df_policy["Train"] == 1])), 
        len(set(df_policy[feature][df_policy["Train"] == 0]))])

feature_category = ["Category_Insured's_ID",
 'Category_Vehicle_Make_and_Model1',
 'Category_Vehicle_Make_and_Model2',
 'Category_Coding_of_Vehicle_Branding_&_Type',
 'Category_Distribution_Channel',
 'Category_aassured_zip',
 'Category_iply_area']

for i in feature_category:
    feature = "_".join(i.split("_")[1:])
    print([feature, len(set(df_policy[feature])), len(set(df_policy[feature][df_policy["Train"] == 1])), 
        len(set(df_policy[feature][df_policy["Train"] == 0]))])

overlap_category_list = []

for  i in feature_category:
    
    feature = "_".join(i.split("_")[1:])
    overlap_category_list.append([feature,
        list(np.intersect1d(list(set(df_policy[feature][df_policy["Train"] == 1])),
                            list(set(df_policy[feature][df_policy["Train"] == 0]))))])

[x[0] for x in overlap_category_list]
[len(x[1]) for x in overlap_category_list]

## 重複的category
overlap_category_not_other_dict = dict()

for  i in overlap_category_list:
    
    category_list = i[1]    
    overlap_category_not_other_dict[i[0]] = category_list

## 重複且量不過少的category
overlap_category_not_rare_dict = dict()

for  i in overlap_category_list:
    
    category_list = i[1]
    
    category_count_threshold = np.ceil(len(df_policy[df_policy[i[0]].isin(category_list)]))/len(category_list)

    print(i[0])
    #print(df_policy[i[0]][df_policy[i[0]].isin(category_list)].value_counts().min())
    ind = np.sum(df_policy[i[0]][df_policy[i[0]].isin(category_list)].value_counts() > category_count_threshold)
    
    print(ind)
    
    overlap_category_not_rare_dict[i[0]] = list(df_policy[i[0]][df_policy[i[0]].isin(category_list)].value_counts().sort_values(ascending = False).index)[:ind]

## Feature: Insured's_ID值(轉換成other)
feature_col = "Insured's_ID"
df_train["Category_with_other_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: Vehicle_Make_and_Model1值(轉換成other)
feature_col = "Vehicle_Make_and_Model1"
df_train["Category_with_other_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: Vehicle_Make_and_Model2值(轉換成other)
feature_col = "Vehicle_Make_and_Model2"
df_train["Category_with_other_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: Coding_of_Vehicle_Branding_&_Type值(轉換成other)
feature_col = "Coding_of_Vehicle_Branding_&_Type"
df_train["Category_with_other_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: Distribution_Channel值(轉換成other)
feature_col = "Distribution_Channel"
df_train["Category_with_other_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: aassured_zip值(轉換成other)
feature_col = "aassured_zip"
df_train["Category_with_other_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: Insured's_ID值(轉換成rare)
feature_col = "Insured's_ID"
df_train["Category_with_rare_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: Vehicle_Make_and_Model1值(轉換成rare)
feature_col = "Vehicle_Make_and_Model1"
df_train["Category_with_rare_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: Vehicle_Make_and_Model2值(轉換成rare)
feature_col = "Vehicle_Make_and_Model2"
df_train["Category_with_rare_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: Coding_of_Vehicle_Branding_&_Type值(轉換成rare)
feature_col = "Coding_of_Vehicle_Branding_&_Type"
df_train["Category_with_rare_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: Distribution_Channel值(轉換成rare)
feature_col = "Distribution_Channel"
df_train["Category_with_rare_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: aassured_zip值(轉換成rare)
feature_col = "aassured_zip"
df_train["Category_with_rare_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: Vehicle_Make_and_Model1值 not other(One Hot Encoding)
feature_col = "Vehicle_Make_and_Model1"
le = LabelEncoder()
oe = OneHotEncoder(sparse=False, n_values=len(overlap_category_not_other_dict[feature_col])+1)
le.fit(df_train["Category_with_other_" + feature_col])
oe.fit(le.transform(df_train["Category_with_other_" + feature_col]).reshape([-1, 1]))
one_hot_transformer[feature_col] = [le, oe]

df_train = pd.concat([df_train, pd.DataFrame(one_hot_transformer[feature_col][1].transform(one_hot_transformer[feature_col][0].transform(df_train["Category_with_other_" + feature_col]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_with_other_" + str(x) + "_" + feature_col for x in one_hot_transformer[feature_col][0].classes_])], axis = 1)

## Feature: Distribution_Channel值 not other(One Hot Encoding)
feature_col = "Distribution_Channel"
le = LabelEncoder()
oe = OneHotEncoder(sparse=False, n_values=len(overlap_category_not_other_dict[feature_col])+1)
le.fit(df_train["Category_with_other_" + feature_col])
oe.fit(le.transform(df_train["Category_with_other_" + feature_col]).reshape([-1, 1]))
one_hot_transformer[feature_col] = [le, oe]

df_train = pd.concat([df_train, pd.DataFrame(one_hot_transformer[feature_col][1].transform(one_hot_transformer[feature_col][0].transform(df_train["Category_with_other_" + feature_col]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_with_other_" + str(x) + "_" + feature_col for x in one_hot_transformer[feature_col][0].classes_])], axis = 1)

## Feature: aassured_zip值 not other(One Hot Encoding)
feature_col = "aassured_zip"
le = LabelEncoder()
oe = OneHotEncoder(sparse=False, n_values=len(overlap_category_not_other_dict[feature_col])+1)
le.fit(df_train["Category_with_other_" + feature_col])
oe.fit(le.transform(df_train["Category_with_other_" + feature_col]).reshape([-1, 1]))
one_hot_transformer[feature_col] = [le, oe]

df_train = pd.concat([df_train, pd.DataFrame(one_hot_transformer[feature_col][1].transform(one_hot_transformer[feature_col][0].transform(df_train["Category_with_other_" + feature_col]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_with_other_" + str(x) + "_" + feature_col for x in one_hot_transformer[feature_col][0].classes_])], axis = 1)

## Feature: Value_Engine_Displacement_(Cubic_Centimeter)對Value_qpt的比例
df_train["Ratio_Value_Engine_Displacement_(Cubic_Centimeter)_to_Value_qpt"] = \
    df_train["Value_Engine_Displacement_(Cubic_Centimeter)"]/df_train["Value_qpt"]

## Feature: lia_class值(> 10 都視為10)
#overlap_category_not_rare_dict["lia_class"] = [x for x in sorted(set(df_policy["lia_class"])) if x <= 10]
feature_col = "lia_class"
df_train["Category_with_rare_" + feature_col] = swiftapply(df_train["Value_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else 10)

## Feature: Value_Engine_Displacement_(Cubic_Centimeter)對Value_qpt的比例
df_train["Ratio_Value_Engine_Displacement_(Cubic_Centimeter)_to_Value_Replacement_cost_of_insured_vehicle"] = \
    df_train["Value_Engine_Displacement_(Cubic_Centimeter)"]/df_train["Value_Replacement_cost_of_insured_vehicle"]

## Feature: Sum_Premium對1 + Sum_Value_plia_acc_and_Value_pdmg_acc的比例
df_train["Ratio_Sum_Premium_to_1+Sum_Value_plia_acc_and_Value_pdmg_acc"] = \
    df_train["Sum_Premium"]/(1 + df_train["Sum_Value_plia_acc_and_Value_pdmg_acc"])

## Feature: iply_area值(轉換成other)
feature_col = "iply_area"
df_train["Category_with_other_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: iply_area值(轉換成rare)
feature_col = "iply_area"
df_train["Category_with_rare_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: Imported_or_Domestic_Car值(21, 22, 23 都視為20)
feature_col = "Imported_or_Domestic_Car"
#overlap_category_not_rare_dict[feature_col] = [x for x in sorted(set(df_policy[feature_col])) if x not in [21, 22, 23]]
df_train["Category_with_rare_" + feature_col] = swiftapply(df_train["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else 20)

## Feature: Sum_Premium對的Value_Engine_Displacement_(Cubic_Centimeter)比例
df_train["Ratio_Sum_Premium_to_Value_Engine_Displacement_(Cubic_Centimeter)"] = \
    df_train["Sum_Premium"]/df_train["Value_Engine_Displacement_(Cubic_Centimeter)"]

## Feature: Sum_Premium對的Value_Manafactured_Year_and_Month比例
df_train["Ratio_Sum_Premium_to_Value_Manafactured_Year_and_Month"] = \
    df_train["Sum_Premium"]/df_train["Value_Manafactured_Year_and_Month"]

## Feature: Sum_Premium對的Value_qpt比例
df_train["Ratio_Sum_Premium_to_Value_qpt"] = \
    df_train["Sum_Premium"]/df_train["Value_qpt"]

## Feature: Sum_Premium取Log
df_train["Log_Sum_Premium"] = \
    np.log1p(df_train["Sum_Premium"])

## Feature: Coverage_Deductible_if_applied金額
df_policy["Coverage_Deductible_if_applied_transformed_value"] = swiftapply(df_policy["Coverage_Deductible_if_applied"], lambda x: 7000 if x == 1 else(\
                                                                     8000 if x == 2 else(10000 if x == 3 else(-1 if x < 0 else x))))
df_policy["Coverage_Deductible_if_applied_transformed_value"] = df_policy.loc[:, ["Coverage_Deductible_if_applied_transformed_value", "Sum_Insured_Amount"]].apply(lambda x: x[0]*x[1]/100 if x[0] in [10, 20] else x[0], axis = 1)

insurance_id = list(df_insurance_type_info.loc[(df_insurance_type_info['自負額說明   【0:無自負額  負數:已退保   其他:詳下列說明】'] == "與自負額無關,請忽略"), :]["險種代號"])
df_policy["Coverage_Deductible_if_applied_transformed_value"][df_policy["Insurance_Coverage"].isin(insurance_id)] = -2

# Feature: Sum_Insured_Amount, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Coverage_Deductible_if_applied_transformed_value"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x[x > 0])).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Sum_" + feature_col, ]
df_train["Sum_" + feature_col].fillna(0, inplace = True)

df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x[x > 0])).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Mean_" + feature_col, ]
df_train["Mean_" + feature_col].fillna(0, inplace = True)

df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x[x > 0])).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Median_" + feature_col, ]
df_train["Median_" + feature_col].fillna(0, inplace = True)

df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x[x > 0])).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Max_" + feature_col, ]
df_train["Max_" + feature_col].fillna(0, inplace = True)

df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x[x > 0])).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Min_" + feature_col, ]
df_train["Min_" + feature_col].fillna(0, inplace = True)

df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x[x > 0])).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Std_" + feature_col, ]
df_train["Std_" + feature_col].fillna(0, inplace = True)

df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x == 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_0_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x == 0)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Percentage_0_" + feature_col, ]

df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x == -1)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_-1_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x == -1)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Percentage_-1_" + feature_col, ]

df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x == -2)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_-2_" + feature_col, ]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x == -2)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Percentage_-2_" + feature_col, ]

## Feature: Coverage_Deductible_if_applied是金額的數量
df_policy["Coverage_Deductible_if_applied_transformed_value"] = swiftapply(df_policy["Coverage_Deductible_if_applied"], lambda x: 7000 if x == 1 else(\
                                                                     8000 if x == 2 else(10000 if x == 3 else(-1 if x < 0 else x))))
insurance_id = list(df_insurance_type_info.loc[(df_insurance_type_info['自負額說明   【0:無自負額  負數:已退保   其他:詳下列說明】'] == "與自負額無關,請忽略"), :]["險種代號"])
df_policy["Coverage_Deductible_if_applied_transformed_value"][df_policy["Insurance_Coverage"].isin(insurance_id)] = -2

feature_col = "Coverage_Deductible_if_applied_transformed_value"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x > 20)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_value_" + feature_col, ]

## Feature: Coverage_Deductible_if_applied是金額的比例
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x > 20)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Percentage_value_" + feature_col, ]

## Feature: Coverage_Deductible_if_applied是比例的數量
feature_col = "Coverage_Deductible_if_applied_transformed_value"
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum((x == 10)|(x == 20))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Count_percentage_" + feature_col, ]

## Feature: Coverage_Deductible_if_applied是比例的比例
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum((x == 10)|(x == 20))/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Percentage_percentage_" + feature_col, ]

df_policy["Coverage_Deductible_if_applied_transformed_value"] = swiftapply(df_policy["Coverage_Deductible_if_applied"], lambda x: 7000 if x == 1 else(\
                                                                     8000 if x == 2 else(10000 if x == 3 else(-1 if x < 0 else x))))
df_policy["Coverage_Deductible_if_applied_transformed_value"] = df_policy.loc[:, ["Coverage_Deductible_if_applied_transformed_value", "Sum_Insured_Amount"]].apply(lambda x: x[0]*x[1]/100 if x[0] in [10, 20] else x[0], axis = 1)

insurance_id = list(df_insurance_type_info.loc[(df_insurance_type_info['自負額說明   【0:無自負額  負數:已退保   其他:詳下列說明】'] == "與自負額無關,請忽略"), :]["險種代號"])
df_policy["Coverage_Deductible_if_applied_transformed_value"][df_policy["Insurance_Coverage"].isin(insurance_id)] = -2

## Feature: Sum_Premium對Sum_Coverage_Deductible_if_applied_transformed_value的比例
df_train["Sum_Premium"]/df_train["Sum_Coverage_Deductible_if_applied_transformed_value"]
df_train["Ratio_Sum_Premium_to_Sum_Coverage_Deductible_if_applied_transformed_value"] = \
    df_train.loc[:, ["Sum_Premium", "Sum_Coverage_Deductible_if_applied_transformed_value"]].apply(lambda x: \
                x[0]/x[1] if x[1] > 0 else 0, axis = 1)

## Feature: Count_Policy_Number和Value_Multiple_Products_with_TmNewa_(Yes_or_No?)的總和
df_train["Sum_Count_Policy_Number_and_Value_Multiple_Products_with_TmNewa_(Yes_or_No?)"] = \
    df_train["Count_Policy_Number"] + df_train["Value_Multiple_Products_with_TmNewa_(Yes_or_No?)"]

## Feature: Count_Policy_Number和Value_Multiple_Products_with_TmNewa_(Yes_or_No?)的差距
df_train["Diff_Count_Policy_Number_and_Value_Multiple_Products_with_TmNewa_(Yes_or_No?)"] = \
    df_train["Count_Policy_Number"] - df_train["Value_Multiple_Products_with_TmNewa_(Yes_or_No?)"]  

## Feature: ibirth和dbirth是否一樣
df_policy["Binary_equal_ibirth_and_dbirth"] = (df_policy["ibirth"] == df_policy["dbirth"])*1
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])["Binary_equal_ibirth_and_dbirth"].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train.columns = list(df_train.columns[:-1]) + ["Binary_equal_ibirth_and_dbirth", ]

## Feature: Value_Engine_Displacement_(Cubic_Centimeter)對Manafactured_Year_and_Month的比例
df_train["Ratio_Value_Engine_Displacement_(Cubic_Centimeter)_to_Value_Manafactured_Year_and_Month"] = \
    df_train["Value_Engine_Displacement_(Cubic_Centimeter)"]/df_train["Value_Manafactured_Year_and_Month"]

## Feature: Value_Manafactured_Year_and_Month與今年年份差距和Value_Age_Year_ibirth的差距
df_policy["Age_Manafactured_Year_and_Month"] = 2017 - df_policy["Manafactured_Year_and_Month"]
df_train = df_train.merge(df_policy.groupby(["Policy_Number"])["Age_Manafactured_Year_and_Month"].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"] = \
    df_train["Value_Age_Year_ibirth"] - df_train["Age_Manafactured_Year_and_Month"]

## Feature: Value_Manafactured_Year_and_Month與今年年份差距和Value_Age_Year_dbirth的差距
df_train["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"] = \
    df_train["Value_Age_Year_dbirth"] - df_train["Age_Manafactured_Year_and_Month"]

df_train.drop(["Age_Manafactured_Year_and_Month"], axis = 1, inplace = True)

insurance_id = list(df_insurance_type_info.loc[(df_insurance_type_info['自負額說明   【0:無自負額  負數:已退保   其他:詳下列說明】'] != "與自負額無關,請忽略")&\
                                               (df_insurance_type_info['自負額說明   【0:無自負額  負數:已退保   其他:詳下列說明】'] != '  ')&\
                                               (~df_insurance_type_info['自負額說明   【0:無自負額  負數:已退保   其他:詳下列說明】'].isnull()), :]["險種代號"])

insurance_id = list(df_insurance_type_info.loc[(df_insurance_type_info['自負額說明   【0:無自負額  負數:已退保   其他:詳下列說明】'] != "與自負額無關,請忽略"), :]["險種代號"])    
    
insurance_id = list(df_insurance_type_info.loc[(df_insurance_type_info['自負額說明   【0:無自負額  負數:已退保   其他:詳下列說明】'] == "與自負額無關,請忽略"), :]["險種代號"])

    
df_policy.loc[:, :]["Coverage_Deductible_if_applied"].value_counts().sort_index()
df_policy.loc[df_policy["Insurance_Coverage"].isin(insurance_id), :]["Coverage_Deductible_if_applied"].value_counts().sort_index()

df_policy.groupby(["Policy_Number"])["Coverage_Deductible_if_applied"].apply(lambda x: len(set(x))).max()
df_policy.loc[df_policy["Insurance_Coverage"].isin(insurance_id), :].groupby(["Policy_Number"])["Coverage_Deductible_if_applied"].apply(lambda x: len(set(x))).max()

df_policy["Coverage_Deductible_if_applied"].value_counts()

df_policy["Coverage_Deductible_if_applied_transformed"] = df_policy.loc[:, ["Coverage_Deductible_if_applied_transformed", "Sum_Insured_Amount"]].apply(lambda x: x[0]*x[1]/100 if x[0] in [10, 20] else x[0], axis = 1)

df_policy.loc[df_policy["Insurance_Coverage"].isin(insurance_id), :]["Coverage_Deductible_if_applied_transformed"].value_counts()

## Feature: Prior_Policy_Number的Premium Sum, Mean, Median, Max, Min, Std, 非0數量, 非0比例
feature_col = "Premium"

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.sum(x)))
df_train["Sum_Prior_Policy_Number_" + feature_col] = df_train["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.mean(x)))
df_train["Mean_Prior_Policy_Number_" + feature_col] = df_train["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.median(x)))
df_train["Median_Prior_Policy_Number_" + feature_col] = df_train["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.max(x)))
df_train["Max_Prior_Policy_Number_" + feature_col] = df_train["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.min(x)))
df_train["Min_Prior_Policy_Number_" + feature_col] = df_train["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.std(x)))
df_train["Std_Prior_Policy_Number_" + feature_col] = df_train["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)))
df_train["Count_NOT_0_Prior_Policy_Number_" + feature_col] = df_train["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(list(x))))
df_train["Percentage_NOT_0_Prior_Policy_Number_" + feature_col] = df_train["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))


### df_claims

## Feature: Policy_Number有無在df_claim的Policy_Number裡面
claim_Policy_Number_list = list(set(df_claim["Policy_Number"]))
df_train["Binary_in_df_claims"] = swiftapply(df_train["Policy_Number"], lambda x: 1 if x in claim_Policy_Number_list else 0)

## Feature: Policy_Number出現在df_claim的Policy_Number裡面的次數
mapping_dict_tmp = dict(df_claim["Policy_Number"].value_counts())
df_train["Count_in_df_claims"] = df_train["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))


# df_test
# -----------------------------------------------------------------------------
### Use df_policy

## Feature: Policy_Number出現數量
feature_col = "Policy_Number"
df_test = df_test.merge(df_policy.groupby([feature_col])["Insured's_ID"].agg(lambda x: len(x)).reset_index(),
               on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_" + feature_col, ]

## Feature: Policy_Number對應的Insured's_ID出現在不同的獨立Policy_Number次數
feature_col = "Insured's_ID"
mapping_dict_tmp = (df_policy.groupby([feature_col])["Policy_Number"].agg(lambda x: len(set(x))))
df_policy["Count_" + feature_col + "_in_Unique_Policy_Number"] = list(swiftapply(df_policy[feature_col], lambda x: mapping_dict_tmp[x]))

df_test = df_test.merge(df_policy.groupby(["Policy_Number"])["Count_" + feature_col + "_in_Unique_Policy_Number"].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_" + feature_col + "_in_Unique_Policy_Number", ]

## Feature: Policy_Number對應的Insured's_ID出現的次數
feature_col = "Insured's_ID"
mapping_dict_tmp = (df_policy.groupby([feature_col])["Policy_Number"].agg(lambda x: len(x)))
df_policy["Count_" + feature_col + "_in_Total_Policy_Number"] = list(swiftapply(df_policy[feature_col], lambda x: mapping_dict_tmp[x]))

df_test = df_test.merge(df_policy.groupby(["Policy_Number"])["Count_" + feature_col + "_in_Total_Policy_Number"].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_" + feature_col + "_in_Total_Policy_Number", ]

## Feature: 有無Prior_Policy_Number
feature_col = "Prior_Policy_Number"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(\
            lambda x: 1 if (~pd.Series(x).isnull()).sum() > 0 else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Binary_Not_NA_" + feature_col, ]

## Feature: Cancellation是否為"Y" (看起來和"有無Prior_Policy_Number"是一樣的，不過有一筆不一樣)
feature_col = "Cancellation"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(\
                          lambda x: 1 if list(x)[0] == "Y" else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Binary_Y_" + feature_col, ]

## Feature: Policy_Number對應的Insured's_ID有幾種不同的非NA Vehicle_identifier
feature_col = "Vehicle_identifier"
mapping_dict_tmp = df_policy.groupby(["Insured's_ID"])[feature_col].agg(lambda x:  len(set(x[~pd.Series(x).isnull()])))
df_policy["Count_Insured's_ID_in_Unique_Not_NA_" + feature_col] = list(swiftapply(df_policy["Insured's_ID"], lambda x: mapping_dict_tmp[x]))
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])["Count_Insured's_ID_in_Unique_Not_NA_" + feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_Insured's_ID_in_Unique_Not_NA_" + feature_col, ]

## Feature: Policy_Number對應的Insured's_ID有幾個非NA Vehicle_identifier
feature_col = "Vehicle_identifier"
mapping_dict_tmp = df_policy.groupby(["Insured's_ID"])[feature_col].agg(lambda x:  np.sum((~pd.Series(x).isnull())))
df_policy["Count_Insured's_ID_in_Not_NA_" + feature_col] = list(swiftapply(df_policy["Insured's_ID"], lambda x: mapping_dict_tmp[x]))
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])["Count_Insured's_ID_in_Not_NA_" + feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_Insured's_ID_in_Not_NA_" + feature_col, ]

## Feature: 有無Vehicle_identifier
feature_col = "Vehicle_identifier"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(\
            lambda x: 1 if (~pd.Series(x).isnull()).sum() > 0 else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Binary_Not_NA_" + feature_col, ]

## Feature: Manafactured_Year_and_Month值
feature_col = "Manafactured_Year_and_Month"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: Engine_Displacement_(Cubic_Centimeter)值
feature_col = "Engine_Displacement_(Cubic_Centimeter)"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: Insured's_ID值
feature_col = "Insured's_ID"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: Vehicle_Make_and_Model1值
feature_col = "Vehicle_Make_and_Model1"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: Vehicle_Make_and_Model2值
feature_col = "Vehicle_Make_and_Model2"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: Imported_or_Domestic_Car值
feature_col = "Imported_or_Domestic_Car"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: Imported_or_Domestic_Car值(One Hot Encoding)
feature_col = "Imported_or_Domestic_Car"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")

df_test = pd.concat([df_test, pd.DataFrame(one_hot_transformer[feature_col][1].transform(one_hot_transformer[feature_col][0].transform(df_test[feature_col]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_" + str(x) + "_" + feature_col for x in one_hot_transformer[feature_col][0].classes_])], axis = 1)
df_test.drop([feature_col], axis = 1, inplace = True)

## Feature: Coding_of_Vehicle_Branding_&_Type值
feature_col = "Coding_of_Vehicle_Branding_&_Type"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: qpt值
feature_col = "qpt"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: 各個Main_Insurance_Coverage_Group數量
feature_col = "Main_Insurance_Coverage_Group"
mapping_df_tmp = df_policy.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_test = df_test.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-3]) + ["Count_" + x + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Main_Insurance_Coverage_Group比例
feature_col = "Main_Insurance_Coverage_Group"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_test = df_test.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-3]) + ["Percentage_" + x + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: Policy_Number對應的Main_Insurance_Coverage_Group有幾種不同的值
feature_col = "Main_Insurance_Coverage_Group"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: len(set(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_Unique_" + feature_col, ]

## Feature: 各個Insurance_Coverage數量
feature_col = "Insurance_Coverage"
mapping_df_tmp = df_policy.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_test = df_test.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-60]) + ["Count_" + x + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Insurance_Coverage比例
feature_col = "Insurance_Coverage"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_test = df_test.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-60]) + ["Percentage_" + x + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: Policy_Number對應的Insurance_Coverage有幾種不同的值
feature_col = "Insurance_Coverage"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: len(set(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_Unique_" + feature_col, ]

## Feature: Policy_Number對應的Main_Insurance_Coverage_Group有幾種不同的值對上Policy_Number出現次數的比率
df_test["Ratio_Count_Unique_Main_Insurance_Coverage_Group_to_Count_Policy_Number"] = df_test["Count_Unique_Main_Insurance_Coverage_Group"]/df_test["Count_Policy_Number"]

## Feature: Policy_Number對應的Insurance_Coverage有幾種不同的值對上Policy_Number出現次數的比率
df_test["Ratio_Count_Unique_Insurance_Coverage_to_Count_Policy_Number"] = df_test["Count_Unique_Insurance_Coverage"]/df_test["Count_Policy_Number"]

## Feature: Premium總和
feature_col = "Premium"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Sum_" + feature_col, ]

## Feature: Replacement_cost_of_insured_vehicle值
feature_col = "Replacement_cost_of_insured_vehicle"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: Replacement_cost_of_insured_vehicle值對上Premium總和的比率
df_test["Ratio_Value_Replacement_cost_of_insured_vehicle_to_Sum_Premium"] = df_test["Value_Replacement_cost_of_insured_vehicle"]/df_test["Sum_Premium"]

## Multiple_Products_with_TmNewa_(Yes_or_No?)值
feature_col = "Multiple_Products_with_TmNewa_(Yes_or_No?)"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Value_" + feature_col, ]

## Multiple_Products_with_TmNewa_(Yes_or_No?)值對上Policy_Number出現次數的比率
df_test["Ratio_Value_Multiple_Products_with_TmNewa_(Yes_or_No?)_to_Count_Policy_Number"] = df_test["Value_Multiple_Products_with_TmNewa_(Yes_or_No?)"]/df_test["Count_Policy_Number"]

## Feature: lia_class值
feature_col = "lia_class"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: plia_acc值
feature_col = "plia_acc"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: pdmg_acc值
feature_col = "pdmg_acc"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Value_" + feature_col, ]

## Feature: fassured值(One Hot Encoding)
feature_col = "fassured"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")

df_test = pd.concat([df_test, pd.DataFrame(one_hot_transformer[feature_col][1].transform(one_hot_transformer[feature_col][0].transform(df_test[feature_col]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_" + str(x) + "_" + feature_col for x in one_hot_transformer[feature_col][0].classes_])], axis = 1)
df_test.drop([feature_col], axis = 1, inplace = True)

## Feature: fassured值
feature_col = "fassured"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: Premium, Mean, Median, Max, Min, Std, 非0數量, 非0比例
feature_col = "Premium"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Mean_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Median_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Max_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Min_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Std_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: Insured_Amount1, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Insured_Amount1"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Sum_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Mean_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Median_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Max_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Min_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Std_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: Insured_Amount2, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Insured_Amount2"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Sum_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Mean_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Median_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Max_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Min_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Std_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: Insured_Amount3, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Insured_Amount3"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Sum_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Mean_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Median_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Max_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Min_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Std_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: Insurance_Coverage TF-IDF
feature_col = "Insurance_Coverage"

df_tmp = pd.DataFrame(tf_idf_transformer[feature_col].transform(list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).values)).todense())
df_tmp["Policy_Number"] = list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).index)
df_tmp.columns = ["TF_IDF_" + str(x) + "@" + "_" + feature_col if len(x) == 2 else \
                 "TF_IDF_" + str(x) + "_" + feature_col for x in
                 [x[0] for x in sorted(tf_idf_transformer[feature_col].vocabulary_.items(), key = lambda x: x[1])]] + ["Policy_Number", ]

df_test = df_test.merge(df_tmp, on = ["Policy_Number"], how = "left")

## Feature: Insurance_Coverage_header_number TF-IDF
df_policy["Insurance_Coverage_header_number"] = swiftapply(df_policy["Insurance_Coverage"], lambda x: x[:2])
feature_col = "Insurance_Coverage_header_number"

df_tmp = pd.DataFrame(tf_idf_transformer[feature_col].transform(list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).values)).todense())
df_tmp["Policy_Number"] = list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).index)
df_tmp.columns = ["TF_IDF_" + str(x) + "_" + feature_col for x in
                 [x[0] for x in sorted(tf_idf_transformer[feature_col].vocabulary_.items(), key = lambda x: x[1])]] + ["Policy_Number", ]

df_test = df_test.merge(df_tmp, on = ["Policy_Number"], how = "left")

## Feature: Insurance_Coverage_tail_symbol TF-IDF
feature_col = "Insurance_Coverage_tail_symbol"

df_tmp = pd.DataFrame(tf_idf_transformer[feature_col].transform(list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).values)).todense())
df_tmp["Policy_Number"] = list(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: " ".join(list(x))).index)
df_tmp.columns = ["TF_IDF_" + "@" + "_" + feature_col if len(x) == 3 else \
                 "TF_IDF_" + str(x)[-1] + "_" + feature_col for x in
                 [x[0] for x in sorted(tf_idf_transformer[feature_col].vocabulary_.items(), key = lambda x: x[1])]] + ["Policy_Number", ]

df_test = df_test.merge(df_tmp, on = ["Policy_Number"], how = "left")

## Feature: 各個Insurance_Coverage_header_number數量
feature_col = "Insurance_Coverage_header_number"
mapping_df_tmp = df_policy.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_test = df_test.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-44]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Insurance_Coverage_header_number比例
feature_col = "Insurance_Coverage_header_number"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_test = df_test.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-44]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Insurance_Coverage_tail_symbol數量
feature_col = "Insurance_Coverage_tail_symbol"
mapping_df_tmp = df_policy.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_test = df_test.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-17]) + ["Count_" + str(x)[3] + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Insurance_Coverage_tail_symbol比例
feature_col = "Insurance_Coverage_tail_symbol"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_test = df_test.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-17]) + ["Percentage_" + str(x)[3] + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: Sum_Insured_Amount, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Sum_Insured_Amount"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Sum_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Mean_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Median_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Max_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Min_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Std_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: Distribution_Channel值
feature_col = "Distribution_Channel"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: aassured_zip值
feature_col = "aassured_zip"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: iply_area值(One Hot Encoding)
feature_col = "iply_area"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")

df_test = pd.concat([df_test, pd.DataFrame(one_hot_transformer[feature_col][1].transform(one_hot_transformer[feature_col][0].transform(df_test[feature_col]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_" + str(x) + "_" + feature_col for x in one_hot_transformer[feature_col][0].classes_])], axis = 1)
df_test.drop([feature_col], axis = 1, inplace = True)

## Feature: iply_area值
feature_col = "iply_area"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Category_" + feature_col, ]

## Feature: fsex值(One Hot Encoding)
# 用Insured's_ID回推，找到fsex
feature_col = "fsex"
mapping_dict_tmp = df_policy.groupby(["Insured's_ID"])[feature_col].agg(lambda x: list(set(x)))
mapping_dict_tmp = mapping_dict_tmp.apply(lambda x: x[1] if len(x) == 2 else x[0])
df_policy[feature_col + "_From_Insured's_ID"] = df_policy["Insured's_ID"].apply(lambda x: mapping_dict_tmp[x])
mapping_dict_tmp = df_policy.groupby(["Policy_Number"])[feature_col + "_From_Insured's_ID"].agg(lambda x: list(x)[0])
mapping_dict_tmp[mapping_dict_tmp.isnull()] = '0'
df_test["Value_" + feature_col + "_From_Insured's_ID"] = list(swiftapply(df_test["Policy_Number"], lambda x: mapping_dict_tmp[x]))

df_test = pd.concat([df_test, pd.DataFrame(one_hot_transformer[feature_col + "_From_Insured's_ID"][1].transform(one_hot_transformer[feature_col + "_From_Insured's_ID"][0].transform(df_test["Value_" + feature_col + "_From_Insured's_ID"]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_" + str(x) + "_" + feature_col + "_From_Insured's_ID" for x in one_hot_transformer[feature_col + "_From_Insured's_ID"][0].classes_])], axis = 1)

## Feature: fsex值
feature_col = "fsex"
df_test["Category_" + feature_col + "_From_Insured's_ID"] = df_test["Value_" + feature_col + "_From_Insured's_ID"]
df_test.drop(["Value_" + feature_col + "_From_Insured's_ID"], axis = 1, inplace = True)

## Feature: fmarriage值(One Hot Encoding)
# 用Insured's_ID回推，找到fmarriage
feature_col = "fmarriage"
mapping_dict_tmp = df_policy.groupby(["Insured's_ID"])[feature_col].agg(lambda x: list(set(x)))
mapping_dict_tmp = mapping_dict_tmp.apply(lambda x: x[1] if len(x) == 2 else (x[2] if len(x) == 3 else x[0]))
df_policy[feature_col + "_From_Insured's_ID"] = df_policy["Insured's_ID"].apply(lambda x: mapping_dict_tmp[x])
mapping_dict_tmp = df_policy.groupby(["Policy_Number"])[feature_col + "_From_Insured's_ID"].agg(lambda x: list(x)[0])
mapping_dict_tmp[mapping_dict_tmp.isnull()] = '0'
df_test["Value_" + feature_col + "_From_Insured's_ID"] = list(swiftapply(df_test["Policy_Number"], lambda x: mapping_dict_tmp[x]))

df_test = pd.concat([df_test, pd.DataFrame(one_hot_transformer[feature_col + "_From_Insured's_ID"][1].transform(one_hot_transformer[feature_col + "_From_Insured's_ID"][0].transform(df_test["Value_" + feature_col + "_From_Insured's_ID"]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_" + str(x) + "_" + feature_col + "_From_Insured's_ID" for x in one_hot_transformer[feature_col + "_From_Insured's_ID"][0].classes_])], axis = 1)

## Feature: fmarriage值
feature_col = "fmarriage"
df_test["Category_" + feature_col + "_From_Insured's_ID"] = df_test["Value_" + feature_col + "_From_Insured's_ID"]
df_test.drop(["Value_" + feature_col + "_From_Insured's_ID"], axis = 1, inplace = True)

## Feature: ibirth值(有NA值)
end_date = "2016/12"
feature_col = "ibirth"
#df_policy.groupby(["Insured's_ID"])[feature_col].agg(lambda x: len(set(x))).max()
#a = df_policy.groupby(["Insured's_ID"])[feature_col].value_counts().unstack()
#a_na = a.apply(lambda x: np.sum(~np.isnan(x)), axis = 1)
#a_na.min() # 1
#a_na.max() # 2 
# 有不少例子會有多個生日，以Policy_Number對應到的ibirth的為主
df_policy["Age_Year_" + feature_col] = (pd.to_datetime(end_date) - pd.to_datetime(df_policy[feature_col])).astype('timedelta64[Y]')
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Value_" + "Age_Year_" + feature_col, ]

## Feature: dbirth值(有NA值)
feature_col = "dbirth"
df_policy["Age_Year_" + feature_col] = (pd.to_datetime(end_date) - pd.to_datetime(df_policy[feature_col], errors = "coerce")).astype('timedelta64[Y]')
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Value_" + "Age_Year_" + feature_col, ]

## Feature: Value_plia_acc, Value_pdmg_acc總和
df_test["Sum_Value_plia_acc_and_Value_pdmg_acc"] = df_test["Value_plia_acc"] + df_test["Value_pdmg_acc"]

## Feature: Sum_Insured_Amount1對Sum_Premium的比例
df_test["Ratio_Sum_Insured_Amount1_to_Sum_Premium"] = df_test["Sum_Insured_Amount1"]/df_test["Sum_Premium"]

## Feature: Sum_Insured_Amount2對Sum_Premium的比例
df_test["Ratio_Sum_Insured_Amount2_to_Sum_Premium"] = df_test["Sum_Insured_Amount2"]/df_test["Sum_Premium"]

## Feature: Sum_Insured_Amount3對Sum_Premium的比例
df_test["Ratio_Sum_Insured_Amount3_to_Sum_Premium"] = df_test["Sum_Insured_Amount3"]/df_test["Sum_Premium"]

## Feature: Sum_Insured_Amount1對Value_Replacement_cost_of_insured_vehicle的比例
df_test["Ratio_Amount1_to_Sum_Premium_to_Value_Replacement_cost_of_insured_vehicle"] = df_test["Sum_Insured_Amount1"]/df_test["Value_Replacement_cost_of_insured_vehicle"]

## Feature: Sum_Insured_Amount2對Value_Replacement_cost_of_insured_vehicle的比例
df_test["Ratio_Amount2_to_Sum_Premium_to_Value_Replacement_cost_of_insured_vehicle"] = df_test["Sum_Insured_Amount2"]/df_test["Value_Replacement_cost_of_insured_vehicle"]

## Feature: Sum_Insured_Amount3對Value_Replacement_cost_of_insured_vehicle的比例
df_test["Ratio_Amount3_to_Sum_Premium_to_Value_Replacement_cost_of_insured_vehicle"] = df_test["Sum_Insured_Amount3"]/df_test["Value_Replacement_cost_of_insured_vehicle"]

## Feature: Insured's_ID值(轉換成other)
feature_col = "Insured's_ID"
df_test["Category_with_other_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: Vehicle_Make_and_Model1值(轉換成other)
feature_col = "Vehicle_Make_and_Model1"
df_test["Category_with_other_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: Vehicle_Make_and_Model2值(轉換成other)
feature_col = "Vehicle_Make_and_Model2"
df_test["Category_with_other_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: Coding_of_Vehicle_Branding_&_Type值(轉換成other)
feature_col = "Coding_of_Vehicle_Branding_&_Type"
df_test["Category_with_other_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: Distribution_Channel值(轉換成other)
feature_col = "Distribution_Channel"
df_test["Category_with_other_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: aassured_zip值(轉換成other)
feature_col = "aassured_zip"
df_test["Category_with_other_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: Insured's_ID值(轉換成rare)
feature_col = "Insured's_ID"
df_test["Category_with_rare_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: Vehicle_Make_and_Model1值(轉換成rare)
feature_col = "Vehicle_Make_and_Model1"
df_test["Category_with_rare_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: Vehicle_Make_and_Model2值(轉換成rare)
feature_col = "Vehicle_Make_and_Model2"
df_test["Category_with_rare_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: Coding_of_Vehicle_Branding_&_Type值(轉換成rare)
feature_col = "Coding_of_Vehicle_Branding_&_Type"
df_test["Category_with_rare_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: Distribution_Channel值(轉換成rare)
feature_col = "Distribution_Channel"
df_test["Category_with_rare_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: aassured_zip值(轉換成rare)
feature_col = "aassured_zip"
df_test["Category_with_rare_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: Vehicle_Make_and_Model1值 not other(One Hot Encoding)
feature_col = "Vehicle_Make_and_Model1"

df_test = pd.concat([df_test, pd.DataFrame(one_hot_transformer[feature_col][1].transform(one_hot_transformer[feature_col][0].transform(df_test["Category_with_other_" + feature_col]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_with_other_" + str(x) + "_" + feature_col for x in one_hot_transformer[feature_col][0].classes_])], axis = 1)

## Feature: Distribution_Channel值 not other(One Hot Encoding)
feature_col = "Distribution_Channel"

df_test = pd.concat([df_test, pd.DataFrame(one_hot_transformer[feature_col][1].transform(one_hot_transformer[feature_col][0].transform(df_test["Category_with_other_" + feature_col]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_with_other_" + str(x) + "_" + feature_col for x in one_hot_transformer[feature_col][0].classes_])], axis = 1)

## Feature: aassured_zip值 not other(One Hot Encoding)
feature_col = "aassured_zip"

df_test = pd.concat([df_test, pd.DataFrame(one_hot_transformer[feature_col][1].transform(one_hot_transformer[feature_col][0].transform(df_test["Category_with_other_" + feature_col]).reshape([-1, 1])),
             columns = ["Binary_One_Hot_Encoding_with_other_" + str(x) + "_" + feature_col for x in one_hot_transformer[feature_col][0].classes_])], axis = 1)

## Feature: Value_Engine_Displacement_(Cubic_Centimeter)對Value_qpt的比例
df_test["Ratio_Value_Engine_Displacement_(Cubic_Centimeter)_to_Value_qpt"] = \
    df_test["Value_Engine_Displacement_(Cubic_Centimeter)"]/df_test["Value_qpt"]

## Feature: lia_class值(> 10 都視為10)
feature_col = "lia_class"
df_test["Category_with_rare_" + feature_col] = swiftapply(df_test["Value_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else 10)

## Feature: Value_Engine_Displacement_(Cubic_Centimeter)對Value_qpt的比例
df_test["Ratio_Value_Engine_Displacement_(Cubic_Centimeter)_to_Value_Replacement_cost_of_insured_vehicle"] = \
    df_test["Value_Engine_Displacement_(Cubic_Centimeter)"]/df_test["Value_Replacement_cost_of_insured_vehicle"]

## Feature: Sum_Premium對1 + Sum_Value_plia_acc_and_Value_pdmg_acc的比例
df_test["Ratio_Sum_Premium_to_1+Sum_Value_plia_acc_and_Value_pdmg_acc"] = \
    df_test["Sum_Premium"]/(1 + df_test["Sum_Value_plia_acc_and_Value_pdmg_acc"])

## Feature: iply_area值(轉換成other)
feature_col = "iply_area"
df_test["Category_with_other_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_other_dict[feature_col] else "other")

## Feature: iply_area值(轉換成rare)
feature_col = "iply_area"
df_test["Category_with_rare_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else "rare")

## Feature: Imported_or_Domestic_Car值(21, 22, 23 都視為20)
feature_col = "Imported_or_Domestic_Car"
#overlap_category_not_rare_dict[feature_col] = [x for x in sorted(set(df_policy[feature_col])) if x not in [21, 22, 23]]
df_test["Category_with_rare_" + feature_col] = swiftapply(df_test["Category_" + feature_col],
        lambda x: x if x in overlap_category_not_rare_dict[feature_col] else 20)

## Feature: Sum_Premium對的Value_Engine_Displacement_(Cubic_Centimeter)比例
df_test["Ratio_Sum_Premium_to_Value_Engine_Displacement_(Cubic_Centimeter)"] = \
    df_test["Sum_Premium"]/df_test["Value_Engine_Displacement_(Cubic_Centimeter)"]

## Feature: Sum_Premium對的Value_Manafactured_Year_and_Month比例
df_test["Ratio_Sum_Premium_to_Value_Manafactured_Year_and_Month"] = \
    df_test["Sum_Premium"]/df_test["Value_Manafactured_Year_and_Month"]

## Feature: Sum_Premium對的Value_qpt比例
df_test["Ratio_Sum_Premium_to_Value_qpt"] = \
    df_test["Sum_Premium"]/df_test["Value_qpt"]

## Feature: Sum_Premium取Log
df_test["Log_Sum_Premium"] = \
    np.log1p(df_test["Sum_Premium"])

## Feature: Coverage_Deductible_if_applied金額
df_policy["Coverage_Deductible_if_applied_transformed_value"] = swiftapply(df_policy["Coverage_Deductible_if_applied"], lambda x: 7000 if x == 1 else(\
                                                                     8000 if x == 2 else(10000 if x == 3 else(-1 if x < 0 else x))))
df_policy["Coverage_Deductible_if_applied_transformed_value"] = df_policy.loc[:, ["Coverage_Deductible_if_applied_transformed_value", "Sum_Insured_Amount"]].apply(lambda x: x[0]*x[1]/100 if x[0] in [10, 20] else x[0], axis = 1)

insurance_id = list(df_insurance_type_info.loc[(df_insurance_type_info['自負額說明   【0:無自負額  負數:已退保   其他:詳下列說明】'] == "與自負額無關,請忽略"), :]["險種代號"])
df_policy["Coverage_Deductible_if_applied_transformed_value"][df_policy["Insurance_Coverage"].isin(insurance_id)] = -2

# Feature: Sum_Insured_Amount, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Coverage_Deductible_if_applied_transformed_value"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x[x > 0])).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Sum_" + feature_col, ]
df_test["Sum_" + feature_col].fillna(0, inplace = True)

df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x[x > 0])).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Mean_" + feature_col, ]
df_test["Mean_" + feature_col].fillna(0, inplace = True)

df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x[x > 0])).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Median_" + feature_col, ]
df_test["Median_" + feature_col].fillna(0, inplace = True)

df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x[x > 0])).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Max_" + feature_col, ]
df_test["Max_" + feature_col].fillna(0, inplace = True)

df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x[x > 0])).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Min_" + feature_col, ]
df_test["Min_" + feature_col].fillna(0, inplace = True)

df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x[x > 0])).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Std_" + feature_col, ]
df_test["Std_" + feature_col].fillna(0, inplace = True)

df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x == 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_0_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x == 0)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Percentage_0_" + feature_col, ]

df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x == -1)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_-1_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x == -1)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Percentage_-1_" + feature_col, ]

df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x == -2)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_-2_" + feature_col, ]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x == -2)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Percentage_-2_" + feature_col, ]

## Feature: Coverage_Deductible_if_applied是金額的數量
df_policy["Coverage_Deductible_if_applied_transformed_value"] = swiftapply(df_policy["Coverage_Deductible_if_applied"], lambda x: 7000 if x == 1 else(\
                                                                     8000 if x == 2 else(10000 if x == 3 else(-1 if x < 0 else x))))
insurance_id = list(df_insurance_type_info.loc[(df_insurance_type_info['自負額說明   【0:無自負額  負數:已退保   其他:詳下列說明】'] == "與自負額無關,請忽略"), :]["險種代號"])
df_policy["Coverage_Deductible_if_applied_transformed_value"][df_policy["Insurance_Coverage"].isin(insurance_id)] = -2

feature_col = "Coverage_Deductible_if_applied_transformed_value"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x > 20)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_value_" + feature_col, ]

## Feature: Coverage_Deductible_if_applied是金額的比例
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x > 20)/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Percentage_value_" + feature_col, ]

## Feature: Coverage_Deductible_if_applied是比例的數量
feature_col = "Coverage_Deductible_if_applied_transformed_value"
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum((x == 10)|(x == 20))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Count_percentage_" + feature_col, ]

## Feature: Coverage_Deductible_if_applied是比例的比例
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum((x == 10)|(x == 20))/len(list(x))).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Percentage_percentage_" + feature_col, ]

df_policy["Coverage_Deductible_if_applied_transformed_value"] = swiftapply(df_policy["Coverage_Deductible_if_applied"], lambda x: 7000 if x == 1 else(\
                                                                     8000 if x == 2 else(10000 if x == 3 else(-1 if x < 0 else x))))
df_policy["Coverage_Deductible_if_applied_transformed_value"] = df_policy.loc[:, ["Coverage_Deductible_if_applied_transformed_value", "Sum_Insured_Amount"]].apply(lambda x: x[0]*x[1]/100 if x[0] in [10, 20] else x[0], axis = 1)

insurance_id = list(df_insurance_type_info.loc[(df_insurance_type_info['自負額說明   【0:無自負額  負數:已退保   其他:詳下列說明】'] == "與自負額無關,請忽略"), :]["險種代號"])
df_policy["Coverage_Deductible_if_applied_transformed_value"][df_policy["Insurance_Coverage"].isin(insurance_id)] = -2

## Feature: Sum_Premium對Sum_Coverage_Deductible_if_applied_transformed_value的比例
df_test["Sum_Premium"]/df_test["Sum_Coverage_Deductible_if_applied_transformed_value"]
df_test["Ratio_Sum_Premium_to_Sum_Coverage_Deductible_if_applied_transformed_value"] = \
    df_test.loc[:, ["Sum_Premium", "Sum_Coverage_Deductible_if_applied_transformed_value"]].apply(lambda x: \
                x[0]/x[1] if x[1] > 0 else 0, axis = 1)

## Feature: Count_Policy_Number和Value_Multiple_Products_with_TmNewa_(Yes_or_No?)的總和
df_test["Sum_Count_Policy_Number_and_Value_Multiple_Products_with_TmNewa_(Yes_or_No?)"] = \
    df_test["Count_Policy_Number"] + df_test["Value_Multiple_Products_with_TmNewa_(Yes_or_No?)"]

## Feature: Count_Policy_Number和Value_Multiple_Products_with_TmNewa_(Yes_or_No?)的差距
df_test["Diff_Count_Policy_Number_and_Value_Multiple_Products_with_TmNewa_(Yes_or_No?)"] = \
    df_test["Count_Policy_Number"] - df_test["Value_Multiple_Products_with_TmNewa_(Yes_or_No?)"]  

## Feature: ibirth和dbirth是否一樣
df_policy["Binary_equal_ibirth_and_dbirth"] = (df_policy["ibirth"] == df_policy["dbirth"])*1
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])["Binary_equal_ibirth_and_dbirth"].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test.columns = list(df_test.columns[:-1]) + ["Binary_equal_ibirth_and_dbirth", ]

## Feature: Value_Engine_Displacement_(Cubic_Centimeter)對Manafactured_Year_and_Month的比例
df_test["Ratio_Value_Engine_Displacement_(Cubic_Centimeter)_to_Value_Manafactured_Year_and_Month"] = \
    df_test["Value_Engine_Displacement_(Cubic_Centimeter)"]/df_test["Value_Manafactured_Year_and_Month"]

## Feature: Value_Manafactured_Year_and_Month與今年年份差距和Value_Age_Year_ibirth的差距
df_policy["Age_Manafactured_Year_and_Month"] = 2017 - df_policy["Manafactured_Year_and_Month"]
df_test = df_test.merge(df_policy.groupby(["Policy_Number"])["Age_Manafactured_Year_and_Month"].agg(lambda x: list(x)[0]).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"] = \
    df_test["Value_Age_Year_ibirth"] - df_test["Age_Manafactured_Year_and_Month"]

## Feature: Value_Manafactured_Year_and_Month與今年年份差距和Value_Age_Year_dbirth的差距
df_test["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"] = \
    df_test["Value_Age_Year_dbirth"] - df_test["Age_Manafactured_Year_and_Month"]

df_test.drop(["Age_Manafactured_Year_and_Month"], axis = 1, inplace = True)

## Feature: Prior_Policy_Number的Premium Sum, Mean, Median, Max, Min, Std, 非0數量, 非0比例
feature_col = "Premium"

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.sum(x)))
df_test["Sum_Prior_Policy_Number_" + feature_col] = df_test["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.mean(x)))
df_test["Mean_Prior_Policy_Number_" + feature_col] = df_test["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.median(x)))
df_test["Median_Prior_Policy_Number_" + feature_col] = df_test["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.max(x)))
df_test["Max_Prior_Policy_Number_" + feature_col] = df_test["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.min(x)))
df_test["Min_Prior_Policy_Number_" + feature_col] = df_test["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.std(x)))
df_test["Std_Prior_Policy_Number_" + feature_col] = df_test["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)))
df_test["Count_NOT_0_Prior_Policy_Number_" + feature_col] = df_test["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

mapping_dict_tmp = dict(df_policy.groupby(["Prior_Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(list(x))))
df_test["Percentage_NOT_0_Prior_Policy_Number_" + feature_col] = df_test["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))

### df_claims

## Feature: Policy_Number有無在df_claim的Policy_Number裡面
claim_Policy_Number_list = list(set(df_claim["Policy_Number"]))
df_test["Binary_in_df_claims"] = swiftapply(df_test["Policy_Number"], lambda x: 1 if x in claim_Policy_Number_list else 0)

## Feature: Policy_Number出現在df_claim的Policy_Number裡面的次數
mapping_dict_tmp = dict(df_claim["Policy_Number"].value_counts())
df_test["Count_in_df_claims"] = df_test["Policy_Number"].apply(lambda x: mapping_dict_tmp.get(x, 0))


# -----------------------------------------------------------------------------
### Feature Engineering (with claims)
# -----------------------------------------------------------------------------
df_train_with_claims = df_train.loc[df_train["Binary_in_df_claims"] == 1, :].copy()
df_test_with_claims = df_test.loc[df_test["Binary_in_df_claims"] == 1, :].copy()

# df_train
# -----------------------------------------------------------------------------

## Feature: 各Poicy_Number的獨立Claim_Number的個數
feature_col = "Claim_Number"
mapping_dict_tmp = df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: len(set(x)))
df_train_with_claims["Count_Unique_" + feature_col] = list(swiftapply(df_train_with_claims["Policy_Number"], lambda x: mapping_dict_tmp[x]))

## Feature: 各Poicy_Number的Claim_Number的個數
feature_col = "Claim_Number"
mapping_dict_tmp = df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: len(x))
df_train_with_claims["Count_" + feature_col] = list(swiftapply(df_train_with_claims["Policy_Number"], lambda x: mapping_dict_tmp[x]))

## Feature: 各個Nature_of_the_claim數量
feature_col = "Nature_of_the_claim"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-2]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Nature_of_the_claim比例
feature_col = "Nature_of_the_claim"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-2]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Driver's_Gender數量
feature_col = "Driver's_Gender"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-2]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Driver's_Gender比例
feature_col = "Driver's_Gender"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-2]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Driver's_Relationship_with_Insured數量
feature_col = "Driver's_Relationship_with_Insured"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-7]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Driver's_Relationship_with_Insured比例
feature_col = "Driver's_Relationship_with_Insured"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-7]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: DOB_of_Driver年齡統計值, Mean, Median, Max, Min, Std, 某年齡等於或以上數量, 某年齡等於或以上比例
end_date = "2016/12"
feature_col = "DOB_of_Driver"
specific_age = 65
df_claim["Age_Year_" + feature_col] = (pd.to_datetime(end_date) - pd.to_datetime(df_claim[feature_col])).astype('timedelta64[Y]')

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Mean_" + "Age_Year_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Median_" + "Age_Year_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Max_" + "Age_Year_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Min_" + "Age_Year_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Std_" + "Age_Year_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.sum(x >= specific_age)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Count_" + str(specific_age) + "_Age_Year_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.sum(x >= specific_age)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Percentage_" + str(specific_age) + "_Age_Year_" + feature_col, ]

## Feature: 各個Marital_Status_of_Driver數量
feature_col = "Marital_Status_of_Driver"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-2]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Marital_Status_of_Driver比例
feature_col = "Marital_Status_of_Driver"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-2]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: Accident_Date經過時間統計值, Mean, Median, Max, Min, Std, 某經過時間以上數量, 某經過時間以上比例
end_date = "2017/1"
feature_col = "Accident_Date"
specific_count = 12 # 2015年發生的
df_claim["Diff_Month_" + feature_col] = (pd.to_datetime(end_date) - pd.to_datetime(df_claim[feature_col])).astype('timedelta64[M]')

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Mean_" + "Diff_Month_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Median_" + "Diff_Month_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Max_" + "Diff_Month_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Min_" + "Diff_Month_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Std_" + "Diff_Month_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.sum(x > specific_count)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Count_" + str(specific_count) + "_Diff_Month_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.sum(x > specific_count)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Percentage_" + str(specific_count) + "_Diff_Month_" + feature_col, ]

## Feature: 各個Cause_of_Loss數量
feature_col = "Cause_of_Loss"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-17]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Cause_of_Loss比例
feature_col = "Cause_of_Loss"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-17]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: Paid_Loss_Amount, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Paid_Loss_Amount"

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Sum_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Mean_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Median_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Max_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Min_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Std_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: paid_Expenses_Amount, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "paid_Expenses_Amount"

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Sum_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Mean_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Median_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Max_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Min_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Std_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: Salvage_or_Subrogation?, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Salvage_or_Subrogation?"

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Sum_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Mean_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Median_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Max_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Min_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Std_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: 各個Coverage數量
feature_col = "Coverage"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-48]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Coverage比例
feature_col = "Coverage"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-48]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 每個Policy Number擁有的相異Vehicle_identifier數量
feature_col = "Vehicle_identifier"
mapping_dict_tmp = df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: len(set(x)) - 1 if float in [type(y) for y in list(x)] else len(set(x)))
df_train_with_claims["Count_Unique_" + feature_col] = list(swiftapply(df_train_with_claims["Policy_Number"], lambda x: mapping_dict_tmp[x]))

## Feature: Deductible, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Deductible"

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Sum_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Mean_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Median_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Max_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Min_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Std_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: At_Fault?, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "At_Fault?"

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.nanmean(x) if not np.isnan(np.nanmean(x)) else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Mean_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x:np.nanmean(x) if not np.isnan(np.nanmedian(x)) else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Median_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x:np.nanmean(x) if not np.isnan(np.nanmax(x)) else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Max_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x:np.nanmean(x) if not np.isnan(np.nanmin(x)) else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Min_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x:np.nanmean(x) if not np.isnan(np.nanstd(x)) else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Std_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: 各個Accident_area數量
feature_col = "Accident_area"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-22]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Cause_of_Loss比例
feature_col = "Accident_area"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-22]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: number_of_claimants,  Mean, Median, Max, Min, Std
feature_col = "number_of_claimants"

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Mean_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Median_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Max_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Min_" + feature_col, ]

df_train_with_claims = df_train_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-1]) + ["Std_" + feature_col, ]

## Feature: 各個Accident_Time_new_interval數量
#df_claim["Accident_Time_new"] = df_claim["Accident_Time"].apply(lambda x: datetime.strptime(x, "%H:%M"))
df_claim["Accident_Time_new_interval"] = swiftapply(df_claim["Accident_Time_new"], lambda x: 0 if x < datetime.strptime("12:00", "%H:%M") else(1 if x < datetime.strptime("18:00", "%H:%M") else 2))

feature_col = "Accident_Time_new_interval"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-3]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Accident_Time_new_interval比例
feature_col = "Accident_Time_new_interval"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_train_with_claims = df_train_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_train_with_claims.columns = list(df_train_with_claims.columns[:-3]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

# df_test
# -----------------------------------------------------------------------------

## Feature: 各Poicy_Number的獨立Claim_Number的個數
feature_col = "Claim_Number"
mapping_dict_tmp = df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: len(set(x)))
df_test_with_claims["Count_Unique_" + feature_col] = list(swiftapply(df_test_with_claims["Policy_Number"], lambda x: mapping_dict_tmp[x]))

## Feature: 各Poicy_Number的Claim_Number的個數
feature_col = "Claim_Number"
mapping_dict_tmp = df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: len(x))
df_test_with_claims["Count_" + feature_col] = list(swiftapply(df_test_with_claims["Policy_Number"], lambda x: mapping_dict_tmp[x]))

## Feature: 各個Nature_of_the_claim數量
feature_col = "Nature_of_the_claim"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-2]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Nature_of_the_claim比例
feature_col = "Nature_of_the_claim"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-2]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Driver's_Gender數量
feature_col = "Driver's_Gender"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-2]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Driver's_Gender比例
feature_col = "Driver's_Gender"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-2]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Driver's_Relationship_with_Insured數量
feature_col = "Driver's_Relationship_with_Insured"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-7]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Driver's_Relationship_with_Insured比例
feature_col = "Driver's_Relationship_with_Insured"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-7]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: DOB_of_Driver年齡統計值, Mean, Median, Max, Min, Std, 某年齡等於或以上數量, 某年齡等於或以上比例
end_date = "2016/12"
feature_col = "DOB_of_Driver"
specific_age = 65
df_claim["Age_Year_" + feature_col] = (pd.to_datetime(end_date) - pd.to_datetime(df_claim[feature_col])).astype('timedelta64[Y]')

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Mean_" + "Age_Year_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Median_" + "Age_Year_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Max_" + "Age_Year_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Min_" + "Age_Year_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Std_" + "Age_Year_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.sum(x >= specific_age)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Count_" + str(specific_age) + "_Age_Year_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Age_Year_" + feature_col].agg(lambda x: np.sum(x >= specific_age)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Percentage_" + str(specific_age) + "_Age_Year_" + feature_col, ]

## Feature: 各個Marital_Status_of_Driver數量
feature_col = "Marital_Status_of_Driver"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-2]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Marital_Status_of_Driver比例
feature_col = "Marital_Status_of_Driver"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-2]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: Accident_Date經過時間統計值, Mean, Median, Max, Min, Std, 某經過時間以上數量, 某經過時間以上比例
end_date = "2017/1"
feature_col = "Accident_Date"
specific_count = 12 # 2015年發生的
df_claim["Diff_Month_" + feature_col] = (pd.to_datetime(end_date) - pd.to_datetime(df_claim[feature_col])).astype('timedelta64[M]')

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Mean_" + "Diff_Month_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Median_" + "Diff_Month_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Max_" + "Diff_Month_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Min_" + "Diff_Month_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Std_" + "Diff_Month_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.sum(x > specific_count)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Count_" + str(specific_count) + "_Diff_Month_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])["Diff_Month_" + feature_col].agg(lambda x: np.sum(x > specific_count)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Percentage_" + str(specific_count) + "_Diff_Month_" + feature_col, ]

## Feature: 各個Cause_of_Loss數量
feature_col = "Cause_of_Loss"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-17]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Cause_of_Loss比例
feature_col = "Cause_of_Loss"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-17]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: Paid_Loss_Amount, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Paid_Loss_Amount"

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Sum_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Mean_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Median_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Max_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Min_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Std_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: paid_Expenses_Amount, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "paid_Expenses_Amount"

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Sum_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Mean_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Median_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Max_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Min_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Std_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: Salvage_or_Subrogation?, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Salvage_or_Subrogation?"

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Sum_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Mean_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Median_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Max_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Min_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Std_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: 各個Coverage數量
feature_col = "Coverage"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-48]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Coverage比例
feature_col = "Coverage"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-48]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 每個Policy Number擁有的相異Vehicle_identifier數量
feature_col = "Vehicle_identifier"
mapping_dict_tmp = df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: len(set(x)) - 1 if float in [type(y) for y in list(x)] else len(set(x)))
df_test_with_claims["Count_Unique_" + feature_col] = list(swiftapply(df_test_with_claims["Policy_Number"], lambda x: mapping_dict_tmp[x]))

## Feature: Deductible, Sum, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "Deductible"

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Sum_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Mean_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Median_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Max_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Min_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Std_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: At_Fault?, Mean, Median, Max, Min, Std, 0數量, 0比例
feature_col = "At_Fault?"

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.nanmean(x) if not np.isnan(np.nanmean(x)) else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Mean_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x:np.nanmean(x) if not np.isnan(np.nanmedian(x)) else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Median_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x:np.nanmean(x) if not np.isnan(np.nanmax(x)) else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Max_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x:np.nanmean(x) if not np.isnan(np.nanmin(x)) else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Min_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x:np.nanmean(x) if not np.isnan(np.nanstd(x)) else 0).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Std_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Count_NOT_0_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.sum(x != 0)/len(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Percentage_NOT_0_" + feature_col, ]

## Feature: 各個Accident_area數量
feature_col = "Accident_area"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-22]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Cause_of_Loss比例
feature_col = "Accident_area"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-22]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: number_of_claimants,  Mean, Median, Max, Min, Std
feature_col = "number_of_claimants"

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.mean(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Mean_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.median(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Median_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.max(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Max_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.min(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Min_" + feature_col, ]

df_test_with_claims = df_test_with_claims.merge(df_claim.groupby(["Policy_Number"])[feature_col].agg(lambda x: np.std(x)).reset_index(),
                          on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-1]) + ["Std_" + feature_col, ]

## Feature: 各個Accident_Time_new_interval數量
#df_claim["Accident_Time_new"] = df_claim["Accident_Time"].apply(lambda x: datetime.strptime(x, "%H:%M"))
df_claim["Accident_Time_new_interval"] = swiftapply(df_claim["Accident_Time_new"], lambda x: 0 if x < datetime.strptime("12:00", "%H:%M") else(1 if x < datetime.strptime("18:00", "%H:%M") else 2))

feature_col = "Accident_Time_new_interval"
mapping_df_tmp = df_claim.groupby(["Policy_Number"])[feature_col].value_counts().unstack().reset_index()
mapping_df_tmp = mapping_df_tmp.fillna(0)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-3]) + ["Count_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

## Feature: 各個Accident_Time_new_interval比例
feature_col = "Accident_Time_new_interval"
mapping_df_tmp = pd.concat([mapping_df_tmp["Policy_Number"], mapping_df_tmp.iloc[:, 1:].apply(lambda x: x/np.sum(x), axis = 1)], axis = 1)
df_test_with_claims = df_test_with_claims.merge(mapping_df_tmp, on = ["Policy_Number"], how = "left")
df_test_with_claims.columns = list(df_test_with_claims.columns[:-3]) + ["Percentage_" + str(x) + "_" + feature_col for x in list(mapping_df_tmp.columns)[1:]]

# -----------------------------------------------------------------------------
### Update the data
# -----------------------------------------------------------------------------
file = open(os.path.join(new_data_path, 'df_train_modeling.pickle'), 'wb')
pickle.dump(df_train, file)
file.close()
file = open(os.path.join(new_data_path, 'df_test_modeling.pickle'), 'wb')
pickle.dump(df_test, file)
file.close()

file = open(os.path.join(new_data_path, 'df_train_with_claims_modeling.pickle'), 'wb')
pickle.dump(df_train_with_claims, file)
file.close()
file = open(os.path.join(new_data_path, 'df_test_with_claims_modeling.pickle'), 'wb')
pickle.dump(df_test_with_claims, file)
file.close()

file = open(os.path.join(new_data_path, 'df_policy.pickle'), 'wb')
pickle.dump(df_policy, file)
file.close()

file = open(os.path.join(new_data_path, 'df_claim.pickle'), 'wb')
pickle.dump(df_claim, file)
file.close()