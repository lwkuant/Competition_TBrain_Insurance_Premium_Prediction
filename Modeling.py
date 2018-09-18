# -*- coding: utf-8 -*-
"""
1. Preprocessing
2. Building model
3. Making predictions
"""

# -----------------------------------------------------------------------------
### Initial Setup
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os 
import re
from time import time
from functools import reduce
from collections import defaultdict
from collections import Counter
from itertools import chain
import pickle
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import random as rn

seed = 12345
np.random.seed(seed)

from sklearn.preprocessing import StandardScaler

### keras
from keras.layers import *
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import PReLU
import keras.backend as K
from keras.models import Model
from keras.models import model_from_yaml

raw_data_path = "..."
new_data_path = "..."
model_path = "..."

regression_target = "Next_Premium"
classification_target = "Churn"
n_split = 5

# -----------------------------------------------------------------------------
### Helper Function
# -----------------------------------------------------------------------------

### Feature selection(Predictive power): Classification, Regression
def feature_selection(df, feature_list, target, task_type = "Classification", model = LogisticRegression()):
    if task_type == "Classification":
        feature_score_list = []
        for ind, feature in enumerate(feature_list):         
            print(ind)
            model.fit(df.loc[:, feature].values.reshape([-1, 1]), df[target].values)
            y_pred = model.predict_proba(df.loc[:, feature].values.reshape([-1, 1]))
            score = roc_auc_score(df[target].values, y_pred[:, 1])
            feature_score_list.append([feature, score])
            
        return sorted(feature_score_list, key = lambda x: x[1], reverse = True)
    
    elif task_type == "Regression":
        feature_score_list = []
        for ind, feature in enumerate(feature_list):
            print(ind)
            model.fit(df.loc[:, feature].values.reshape([-1, 1]), df[target].values)
            y_pred = model.predict(df.loc[:, feature].values.reshape([-1, 1]))
            score = mean_absolute_error(df[target].values, y_pred)
            feature_score_list.append([feature, score])
            
        return sorted(feature_score_list, key = lambda x: x[1])   

### Get the grouped data for embedding
def get_grouped_data_for_embedding(df, df_name, feature_category_embedding_list, feature_not_category_embedding_list):
    
    input_train_list = []
    
    # Embedding feature
    for feature in feature_category_embedding_list:
        input_train_list.append(df[feature].map(embedding_dict[df_name][feature]).values)
        
    # Other feature
    input_train_list.append(df[feature_not_category_embedding_list].values)
    
    return input_train_list

def get_category_embedding_data(df, df_name, feature_category_embedding_list,
                                embedding_dict, weights_list):
    
    array_list_tmp = []
    column_list = []
    for ind, feature in enumerate(feature_category_embedding_list):
        array_list_tmp.append(weights_list[ind*2][0][df[feature].map(embedding_dict[df_name][feature]).values, :])
       
        column_list.extend(["Category_Embedding_" + feature + "_" + str(x) for x in range(weights_list[ind*2][0].shape[1])])
        
    df_category_embedding = pd.DataFrame(np.concatenate(array_list_tmp, axis = 1))
    df_category_embedding.columns = column_list
    
    return df_category_embedding

# -----------------------------------------------------------------------------
### load data
# -----------------------------------------------------------------------------

with open(os.path.join(new_data_path, 'df_train_modeling.pickle'), 'rb') as file:
    df_train = pickle.load(file)

with open(os.path.join(new_data_path, 'df_test_modeling.pickle'), 'rb') as file:
    df_test = pickle.load(file)

with open(os.path.join(new_data_path, 'df_train_with_claims_modeling.pickle'), 'rb') as file:
    df_train_with_claims_modeling = pickle.load(file)

with open(os.path.join(new_data_path, 'df_test_with_claims_modeling.pickle'), 'rb') as file:
    df_test_with_claims_modeling = pickle.load(file) 

df_exception_dbirth = pd.read_csv(os.path.join(new_data_path, 'wrong_dbirth.csv'), header = None)
df_exception_duplicate_policy = pd.read_csv(os.path.join(new_data_path, 'duplicate_policy.csv'), header = None)

# -----------------------------------------------------------------------------
### Preprocessing
# -----------------------------------------------------------------------------

### Remove instances with claims
exception_list = list(set(df_train_with_claims_modeling["Policy_Number"]))
df_train = df_train.loc[~df_train["Policy_Number"].isin(exception_list), :].copy()
df_train.index = list(range(df_train.shape[0]))

### Exception
exception_list = list(set(list(df_exception_dbirth[0]) + list(df_exception_duplicate_policy[0])))
df_train = df_train.loc[~df_train["Policy_Number"].isin(exception_list), :].copy()
df_train.index = list(range(df_train.shape[0]))

### Remove problemic rows
df_train = df_train.loc[df_train["Category_fassured"] != 6, :].copy()
df_train.index = range(df_train.shape[0])

Counter([x.split("_")[0] for x in df_train.columns[3:]])

### Shuffle the data
np.random.seed(seed)
shuffle_index = np.random.choice(range(df_train.shape[0]), df_train.shape[0], replace = False)

df_train = df_train.iloc[shuffle_index, :].copy()
df_train.index = range(df_train.shape[0])

### Remove problemtic features
feature_remove_list = []

## Numerical Feature

# No variation(std = 0)
feature_remove_list.extend(list(np.union1d(list(df_train.columns[3:][df_train.dtypes[3:] != "object"][df_train[list(df_train.columns[3:][df_train.dtypes[3:] != "object"])].std() == 0]),
           list(df_test.columns[3:][df_test.dtypes[3:] != "object"][df_test[list(df_test.columns[3:][df_test.dtypes[3:] != "object"])].std() == 0]))))

# Categorical Feature(all the same category)
feature_remove_list.extend(list(reduce(np.union1d, [list(np.array([x for x in df_train.columns[3:] if "Binary" in x])[(df_train[[x for x in df_train.columns[3:] if "Binary" in x]] == 0).sum() == 0]),
    list(np.array([x for x in df_train.columns[3:] if "Binary" in x])[(df_train[[x for x in df_train.columns[3:] if "Binary" in x]] == 1).sum() == 0]),
    list(np.array([x for x in df_test.columns[3:] if "Binary" in x])[(df_test[[x for x in df_test.columns[3:] if "Binary" in x]] == 0).sum() == 0]),
    list(np.array([x for x in df_test.columns[3:] if "Binary" in x])[(df_test[[x for x in df_test.columns[3:] if "Binary" in x]] == 1).sum() == 0])])))

# 
feature_remove_list = list(set(feature_remove_list))

### Outlier

## Clip age to 18 ~ 100
df_train["Value_Age_Year_ibirth"] = df_train["Value_Age_Year_ibirth"].apply(lambda x: x if np.isnan(x) else(18 if x < 18 else(100 if x > 100 else x)))
df_train["Value_Age_Year_dbirth"] = df_train["Value_Age_Year_dbirth"].apply(lambda x: x if np.isnan(x) else(18 if x < 18 else(100 if x > 100 else x)))
df_test["Value_Age_Year_ibirth"] = df_test["Value_Age_Year_ibirth"].apply(lambda x: x if np.isnan(x) else(18 if x < 18 else(100 if x > 100 else x)))
df_test["Value_Age_Year_dbirth"] = df_test["Value_Age_Year_dbirth"].apply(lambda x: x if np.isnan(x) else(18 if x < 18 else(100 if x > 100 else x)))

# -----------------------------------------------------------------------------
### Sift the data
# -----------------------------------------------------------------------------

### Remove problemtic features
df_train = df_train.loc[:, [x for x in df_train.columns if x not in feature_remove_list]].copy()
df_test = df_test.loc[:, [x for x in df_test.columns if x not in feature_remove_list]].copy()

# -----------------------------------------------------------------------------
### Copy the data
# -----------------------------------------------------------------------------

df_train_ = df_train.copy()
df_train = df_train_.copy()

# -----------------------------------------------------------------------------
### Initial Feature List
# -----------------------------------------------------------------------------
feature_initial_non_binary_list = [x for x in df_train.columns[3:] if np.sum([y in x for y in ["Category", "Binary"]]) == 0]
feature_initial_binary_list = [x for x in df_train.columns[3:] if np.sum([y in x for y in ["Binary"]]) != 0]
feature_initial_category_list = [x for x in df_train.columns[3:] if np.sum([y in x for y in ["Category"]]) != 0]

feature_category_embedding_list = ['Category_with_rare_Vehicle_Make_and_Model1',
                 'Category_with_rare_Vehicle_Make_and_Model2',
                 'Category_with_rare_Coding_of_Vehicle_Branding_&_Type',
                 'Category_with_rare_Distribution_Channel',
                 'Category_with_rare_aassured_zip']

feature_not_category_embedding_list = [x for x in df_train.columns[3:] if np.sum([y in x for y in ["Category", "Vehicle_Make_and_Model1", "Vehicle_Make_and_Model2",
                                                                 "Coding_of_Vehicle_Branding_&_Type", "Distribution_Channel", "aassured_zip"]]) == 0]


# -----------------------------------------------------------------------------
### Modeling
# -----------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------------------------------
### Imputation dictionary
imputation_ind_dict = defaultdict(dict)
imputation_value_dict = defaultdict(dict)

## imputation_ind_dict
imputation_ind_dict["df_train"]["Value_Age_Year_ibirth"] = df_train["Value_Age_Year_ibirth"].isnull()
imputation_ind_dict["df_train"]["Value_Age_Year_dbirth"] = df_train["Value_Age_Year_dbirth"].isnull()
imputation_ind_dict["df_train"]["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"] = df_train["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"].isnull()
imputation_ind_dict["df_train"]["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"] = df_train["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"].isnull()

imputation_ind_dict["df_test"]["Value_Age_Year_ibirth"] = df_test["Value_Age_Year_ibirth"].isnull()
imputation_ind_dict["df_test"]["Value_Age_Year_dbirth"] = df_test["Value_Age_Year_dbirth"].isnull()
imputation_ind_dict["df_test"]["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"] = df_test["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"].isnull()
imputation_ind_dict["df_test"]["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"] = df_test["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"].isnull()

## imputation_value_dict
imputation_value_dict["df_train"]["Value_Age_Year_ibirth"] = df_train["Value_Age_Year_ibirth"].median()
imputation_value_dict["df_train"]["Value_Age_Year_dbirth"] = df_train["Value_Age_Year_dbirth"].median()
imputation_value_dict["df_train"]["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"] = df_train["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"].median()
imputation_value_dict["df_train"]["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"] = df_train["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"].median()

imputation_value_dict["df_test"]["Value_Age_Year_ibirth"] = df_train["Value_Age_Year_ibirth"].median()
imputation_value_dict["df_test"]["Value_Age_Year_dbirth"] = df_train["Value_Age_Year_dbirth"].median()
imputation_value_dict["df_test"]["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"] = df_train["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"].median()
imputation_value_dict["df_test"]["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"] = df_train["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"].median()

## Imputation
df_train["Value_Age_Year_ibirth"][imputation_ind_dict["df_train"]["Value_Age_Year_ibirth"]] = imputation_value_dict["df_train"]["Value_Age_Year_ibirth"]
df_train["Value_Age_Year_dbirth"][imputation_ind_dict["df_train"]["Value_Age_Year_dbirth"]] = imputation_value_dict["df_train"]["Value_Age_Year_dbirth"]
df_train["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"][imputation_ind_dict["df_train"]["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"]] = imputation_value_dict["df_train"]["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"]
df_train["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"][imputation_ind_dict["df_train"]["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"]] = imputation_value_dict["df_train"]["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"]

df_test["Value_Age_Year_ibirth"][imputation_ind_dict["df_test"]["Value_Age_Year_ibirth"]] = imputation_value_dict["df_test"]["Value_Age_Year_ibirth"]
df_test["Value_Age_Year_dbirth"][imputation_ind_dict["df_test"]["Value_Age_Year_dbirth"]] = imputation_value_dict["df_test"]["Value_Age_Year_dbirth"]
df_test["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"][imputation_ind_dict["df_test"]["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"]] = imputation_value_dict["df_test"]["Diff_Value_Age_Year_ibirth_and_Age_Manafactured_Year_and_Month"]
df_test["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"][imputation_ind_dict["df_test"]["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"]] = imputation_value_dict["df_test"]["Diff_Value_Age_Year_dbirth_and_Age_Manafactured_Year_and_Month"]

## Normalization
scaler = StandardScaler()
df_train = pd.concat([df_train.iloc[:, :3], 
                         pd.DataFrame(scaler.fit_transform(df_train.loc[:, feature_initial_non_binary_list]), columns = feature_initial_non_binary_list),
                         df_train.loc[:, feature_initial_binary_list],
                         df_train.loc[:, feature_initial_category_list]], axis = 1)
            
df_test = pd.concat([df_test.iloc[:, :2], 
                         pd.DataFrame(scaler.transform(df_test.loc[:, feature_initial_non_binary_list]), columns = feature_initial_non_binary_list),
                         df_test.loc[:, feature_initial_binary_list],
                         df_test.loc[:, feature_initial_category_list]], axis = 1)

## Feature Selection

# Predictive Power
feature_regression_not_category_embedding_list = feature_selection(df_train, [x for x in feature_not_category_embedding_list if "Binary" not in x], regression_target, "Regression", LinearRegression())

# Correlation within features
feature_regression_not_category_embedding_list_complete = [x[0] for x in feature_regression_not_category_embedding_list][::-1]
feature_regression_not_category_embedding_list_tmp = [x[0] for x in feature_regression_not_category_embedding_list][::-1]

correlation_matrix_within_features = df_train[feature_regression_not_category_embedding_list_complete].corr() # 80s

correlation_matrix_within_features_tmp = correlation_matrix_within_features.copy()

threshold_corr_between_features = 0.8
for ind, feature in enumerate(feature_regression_not_category_embedding_list_complete):
    if np.sum(np.abs(correlation_matrix_within_features_tmp[feature]) > threshold_corr_between_features) >= 2:
        feature_regression_not_category_embedding_list_tmp.remove(feature)
        correlation_matrix_within_features_tmp = correlation_matrix_within_features_tmp.loc[\
            feature_regression_not_category_embedding_list_tmp, feature_regression_not_category_embedding_list_tmp]
        
    print(ind)

feature_regression_not_category_embedding_list_tmp = feature_regression_not_category_embedding_list_tmp[::-1] + [x for x in feature_not_category_embedding_list if "Binary" in x]

## Modeling Testing: Deep Learning
model_list = []

num_embedding_max = 100
epochs = 10
feature_list = feature_regression_not_category_embedding_list_tmp[:]
n_split = 5
skf = StratifiedKFold(n_splits = n_split, random_state = seed)
skf_ind_list = list(skf.split(df_train, df_train[classification_target]))
for train_index, test_index in skf_ind_list:
    embedding_dict = defaultdict(dict)
    for feature in feature_category_embedding_list:
        
        category_list_tmp = sorted(set(df_train.loc[train_index, :][feature]))
        category_mapping_dict = dict()
        
        for ind, category in enumerate(category_list_tmp):
            category_mapping_dict[category] = ind
            
        embedding_dict["df_train"][feature] = category_mapping_dict
        
        
    start_time = time()
    df_training_list = get_grouped_data_for_embedding(df_train.loc[train_index, :], "df_train", feature_category_embedding_list, feature_list)
    df_testing_list = get_grouped_data_for_embedding(df_train.loc[test_index, :], "df_train", feature_category_embedding_list, feature_list)
    print(time() - start_time)
    
    models = []
    
    for feature in feature_category_embedding_list:
        model = Sequential()
        num_of_category = df_train.loc[train_index, :][feature].nunique()
        embedding_size = min(np.ceil((num_of_category)/2), num_embedding_max)
        embedding_size = int(embedding_size)
        model.add(Embedding(num_of_category, embedding_size, input_length = 1) )
        model.add(Reshape(target_shape = (embedding_size,)))
        models.append(model)
        
    model_rest = Sequential()
    model_rest.add(Dense(32, input_dim = df_training_list[-1].shape[1]))
    models.append(model_rest)
    
    full_model = Sequential()
    full_model.add(Merge(models, mode='concat'))
    full_model.add(Dense(1000))
    full_model.add(Activation('relu'))
    full_model.add(Dropout(0.3))
    full_model.add(Dense(500))
    full_model.add(Activation('relu'))
    full_model.add(Dropout(0.1))
    full_model.add(Dense(1))
    full_model.add(Activation('relu'))
    
    full_model.compile(loss = 'mean_absolute_error', optimizer = 'adam') 
    
    y_train = df_train.loc[train_index, :][regression_target].values
    y_test = df_train.loc[test_index, :][regression_target].values
    
    history = full_model.fit(df_training_list, y_train,\
            validation_data = (df_testing_list, y_test), epochs = epochs, shuffle = True, 
            batch_size = 32)
   
    model_list.append(full_model)

# -----------------------------------------------------------------------------
### Predicting
# -----------------------------------------------------------------------------

## Modeling
    
# Deep Learning
# -----------------------------------------------------------------------------
# Transform category to number
embedding_dict = defaultdict(dict)

for feature in feature_category_embedding_list:
    
    category_list_tmp = sorted(set(df_train[feature]))
    category_mapping_dict = dict()
    
    for ind, category in enumerate(category_list_tmp):
        category_mapping_dict[category] = ind
        
    embedding_dict["df_train"][feature] = category_mapping_dict

# Run NN with category embedding     
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed)
rn.seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(seed)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)      
        
num_embedding_max = 100
epochs = 10
feature_list = feature_regression_not_category_embedding_list_tmp[:]

start_time = time()
df_train_list = get_grouped_data_for_embedding(df_train, "df_train", feature_category_embedding_list, feature_list)
df_test_list = get_grouped_data_for_embedding(df_test, "df_train", feature_category_embedding_list, feature_list)
print(time() - start_time)

models = []

for feature in feature_category_embedding_list:
    model = Sequential()
    num_of_category = df_train[feature].nunique()
    embedding_size = min(np.ceil((num_of_category)/2), num_embedding_max)
    embedding_size = int(embedding_size)
    model.add(Embedding(num_of_category, embedding_size, input_length = 1 ) )
    model.add(Reshape(target_shape = (embedding_size,)))
    models.append(model)
    
model_rest = Sequential()
model_rest.add(Dense(32, input_dim = df_train_list[-1].shape[1]))
models.append(model_rest)

full_model = Sequential()
full_model.add(Merge(models, mode='concat'))
full_model.add(Dense(1000))
full_model.add(Activation('relu'))
full_model.add(Dropout(0.3))
full_model.add(Dense(500))
full_model.add(Activation('relu'))
full_model.add(Dropout(0.1))
full_model.add(Dense(1))
full_model.add(Activation('relu'))

full_model.compile(loss = 'mean_absolute_error', optimizer = 'adam')

y_train = df_train[regression_target].values

history = full_model.fit(df_train_list, y_train, epochs = epochs)

model_name = "mdoel_no_claims_regression_final"
model_weights_name = "mdoel_no_claims_weights_regression_final"
data_name = "df_test_list_no_claims_regression_final"

model_name = "mdoel_all_regression_final"
model_weights_name = "mdoel_all_weights_regression_final"
data_name = "df_test_list_all_regression_final"

model_version = "v1"

model_yaml = full_model.to_yaml()
with open(os.path.join(model_path, model_name + "_" + model_version + "_"  + ".yaml"), "w") as yaml_file:
    yaml_file.write(model_yaml)

full_model.save_weights(os.path.join(model_path, model_weights_name + "_" + model_version + ".h5"))

file = open(os.path.join(model_path, data_name + "_" + model_version + ".pickle"), 'wb')
pickle.dump(df_test_list, file)
file.close()

# -----------------------------------------------------------------------------
### Submission
# -----------------------------------------------------------------------------

model_dict = {}
data_dict = {}

###
model_name = "mdoel_no_claims_regression_final"
model_weights_name = "mdoel_no_claims_weights_regression_final"
data_name = "df_test_list_no_claims_regression_final"
model_version = "v1"

yaml_file = open(os.path.join(model_path, model_name + "_" + model_version + "_"  + ".yaml"), "r")
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model_dict[model_name] = model_from_yaml(loaded_model_yaml)

model_dict[model_name].load_weights(os.path.join(model_path, model_weights_name + "_" + model_version + ".h5"))
model_dict[model_name].compile(loss = 'mean_absolute_error', optimizer = 'adam')  
model_dict[model_name] = model_dict[model_name]

with open(os.path.join(model_path, data_name + "_" + model_version + ".pickle"), 'rb') as file:
    data_dict[model_name] = pickle.load(file)  

###
model_name = "mdoel_all_regression_final"
model_weights_name = "mdoel_all_weights_regression_final"
data_name = "df_test_list_all_regression_final"
model_version = "v1"

yaml_file = open(os.path.join(model_path, model_name + "_" + model_version + "_"  + ".yaml"), "r")
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model_dict[model_name] = model_from_yaml(loaded_model_yaml)

model_dict[model_name].load_weights(os.path.join(model_path, model_weights_name + "_" + model_version + ".h5"))
model_dict[model_name].compile(loss = 'mean_absolute_error', optimizer = 'adam')  
model_dict[model_name] = model_dict[model_name]

with open(os.path.join(model_path, data_name + "_" + model_version + ".pickle"), 'rb') as file:
    data_dict[model_name] = pickle.load(file) 

###
ind_claims = df_test["Policy_Number"].isin(list(df_test_with_claims_modeling["Policy_Number"]))
y_test = model_dict["mdoel_no_claims_regression_final"].predict(data_dict["mdoel_no_claims_regression_final"])
y_test = y_test.reshape(len(y_test))

y_test_ = model_dict["mdoel_all_regression_final"].predict(data_dict["mdoel_all_regression_final"])
y_test_ = y_test_.reshape(len(y_test_))
y_test[ind_claims] = y_test_[ind_claims]

df_submission = pd.read_csv(os.path.join(raw_data_path, "testing-set.csv"))
df_submission["Next_Premium"] = y_test
df_submission.to_csv(os.path.join(submission_data_path, "df_submission_v12.csv"), index = None)




        





        
