# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 18:54:05 2023

@author: Administrator
"""
import argparse
import warnings
warnings.filterwarnings('ignore')
import csv

import time
import random
import numpy as np
import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score as R2S

from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
import lightgbm as lgb
# 导入机器学习库
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import GradientBoostingRegressor as GBDT
from xgboost import XGBRegressor as XGBoost

def calculate_mape(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("实际值和预测值的长度不一致")
    errors = []
    for i in range(len(actual)):
        if actual[i] == 0:  # 避免分母为0的情况
            continue
        error = abs((actual[i] - predicted[i]) / actual[i])
        errors.append(error)

    mape = sum(errors) / len(errors) * 100
    return mape
# # data = pd.read_csv('G:/桌面/DW/dataset/peng999.csv')
x = data.iloc[:,0:-1].values
y = data.iloc[:,20].values

mae_scores = []
rmse_scores = []
r2_scores = []
mape_scores = []

mae_scores_test = []
rmse_scores_test = []
r2_scores_test = []
mape_scores_test = []

params =  {'objective': 'regression', 
            'boosting_type': 'gbdt', 
            'metric': 'rmse', 
            'verbose': -1, 
            'learning_rate': 0.2, 
            'num_leaves': 14, 
            'max_depth': 15, 
            'subsample': 0.9,
            'colsample_bytree': 0.7, 
            'reg_alpha': 0.5, 
            'reg_lambda': 3.4
            } 


mse_scores = []
mse_scores_train = []
i = 0
for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    i = i + 1
    # 训练模型
    model = lgb.LGBMRegressor(**params)
    # model = RF()
    # model = GBDT()
    # model = XGBoost(learning_rate=0.12, max_depth=5, subsample=0.7, colsample_bytree=0.7)
    # model = lgb.LGBMRegressor(learning_rate=0.10, max_depth = 5, subsample=0.7, colsample_bytree=0.8)
    model.fit(X_train, y_train)

    # 预测训练集结果
    y_pred = model.predict(X_train)
    
    mae = MAE(y_pred, y_train)
    mae_scores.append(mae)
    
    rmse = MSE(y_pred, y_train) ** 0.5
    rmse_scores.append(rmse)
    
    r2 = R2S(y_pred, y_train)
    r2_scores.append(r2)
    
    mape = calculate_mape(y_pred, y_train)
    mape_scores.append(mape)
    
    # 预测训练集结果
    y_pred_test = model.predict(X_test)
    
    mae_test = MAE(y_pred_test, y_test)
    mae_scores_test.append(mae_test)
    
    rmse_test = MSE(y_pred_test, y_test) ** 0.5
    rmse_scores_test.append(rmse_test)
    
    r2_test = R2S(y_pred_test, y_test)
    r2_scores_test.append(r2_test)
    
    mape_test = calculate_mape(y_pred_test, y_test)
    mape_scores_test.append(mape_test)  

print(params)
print("训练集：")
print('mae_scores = ', np.mean(mae_scores))
print('rmse_scores = ', np.mean(rmse_scores))
print('r2_scores = ', np.mean(r2_scores))
print('mape_scores = ', np.mean(mape_scores))

# print('mae_scores = ', (mae_scores))
# print('rmse_scores = ', (rmse_scores))
# print('r2_scores = ', (r2_scores))

print("测试集：")
print('mae_scores = ', np.mean(mae_scores_test))
print('rmse_scores = ', np.mean(rmse_scores_test))
print('r2_scores = ', np.mean(r2_scores_test))
print('mape_scores = ', np.mean(mape_scores_test))
print('\n')
