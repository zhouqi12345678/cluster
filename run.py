import argparse
import warnings
warnings.filterwarnings('ignore')
import csv
import random
import numpy as np
import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
import lightgbm as lgb

from src.animate_scatter import AnimateScatter
from src.whale_optimization import WhaleOptimization
from sklearn.model_selection import RepeatedKFold

data = pd.read_csv('G:/桌面/DW/dataset/peng_All.csv')
x = data.iloc[:,0:-1].values
y = data.iloc[:,20].values

def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nsols", type=int, default=50, dest='nsols', help='number of solutions per generation, default: 50')
    parser.add_argument("-ngens", type=int, default=200, dest='ngens', help='number of generations, default: 20')
    parser.add_argument("-a", type=float, default=2.0, dest='a', help='woa algorithm specific parameter, controls search spread default: 2.0')
    parser.add_argument("-b", type=float, default=0.5, dest='b', help='woa algorithm specific parameter, controls spiral, default: 0.5')
    parser.add_argument("-c", type=float, default=None, dest='c', help='absolute solution constraint value, default: None, will use default constraints')
    parser.add_argument("-func", type=str, default='objective_func', dest='func', help='function to be optimized, default: booth; options: matyas, cross, eggholder, schaffer, booth')
    parser.add_argument("-r", type=float, default=0.25, dest='r', help='resolution of function meshgrid, default: 0.25')
    parser.add_argument("-t", type=float, default=0.1, dest='t', help='animate sleep time, lower values increase animation speed, default: 0.1')
    parser.add_argument("-max", default=False, dest='max', action='store_true', help='enable for maximization, default: False (minimization)')

    args = parser.parse_args()
    return args

def objective_func(learning_rate, num_leaves, max_depth, colsample_bytree, subsample, reg_alpha, reg_lambda):
    
    num_leaves = np.array(list(map(int, num_leaves)))
    max_depth = np.array(list(map(int, max_depth)))
    
    mse_scores_means = []
    mse_scores_means_train = []
    
    for i in range(len(num_leaves)):
        if num_leaves[i] > 2**max_depth[i]:
            num_leaves[i] = random.randint(2, 2**max_depth[i])
            
        params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'verbose': -1,
            'learning_rate': learning_rate[i],
            'num_leaves': num_leaves[i],
            'max_depth': max_depth[i],
            'subsample': subsample[i],
            'colsample_bytree': colsample_bytree[i],
            'reg_alpha': reg_alpha[i],
            'reg_lambda': reg_lambda[i], 
        }
        
        rmse_scores = []
        for train_index, test_index in kf.split(x):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # 训练模型
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
    
            # 预测并计算MSE
            y_pred = model.predict(X_test)
            rmse = MSE(y_pred, y_test) ** 0.05
            rmse_scores.append(rmse)
           
        mse_scores_means.append(np.mean(rmse_scores))
    
    return mse_scores_means

def main():
    args = parse_cl_args()

    nsols = args.nsols
    ngens = args.ngens

    constraints = [[0.01, 0.2], [3, 60], [3, 25], [0.5, 1], [0.5, 1], [0,100], [0,100]]

    opt_func = func
    
    b = args.b
    a = args.a
    a_step = a/ngens

    maximize = args.max

    opt_alg = WhaleOptimization(opt_func, constraints, nsols, b, a, a_step, maximize)
    solutions = opt_alg.get_solutions()

    for i in range(ngens):
        print('i = ', i)
        opt_alg.optimize(i)
        solutions = opt_alg.get_solutions()

    b_sol = opt_alg.print_best_solutions()
    print('b_sol = ', b_sol)
    df = pd.DataFrame(b_sol)
    
    df.to_excel('G:/桌面/diedai.xlsx', index=False)

if __name__ == '__main__':
    main()
