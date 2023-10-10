# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:59:28 2023

@author: xiaoyu
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 11:06:14 2022

@author: 86182
"""
import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

coulumns = ['energy']
for i in range(1,21):
    coulumns.append(f'Au_{i}_x')
    coulumns.append(f'Au_{i}_y')
    coulumns.append(f'Au_{i}_z')
    all_list = []
    
for num in range(1):
    result_list = []
    with open(f'G:\桌面\DW\dataset\Au20_OPT_1000\{num}.xyz') as f1:
    # with open(f'{num}.xyz', 'r+') as f1:
        while True:
            data = f1.readline().strip()
            if data:
                if 'Au' not in data and data != '20':
                    result_list.append(data.split(":")[0])
#                    print(data.split(':')[0])
                elif 'Au' in data:
                    data_list = list(data.split())
                    for item in range(1,4):
                        result_list.append(data_list[item])
                all_list.append(result_list)
            else:
                break


dt = pd.DataFrame(all_list,columns=coulumns)

dt = dt.drop_duplicates()
dt.reset_index(drop = True)


##生成新特征矩阵
l_all=[]
for k in range(1):
    h=['Au_1_x','Au_2_x','Au_3_x','Au_4_x','Au_5_x','Au_6_x','Au_7_x','Au_8_x','Au_9_x','Au_10_x',
       'Au_11_x','Au_12_x','Au_13_x','Au_14_x','Au_15_x','Au_16_x','Au_17_x','Au_18_x','Au_19_x','Au_20_x']
    m=[]
    for t in h:
        dt[t]=np.float64(dt[t])
        m.append(dt.iloc[k,:][t])

    h2=['Au_1_y','Au_2_y','Au_3_y','Au_4_y','Au_5_y','Au_6_y','Au_7_y','Au_8_y','Au_9_y','Au_10_y',
        'Au_11_y','Au_12_y','Au_13_y','Au_14_y','Au_15_y','Au_16_y','Au_17_y','Au_18_y','Au_19_y','Au_20_y']
    n=[]
    for t in h2:
        dt[t]=np.float64(dt[t])
        n.append(dt.iloc[k,:][t])

    h3=['Au_1_z','Au_2_z','Au_3_z','Au_4_z','Au_5_z','Au_6_z','Au_7_z','Au_8_z','Au_9_z','Au_10_z',
        'Au_11_z','Au_12_z','Au_13_z','Au_14_z','Au_15_z','Au_16_z','Au_17_z','Au_18_z','Au_19_z','Au_20_z']
    l=[]
    for t in h3:
        dt[t]=np.float64(dt[t])
        l.append(dt.iloc[k,:][t])
    xyzmatrix = [m,n,l]
    cij = np.zeros((20, 20))
    xyzmatrix=np.transpose([xyzmatrix])  
    for i in range(20):
        for j in range(20):
            if i == j:
                cij[i][j] = 0.5 * 79 ** 2.4 
            else:
                dist = np.linalg.norm(np.array(xyzmatrix[i]) - np.array(xyzmatrix[j]))
                cij[i][j] = 79 * 79 / dist 
  
    # 特征提取            
    # w, v = np.linalg.eig(cij)
    
    # data = cij
    # 创建PCA对象，设置降维后的维度
    # pca = PCA(n_components=1)
    # 对数据进行降维转换
    # w = pca.fit_transform(data)
    # 创建KernelPCA对象
    kpca = KernelPCA(n_components=1, kernel='rbf')  # 随意选择核函数和维度数量
    
    # 进行降维
    transformed_data = kpca.fit_transform(cij)
    print('transformed_data = ', transformed_data)
    
    # 计算贡献率
    explained_variance_ratio = kpca.lambdas_ / np.sum(kpca.lambdas_)
    print('explained_variance_ratio = ', explained_variance_ratio)
    
    # 计算贡献率的累计和
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # 输出贡献率的累计和
    print('cumulative_explained_variance_ratio = ', cumulative_explained_variance_ratio)
#     list_t=list(w)
    
#     l_all.append(list_t)
    
# coulumns = []
# for i in range(1,21):
#     coulumns.append(f't_{i}')
    
# df=pd.DataFrame(np.transpose(l_all).real,coulumns)
# df=np.transpose(df)

# #添加因变量列
# df['energy']=np.float64(dt['energy'])
# ene=np.mean(df['energy']) 
# #print("ene = ", ene) 
# #df['energy']=np.log(list(map(abs,list(df['energy']))))
# # df['energy']=df['energy']-np.mean(df['energy'])
# data=df.copy()
# data = df.copy()
# data.to_csv('G:\\桌面\\DW\dataset\\pengPCA.csv')
