import numpy as np
import pandas as pd
import seaborn as sb

# A synthetic data set
# Formed by 20 continuous variables ([x1, x2, ... , x20]) 
# With a size of 500 instances.
# 1. Select the IVs from the first ten variables ([x1, x2, ... , x10]) 
# 2. The DVs are from the rest ten variables ([x11, x12, ... , x20]).
dataset_size = 500
var_num = 20
iv_num = 10
dv_num = var_num - 10
SyntheticData = np.zeros([dataset_size, var_num])

for i in range(dataset_size):
    for j in range(iv_num):
        SyntheticData[i][j] = np.random.rand()

# Next, form the polynomial function 
# and replace the values of the DV variable with those calculated from the function.

readme = open('../DataSet/SyntheticData/Readme.txt', 'a')

for i in range(1, 6):              # order of function
    for j in range(1, 11):         # number of function
        name = 'D' + str(i) + '-' + str(j)    
        path = '../DataSet/SyntheticData/' + name + '.dat'   
        readme.write('\n \n合成数据集' + name)

        SyntheticData[:, 10:20] = 0
        inputVariable = [1,2,3,4,5,6,7,8,9,10]
        outputVariable = [11,12,13,14,15,16,17,18,19,20]
        for k in range(j):                     
            # print('\n这是第', k+1, '条hidden function')
            # print('inputVariable为: ' , inputVariable)
            
            IV = []
            for p in range(i):
                if len(inputVariable) == 0:
                    inputVariable.clear()
                    inputVariable = [1,2,3,4,5,6,7,8,9,10]
                IV.append((np.random.choice(inputVariable, 1))[0])
                inputVariable.remove(IV[p])   

            DV = (iv_num + 1) + k
            SyntheticData[:, DV - 1] += 1
            for element in IV:
                x = SyntheticData[:,element-1]
                order = (np.argwhere(IV == element)[0][0] + 1)
                SyntheticData[:, DV - 1] += x ** order

            delimiter = ','
            readme.write('\n  Hidden Function 1 : \n    IV : ' + str(IV) + '\n    DV : ' + str(DV))
                         
        np.savetxt(path, SyntheticData, fmt='%lf', delimiter=',')
        
        print('\n \n 已生成合成数据集', name)