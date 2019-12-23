#-*- coding: utf-8 -*

import numpy as np

from Population import far
from Population import ann
from NeuralNetwork import network
from collections import defaultdict

# =====================================
# Step 0. Input Database
# =====================================
# ANN setting
ATTR_SIZE = 10              # attribute size / input size
HIDDEN_1_SIZE = 10
HIDDEN_2_SIZE = 10
OUTPUT_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 0.03

net = network(ATTR_SIZE, HIDDEN_1_SIZE, HIDDEN_2_SIZE, OUTPUT_SIZE, EPOCHS, LEARNING_RATE)
path = path = 'DataSet/SyntheticData/D3-4.dat'
net.read_data(path)

rhs_range = [11, 14]  # rhs's range
# rhs_range = [1, ATTR_SIZE]

# =====================================
# Step 1 Initialize the FAR & ANN population
# =====================================

# FAR population setting 
FAR_POP_SIZE = 30          # FAR population size
FAR_CROSS_RATE = 0.8        # FAR DNA Crossover rate
FAR_MUTATION_RATE = 0.1     # FAR DNA Mutation rate
FAR_THRESHOLD = 0.83         # FAR threshold

# ANN population setting
ANN_POP_SIZE = 14          # ANN population size
ANN_CROSS_RATE = 0.8        # ANN DNA Crossover rate
ANN_MUTATION_RATE = 0.1     # ANN DNA Crossover rate
ANN_THRESHOLD = 0.85         # ANN DNA Crossover rate

GENERATIONS = 50          # number of iteration

far_pop = far(FAR_POP_SIZE, ATTR_SIZE, rhs_range) 
ann_pop = ann(ANN_POP_SIZE, ATTR_SIZE, HIDDEN_1_SIZE, HIDDEN_2_SIZE, OUTPUT_SIZE)

def pruning(rule_Archive, ann_Archive):
    dict = {} # key 是 RHS， value 是一个 RHS 的所有预测准确率
    rule_dict = {} # key 是 RHS， value 是一个 RHS 的所有rule， index和上面的预测准确率对应起来
    concise_output = [] # 输出的rule
    for i in range(len(rule_Archive)):
        # 如果当前RHS不在dict中，就把当前RHS和当前这个rule的预测准确率记录下来
        # 如果当前RHS不在dict中，就把当前RHS和当前这个rule记录下来，index和上面的预测准确率对应
        if rule_Archive[i][0] in dict.keys():
            dict[rule_Archive[i][0]].append(net.accuracy_measure(rule_Archive[i], ann_Archive[i]))
            rule_dict[rule_Archive[i][0]].append(rule_Archive[i])
        else:
            dict.setdefault(rule_Archive[i][0],[])
            dict[rule_Archive[i][0]].append(net.accuracy_measure(rule_Archive[i],ann_Archive[i]))
            rule_dict.setdefault(rule_Archive[i][0], [])
            rule_dict[rule_Archive[i][0]].append(rule_Archive[i])
    # 遍历字典，找到每个key下最大的预测准确率的rule的index
    # 根据index， 再去rule_dict中找rule，存入concise_output
    for i in dict:
        idx = dict[i].index(max(dict[i]))
        concise_output.append(rule_dict[i][idx])
    return concise_output

def fuzzycurve(individual, rule_Archive, ann, threshold):
    if len(rule_Archive) == 0:
        return individual
    simple_Individual = individual
    b = 0.1  # we typically take b as about 10% of the length of the input interval of Xi, since the input is 0 or 1
    # so b is 0.1
    C = np.zeros([len(rule_Archive), len(individual) - 1])
    # 这里的三重for循环是进行fuzzy membership function的计算 和 fuzzy rule的获取
    for i in range(len(rule_Archive)):
        for j in range(len(individual) - 1):
            molecule = 0
            denominator = 0
            for k in range(len(rule_Archive)):
                molecule += np.exp(-(((rule_Archive[k][j + 1] - rule_Archive[i][j + 1]) / b) ** 2)) * \
                            rule_Archive[k][0]
                denominator += np.exp(-(((rule_Archive[k][j + 1] - rule_Archive[i][j + 1]) / b) ** 2))
            C[i][j] = molecule / denominator
    # 这里转置一下，每一行都是一个input variable的函数值
    C_Transform = C.T  # 0 is yes , 1 is no. if a input is flat , then we make the input in indivisual be 1
    C_Range = []
    # 记录所有input variable的range
    for i in range(len(C_Transform)):
        C_Range.append(max(C_Transform[i]) - min(C_Transform[i]))

    # 把与最大值的比值在0.2之下的给去掉
    mx = max(C_Range)
    if mx == 0:
        return individual
    for i in range(len(C_Range)):
        if C_Range[i] / mx < 0.1:
            simple_Individual[i + 1] = 1  # 使该位数值为1， 0为出现，1为未出现
    # 确保去掉之后预测准确率还在阈值之上
    if net.accuracy_measure(simple_Individual, ann) > threshold:
        return simple_Individual

    return individual

# print(far_pop.pop)
# print(ann_pop.pop)

# =====================================
# Step 2 Evolution of the FAR & ANN population
# 
# while Stopping criteria is not met do
#     Mutual evaluation; 
#     Archive valid solutions; 
#     FAR new population selection, crossover, mutation; 
#     ANN new population crossover, mutation, selection;
# end
# =====================================

# 测试专用
# 测试 far[0]和ann[0]的准确率
# c = net.accuracy_measure(far_pop.pop[0], ann_pop.pop[0])

for k in range(GENERATIONS): 
    # Mark
    print('\n\n==== Generation', k+1, '====')
    # Step 2.1  Mutual Evaluation & Archive Valid Solutions;
    print('  Step 1. Mutual Evaluation')
    c = np.zeros((len(far_pop.pop), len(ann_pop.pop)))
    for i in range(len(far_pop.pop)):   
        print('    Calculating accuracy c_'+ str(i) + '_j')     
        for j in range(len(ann_pop.pop)):
            c[i, j] = net.accuracy_measure(far_pop.pop[i], ann_pop.pop[j])
            
    for i in range(len(far_pop.pop)):        
        c_ri = np.max(c[i])
        list_temp = c[i].copy().tolist()
        max_index = list_temp.index(max(list_temp))
        ann = ann_pop.pop[max_index]
        far_pop.cal_fitness(i, c_ri, ann, FAR_THRESHOLD)
        if c_ri > FAR_THRESHOLD:
            far_pop.archive.append(fuzzycurve((far_pop.pop[i]).copy(),far_pop.archive.copy(), ann, FAR_THRESHOLD))
            print('    ** FAR archive update: ', far_pop.pop[i], 'FAR\'s Accuracy =', c_ri)

    for j in range(len(ann_pop.pop)):
        c_aj = np.max(c[:, j])
        best_i = np.where(c[:,j] == c_aj)[0][0]
        d_aj = far_pop.distance_measure(best_i)
        ann_pop.cal_fitness(j, c_aj, d_aj, ANN_THRESHOLD)

    # Step 2.2 FAR population update
    print('  Step 2. FAR population updating...')
    for i in range(len(far_pop.pop)):
        far_pop.genetic_algorithm(i, FAR_CROSS_RATE, FAR_MUTATION_RATE)

    # Step 2.3 ANN population update
    print('  Step 3. ANN population updating...')
    for j in range(len(ann_pop.pop)):
        # (1) Get a trial individual ac & its parent's index
        (parent_index, ac) = ann_pop.de_trial(ANN_CROSS_RATE, ANN_MUTATION_RATE)    
        # (2) Calculate ac's fitness
        c_ac = 0
        best_i = 0
        for i in range(len(far_pop.pop)):     
            c = net.accuracy_measure(far_pop.pop[i], ac)
            if c > c_ac:
                c_ac = c
                best_i = i
        d_ac = far_pop.distance_measure(best_i)
        ac_fitness = c_ac / pow(d_ac, 2)
        # (3) Selection
        ann_pop.de_selection(parent_index, ac, ac_fitness)
        
    # Check Archive
    print('\nFAR Population', k+1)
    print(far_pop.pop)
    print('\nFAR Archive of Generation', k+1)
    print(far_pop.archive)

print('after pruning:')
print(pruning(far_pop.archive.copy(), ann_pop.archive.copy()))
