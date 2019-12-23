#-*- coding: utf-8 -*

import numpy as np
import random
import torch
from NeuralNetwork import network

# =====================================
# Part. FAR Population
# =====================================
class far:
    pop = []
    fitness= []
    archive = []

    # temporary var for SyntheticData
    rhs_range = []

    def __init__(self, pop_size, attr_size, rhs_range):
        # desc：   该函数用于初始化一个FAR种群。
        #          FAR染色体的表示形式为 r =[o, e1, e2, e3, ... , ek, ..., en]. 
        #          第一个基因 o 表明哪个变量是规则的RHS，而其余基因 e1, ..., en 表明变量出现在规则的LHS上。
        #          LHS中，每一个属性我们随机决定它存在(1)或者不存在(0)
        #          RHS是预测变量的下标，取值1-n
        # args:    pop_size,    种群大小
        #          attr_size,   属性数量
        #          rhs_range    rhs的取值范围，本应是1-n, 为了测试合成数据集，略作修改
        self.pop = np.zeros((pop_size, attr_size + 1), dtype=int)
        self.fitness = np.zeros(pop_size)
        self.rhs_range = rhs_range            
        for i in range(pop_size):
            # first gene o indicates which variable is the RHS of the rule
            # e1, ..., en are binary values indicating whether the corresponding variables are in the LHS
            self.pop[i][0] = np.random.randint(rhs_range[0], rhs_range[1] + 1)
            for j in range(attr_size):
                self.pop[i][j + 1] = np.random.randint(2)
    def cal_fitness(self, i, c_ri, ann, threshold):
        # desc:    该函数用于计算一个FAR的适应度。
        #          FAR适应度的计算根据文章 (2) - (6)式。
        #          对于以每一条ANN染色体Aj构建的ANN，应用BP算法训练从以确定预测准确度c。
        #          采用均方误差作为BP训练的误差函数。
        # args:    i            需要计算适应度的FAR在种群中的下标
        #          c_ri         第i条FAR染色体在所有ANN个体中测试得到的最佳准确度
        #          threshold    阈值
        # returns: 无
        d_ri = self.distance_measure(i)
        fitness = c_ri / pow(d_ri, 2)

        # Archive valid solutions;
        #if c_ri > threshold:
        #    self.archive.append(self.fuzzycurve((self.pop[i]).copy(),self.archive.copy(), ann, threshold))
        #    print('    ** FAR archive update: ', self.pop[i], 'FAR\'s Accuracy =', c_ri)

        self.fitness[i] = fitness
        # return fitness
    
    def distance_measure(self, index):
        # desc:    计算一条FAR的距离测量d。将其单独取出计算，方便ann fitness计算时调用。
        # args:    index        需要计算适应度的FAR在种群中的下标
        # returns: d_rule       距离测量d
        
        rule_i = self.pop[index]
        d_pop = 0
        d_arc = 0

        for i in range(len(self.pop)):
            if i != index:
                d_pop += self.hamming_distance(rule_i, self.pop[i])
        d_pop /= (len(self.pop) - 1) 

        for i in range(len(self.archive)):
            d_arc += self.hamming_distance(rule_i, self.archive[i])
        if len(self.archive) > 0:
            d_arc /= len(self.archive)
        
        d_ri = 1 / (1 + d_pop + d_arc)  
        return d_ri
    
    def hamming_distance(self, rule_1, rule_2):
        # desc:    计算两个FAR之间的 Hamming Distance
        # args:    rule_1, rule_2, 两个FAR数组
        # returns: distance, 两者之间的 Hamming Distance
        distance  = 0
        for i in range(len(rule_1)):
            if rule_1[i] != rule_2[i]:
                distance += 1
        return distance

    def genetic_algorithm(self, index, cross_rate, mutation_rate):
        # desc:    该函数为FAR种群的进化过程
        # args:    index,         当前进化的FAR的数组下标
        #          cross_rate,    交叉因子
        #          mutation_rate, 变异因子
        # returns: 无
        
        # selection
        candidate = np.random.randint(0, len(self.pop), 2)
        if self.fitness[candidate[0]] > self.fitness[candidate[1]]:
            parent_2 = self.pop[candidate[0]]
        else:
            parent_2 = self.pop[candidate[1]]

        # crossover
        parent = self.pop[index]
        for gene in range(len(parent)):
            if np.random.rand() < cross_rate:
                parent[gene], parent_2[gene] = parent_2[gene], parent[gene]

        # mutate 
        child = parent
        for gene in range(len(parent)):
            if np.random.rand() < mutation_rate:
                if gene == 0:
                    child[0] = np.random.randint(self.rhs_range[0], self.rhs_range[1] + 1)
                else:
                    child[gene] = 1 - parent[gene]
        
        self.pop[index] = child


# # =====================================
# # Part. ANN Population
# # =====================================
class ann:
    pop = []
    fitness= []
    archive = []

    def __init__(self, pop_size, attr_size, h1_size, h2_size, out_size):
        # desc:     1. 该函数用于初始化一个ANN种群。
        #           2. 每个ANN都是一个多层前馈神经网络，具有固定数量的层和节点。
        #              对于输入层，节点数由给定数据集中的变量数确定。
        #              隐藏节点的数量是手动选择的，可以试验多个不同数量的隐藏节点
        #           3. ANN染色体编码ANN的权重，选择实数作为ANN权重编码。 
        #              一条ANN染色体是一个实数向量ω，表示相应ANN中每个连接的权重。
        #              初始权重设置为均匀分布为[-1,1]之间的随机值。
        #           4. ANN种群是一个 pop_size * conn_num(ann中的连接数) 的int数组,
        #               表示一个包含 pop_size 个染色体，每条染色体有conn_num个基因的FAR种群
        # args:    pop_size,    种群大小
        #          attr_size，  属性数量/输入节点数
        #          h1_size，    第一隐藏层节点数
        #          h2_size,     第二隐藏层节点数
        #          out_size,    输出层节点数
        conn_num = attr_size * h1_size + h1_size * h2_size + h2_size * out_size
        self.pop = np.zeros((pop_size, conn_num), dtype=float) 
        self.fitness = np.zeros(conn_num)
        for i in range(pop_size):
            for j in range(conn_num):
                self.pop[i][j] = random.uniform(-1, 1)
    
    def cal_fitness(self, j, c_aj, d_aj, threshold):
        # desc:    该函数用于计算一个ANN的适应度。
        #          ANN适应度的计算根据文章 (7) - (9) 式
        # args:    c_aj         第j条ANN染色体在所有FAR个体中测试得到的最佳准确度
        #          d_aj         第j条ANN染色体的距离测量d
        #          threshold    阈值
        # returns: 无

        fitness = c_aj / pow(d_aj, 2)

         # Archive valid solutions;
        if c_aj > threshold:
            self.archive.append((self.pop[j]).copy())

        self.fitness[j] = fitness
        # return fitness

    def de_trial(self, cross_rate, mutation_rate):
        # desc:    该函数为ann种群的差分进化过程一部分，获得实验个体ac
        # args:    cross_rate     交叉算子
        #          mutation_rate  变异算子
        # returns: parent_index   主父母ANN个体的下标
        #          ac             实验个体
        nw = len(self.pop)
        
        jr = random.randint(0, nw)
        for k in range(nw):
            # crossover
            sample = np.random.choice(nw, 3, False)
            ap1 = self.pop[sample[0]]
            ap2 = self.pop[sample[1]]
            ap3 = self.pop[sample[2]]
        
            pr = np.random.rand()
            if pr < cross_rate or k == jr:
                gaussian_1 = np.random.normal(0, 1, 1) # 01正态/高斯分布，返回1个值
                ac = ap1 + gaussian_1 * (ap2 - ap3)
            else:
                ac = ap1
                
            # mutation
            pr = np.random.rand()
            if pr < mutation_rate:
                gaussian_2 = np.random.normal(0, mutation_rate, 1)
                ac = ac + gaussian_2

            return (sample[0], ac)                
            # selection
            # note: ap1 = self.pop[sample[0]], fitness(ap1) -> self.fitness[sample[0]]
            # fitness_ac = 0
            # if fitness_ac >= self.fitness[sample[0]]:
            #     a_next = ac
            # else:
            #     a_next = ap1
            # self.pop[sample[0]] = a_next

    def de_selection(self, parent_index, ac, ac_fitness):
        # choose the better one 
        if ac_fitness > self.fitness[parent_index]:
            self.pop[parent_index] = ac
            