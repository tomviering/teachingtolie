#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:22:34 2019

@author: ziqi
"""
import os

path_exp1 = 'exp1.1/results/'
path_exp2_r = 'exp2/results/random/'
path_exp2_s = 'exp2/results/sticker/'

exp1_data = {'acc_ori':[],
             'acc_con':[],
             'acc_smi':[],
             'exp_con':[],
             'exp_smi':[],
             'pos_con':[],
             'pos_smi':[]}

exp2_r_data = {'acc':[],
               'exp':[],
               'pos':[]}

exp2_s_data = {'acc_ori_ns':[],
               'acc_ori_s':[],
               'acc_tom_ns':[],
               'acc_tom_s':[],
               'exp_s':[],
               'exp_ns':[],
               'pos_s':[],
               'pos_ns':[]}


def sort(path, exp_data):
    for file in os.listdir(path):
        with open(path + file) as f:
            for line, key in zip(f.readlines(), exp_data.keys()):
                number = line.split()[-1]
                exp_data[key].append(float(number))
    return exp_data
            
def avg_dict(d):
    averages = {}
    for k in d.keys():
        cl = [x for x in d[k]]
        averages[k] = sum(cl)/len(cl)
    return averages
    

if __name__ == '__main__':
    exp1 = sort(path_exp1, exp1_data)
    exp2_r = sort(path_exp2_r, exp2_r_data)
    exp2_s = sort(path_exp2_s, exp2_s_data)
    
    exp1_avg = avg_dict(exp1)
    print('exp1')
    print(exp1_avg)
    exp2_r_avg = avg_dict(exp2_r)
    print('exp2.1')
    print(exp2_r_avg)
    exp2_s_avg = avg_dict(exp2_s)
    print('exp2.2')
    print(exp2_s_avg)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
