#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:22:34 2019

@author: ziqi
"""
import os
import numpy as np

path_exp1 = 'exp1.1/results/'
path_exp2_r = 'exp2/results/random/'
path_exp2_s = 'exp2/results/sticker/'

exp1_data = {'acc_ori': [],
             'acc_con': [],
             'acc_smi': [],
             'exp_con': [],
             'exp_smi': [],
             'pos_con': [],
             'pos_smi': []}

exp2_r_data = {'acc': [],
               'exp': [],
               'pos': []}

exp2_s_data = {'acc_ori_ns': [],
               'acc_ori_s': [],
               'acc_tom_ns': [],
               'acc_tom_s': [],
               'exp_s': [],
               'exp_ns': [],
               'pos_s': [],
               'pos_ns': []}


def sort(path):
    diff = {}
    largest_diff = 0
    diff2 = np.empty(50000)
    for file in os.listdir(path):
        print(file)
        if "nosticker" in file:
            print('skipping file %s' % file)
            continue
        print('doing file %s' % file)
        with open(path + file) as f:
            for line in f:
                parts = line.split()
                for (i, p) in enumerate(parts):
                    print('%d: %s ' %( i, p))
                image_id = int(parts[0])
                print(image_id)
                difference = float(parts[3])
                diff[image_id] = difference
                diff2[image_id] = difference
                #return diff
    return (diff, diff2)


def avg_dict(d):
    averages = {}
    for k in d.keys():
        cl = [x for x in d[k]]
        averages[k] = sum(cl) / len(cl)
    return averages


if __name__ == '__main__':

    (diff, diff2) = sort('../diffpoststicker/')
    print(len(diff))
    print(np.mean(diff2))



























