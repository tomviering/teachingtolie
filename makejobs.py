#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:11:28 2019

@author: ziqi
"""

import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

workdir = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/tjviering/teachingtolienewpaper/'
envir = 'source ~/explain/bin/activate'
alljobs = []
jobdir = 'jobs/'

mkdir(jobdir)

def getjobscript(jobname, command):
    return """#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=04:00:00
#SBATCH --mincpus=2
#SBATCH --mem=10000 
#SBATCH --workdir="""+workdir+"""
#SBATCH --job-name=""" + jobname + """
#SBATCH --output="""+jobname+""".txt
#SBATCH --error="""+jobname+""".txt
#SBATCH --mail-type=END
#SBATCH --gres=gpu:pascal:1

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

"""+envir+"""

echo "Starting at $(date)"
srun """+command+""" 
echo "Finished at $(date)"
"""

lr_list = [2, 3, 4] # 1e-2, 1e-3 etc....
op_list = ['adam', 'sgd']
pretrained_list = ['True', 'False']

# only do classification experiment
for lr in lr_list:
    for op in op_list:
        for pretrained in pretrained_list:
            myjobname = 'new_closs_%s_%d_%s' % (op, lr, pretrained)
            jobfile = '%s.sh' % myjobname
            alljobs.append(jobfile)
            with open(jobdir + jobfile, 'w') as f:
                command = 'python main.py --cuda=True --train_batch_size=32 --alpha_c=1.0 --alpha_g=0.0 --vis_name=%s --lr=1e-%d --optimizer=%s --pretrained=%s' % (myjobname, lr, op, pretrained)
                jobstr = getjobscript(myjobname, command)
                f.write(jobstr)

# only do gradcam experiment
for lr in lr_list:
    for op in op_list:
        for pretrained in pretrained_list:
            myjobname = 'new_gloss_%s_%d_%s' % (op, lr, pretrained)
            jobfile = '%s.sh' % myjobname
            alljobs.append(jobfile)
            with open(jobdir + jobfile, 'w') as f:
                command = 'python main.py --cuda=True --train_batch_size=32 --alpha_c=0.0 --alpha_g=1.0 --vis_name=%s --lr=1e-%d --optimizer=%s --pretrained=%s' % (myjobname, lr, op, pretrained)
                jobstr = getjobscript(myjobname, command)
                f.write(jobstr)

# do trade-off experiment
lambda_list = [0, 1, 2] # 1e0, 1e-1, etc.
for lr in lr_list:
    for op in op_list:
        for my_lambda in lambda_list:
            for pretrained in pretrained_list:
                myjobname = 'new_tradeoff_%s_%d_%d_%s' % (op, lr, my_lambda, pretrained)
                jobfile = '%s.sh' % myjobname
                alljobs.append(jobfile)
                with open(jobdir + jobfile, 'w') as f:
                    command = 'python main.py --cuda=True --train_batch_size=32 --alpha_c=1.0 --vis_name=%s --lr=1e-%d --optimizer=%s --alpha_g=1e-%d --pretrained=%s' % (myjobname, lr, op, my_lambda, pretrained)
                    jobstr = getjobscript(myjobname, command)
                    f.write(jobstr)

jobfile_all = jobdir + 'submit_all.sh'
with open(jobfile_all, 'w') as f:
    for j in range(0, len(alljobs)):
        jobfile = alljobs[j]
        f.write('sbatch %s\n' % jobfile)