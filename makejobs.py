#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:11:28 2019

@author: ziqi
"""
from utils import mkdir

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

"""+envir+"""

echo "Starting at $(date)"
srun """+command+""" 
echo "Finished at $(date)"
"""

lr_list = [2, 3, 4] # 1e-2, 1e-3 etc....
op_list = ['adam', 'sgd']

for lr in lr_list:
    for op in op_list:
        myjobname = 'trn_%s_%d' % (op, lr)
        jobfile = '%s.sh' % myjobname
        alljobs.append(jobfile)
        with open(jobdir + jobfile, 'w') as f:
            command = 'python main.py --cuda True --train_batch_size=32 --criterion=1 --vis_name=%s --lr 1e-%d --optimizer %s' % (myjobname, lr, op)
            jobstr = getjobscript(myjobname, command)
            f.write(jobstr)

lambda_list = [0, 1, 2] # 1e0, 1e-1, etc.
for lr in lr_list:
    for op in op_list:
        for my_lambda in lambda_list:
            myjobname = 'gc_%s_%d_%d' % (op, lr, my_lambda)
            jobfile = '%s.sh' % myjobname
            alljobs.append(jobfile)
            with open(jobdir + jobfile, 'w') as f:
                command = 'python main.py --cuda True --train_batch_size=32 --criterion=3 --vis_name=%s --lr 1e-%d --optimizer %s --lambda 1e-%d' % (myjobname, lr, op, my_lambda)
                jobstr = getjobscript(myjobname, command)
                f.write(jobstr)

jobfile_all = jobdir + 'submit_all.sh'
with open(jobfile_all, 'w') as f:
    for j in range(0, len(alljobs)):
        jobfile = alljobs[j]
        f.write('sbatch %s\n' % jobfile)