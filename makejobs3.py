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
#SBATCH --mincpus=8
#SBATCH --mem=10000 
#SBATCH --workdir="""+workdir+"""
#SBATCH --job-name=""" + jobname + """
#SBATCH --output=logs/"""+jobname+""".txt
#SBATCH --error=logs/"""+jobname+""".txt
#SBATCH --mail-type=END
#SBATCH --gres=gpu:pascal:1

module use /opt/insy/modulefiles
module load cuda/10.1 cudnn/10.1-7.6.0.64

"""+envir+"""

echo "Starting at $(date)"
srun """+command+""" 
echo "Finished at $(date)"
"""

lr_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5] # 1e-2, 1e-3 etc....
op_list = ['adam', 'sgd']
pretrained_list = ['True', 'False']


def get_command(myjobname, lr, op, lambda_c, lambda_g, pretrained):
    # careful each line should begin with a space!!
    command = 'python main.py' \
              ' --cuda=True' \
              ' --num_workers=4' \
              ' --RAM_dataset=True' \
              ' --train_batch_size=32' \
              ' --val_batch_size=10' \
              ' --lambda_c={lambda_c:.1e}' \
              ' --vis_name={vis_name:s}' \
              ' --lr={lr:.1e}' \
              ' --optimizer={op:s}' \
              ' --lambda_g={lambda_g:.1e}' \
              ' --pretrained={pretrained:s}'\
              ' --attack_loss=random'.format(vis_name=myjobname, lr=lr, op=op, lambda_c=lambda_c, lambda_g=lambda_g,
                                                   pretrained=pretrained)
    return command

# do trade-off experiment
lambda_g_list = [1e-2, 1e-1, 1e0, 1e1, 1e2] # 1e0, 1e-1, etc.
for lr in lr_list:
    for op in op_list:
        for my_lambda_g in lambda_g_list:
            for pretrained in pretrained_list:
                myjobname = 'tradeoffstd_%s_lr_%.1e_pretrn_%s_lambda_g_%.1e' % (op, lr, pretrained, my_lambda_g)
                jobfile = '%s.sh' % myjobname
                alljobs.append(jobfile)
                with open(jobdir + jobfile, 'w') as f:
                    command = get_command(myjobname=myjobname, lr=lr, op=op, pretrained=pretrained, lambda_c=0,
                                          lambda_g=my_lambda_g)
                    jobstr = getjobscript(myjobname, command)
                    f.write(jobstr)

numjobs = 0
jobfile_all = jobdir + 'submit_all.sh'
with open(jobfile_all, 'w') as f:
    for j in range(0, len(alljobs)):
        jobfile = alljobs[j]
        f.write('sbatch %s\n' % jobfile)
        numjobs = numjobs + 1

print(numjobs)