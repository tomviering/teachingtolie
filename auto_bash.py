#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:11:28 2019

@author: ziqi
"""

workdir = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/tjviering/teachingtolie'
envir = 'source ~/explain/bin/activate'
alljobs = []
jobdir = 'jobs/'


def getjobscript(jobname, command):
    return """#!/bin/sh
#SBATCH --partition=general --qos=long
#SBATCH --time=06:00:00
#SBATCH --mincpus=2
#SBATCH --mem=16048 
#SBATCH --workdir="""+workdir+"""
#SBATCH --job-name=""" + jobname + """
#SBATCH --mail-type=END
#SBATCH --constraint=avx2

"""+envir+"""

echo "Starting at $(date)"
srun """+command+""" 
echo "Finished at $(date)"
"""

for i in range(0, 50):
    jobfile = 'job_exp1_%d.sh' % i
    alljobs.append(jobfile)
    with open(jobdir + jobfile, 'w') as f:
        command = 'python experiment1.py --part %d' % i
        jobstr = getjobscript('e1_%d' % i, command)
        f.write(jobstr)

for i in range(0, 50):
    jobfile = 'job_exp2_1_%d.sh' % i
    alljobs.append(jobfile)
    with open(jobdir + jobfile, 'w') as f:
        command = 'python experiment2.py --no-sticker --part %d' % i
        jobstr = getjobscript('e21_%d' % i, command)
        f.write(jobstr)

for i in range(0, 50):
    jobfile = 'job_exp2_2_%d.sh' % i
    alljobs.append(jobfile)
    with open(jobdir + jobfile, 'w') as f:
        command = 'python experiment2.py --sticker --part %d' % i
        jobstr = getjobscript('e21_%d' % i, command)
        f.write(jobstr)

jobfile_all = jobdir + 'submit_all.sh'
with open(jobfile_all, 'w') as f:
    for j in range(0, len(alljobs)):
        jobfile = alljobs[j]
        f.write('sbatch %s\n' % jobfile)
