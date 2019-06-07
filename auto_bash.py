#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:11:28 2019

@author: ziqi
"""

for i in range(0, 50):
    with open('job_exp2_%d.sh' % i, 'w') as f:
        f.write("""#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=04:00:00
#SBATCH --mincpus=2
#SBATCH --mem=16048 
#SBATCH --workdir=/tudelft.net/staff-bulk/ewi/insy/VisionLab/tjviering/conv-explain 
#SBATCH --job-name=j_exp2_""" + str(i) + '\n'
"""#SBATCH --mail-type=END

source ~/explain/bin/activate

echo "Starting at $(date)"
srun python gradcam4_exp2.py --part """+ str(i) + '\n' + 
"""echo "Finished at $(date)"
"""
)
    