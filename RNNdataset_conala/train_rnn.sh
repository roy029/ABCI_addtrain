#!/bin/bash
#$ -l rt_G.small=1
#$ -l h_rt=05:00:00
#$-j y
#$-m b
#$-m a
#$-m e
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8 cuda/11.2 cudnn/8.1

pip3 install -r requirements.txt

python3 rnn_possi_7_5.py