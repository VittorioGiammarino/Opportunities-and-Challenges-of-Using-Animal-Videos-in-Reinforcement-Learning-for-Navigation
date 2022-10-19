#!/bin/bash

for seed in $(seq 0 9);
do
#qsub -q neuro-autonomy@scc-f03.scc.bu.edu -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/RL/PPO.qsub $seed 
qsub -q neuro-autonomy@scc-f03.scc.bu.edu -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/RL/AWAC_GAE.qsub $seed 
qsub -q neuro-autonomy@scc-f03.scc.bu.edu -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/RL/AWAC_Q_lambda_Peng.qsub $seed 

done 
