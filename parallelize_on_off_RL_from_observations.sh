#!/bin/bash

for seed in $(seq 7 9);
do
qsub -q neuro-autonomy@scc-f03.scc.bu.edu -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/rodent_domain_adaptation_ri_0.01/AWAC_GAE.qsub $seed 
qsub -q neuro-autonomy@scc-f03.scc.bu.edu -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/rodent_domain_adaptation_ri_0.01/AWAC_Q_lambda_Peng.qsub $seed  

qsub -q neuro-autonomy@scc-f03.scc.bu.edu -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/rodent_domain_adaptation_ri_0.005/AWAC_GAE.qsub $seed 
qsub -q neuro-autonomy@scc-f03.scc.bu.edu -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/rodent_domain_adaptation_ri_0.005/AWAC_Q_lambda_Peng.qsub $seed  

qsub -q neuro-autonomy@scc-f03.scc.bu.edu -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/modified_human_expert/AWAC_GAE.qsub $seed 
qsub -q neuro-autonomy@scc-f03.scc.bu.edu -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/modified_human_expert/AWAC_Q_lambda_Peng.qsub $seed 

qsub -q neuro-autonomy@scc-f03.scc.bu.edu -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/modified_human_expert_domain_adaptation/AWAC_GAE.qsub $seed 
qsub -q neuro-autonomy@scc-f03.scc.bu.edu -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/modified_human_expert_domain_adaptation/AWAC_Q_lambda_Peng.qsub $seed 

qsub -q neuro-autonomy@scc-f03.scc.bu.edu -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/rodent_ri_0.01/AWAC_GAE.qsub $seed 
qsub -q neuro-autonomy@scc-f03.scc.bu.edu -l gpus=1 -l gpu_c=3.5 -l gpu_memory=16 QSUBS/on_off_RL_observations/rodent_ri_0.01/AWAC_Q_lambda_Peng.qsub $seed 

done 

#neuro-autonomy@scc-301.scc.bu.edu
#neuro-autonomy@scc-302.scc.bu.edu
#neuro-autonomy@scc-f03.scc.bu.edu
#neuro-autonomy@scc-210.scc.bu.edu