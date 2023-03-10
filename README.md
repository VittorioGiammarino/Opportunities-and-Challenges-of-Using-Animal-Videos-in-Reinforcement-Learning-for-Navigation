# Opportunities-and-Challenges-of-Using-Animal-Videos-in-Reinforcement-Learning-for-Navigation

## Initial instructions

### Use anaconda to create a virtual environment

**Step 1.** install miniconda

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Step 2.** clone repo and create conda environment

```shell
git clone https://github.com/VittorioGiammarino/Opportunities-and-Challenges-of-Using-Animal-Videos-in-Reinforcement-Learning-for-Navigation.git
```

```shell
cd Opportunities-and-Challenges-of-Using-Animal-Videos-in-Reinforcement-Learning-for-Navigation
conda env create -f environment.yml
conda activate miniconda
```

**Step 3.** Download data

Rodent data: https://drive.google.com/file/d/1UOcu3ViEwylLxntVE8slYx3XQyfGFCBc/view?usp=share_link

Minigrid data: https://drive.google.com/file/d/1SVrPysESp6VTEd4USVWxIXbZa4IEosZ2/view?usp=share_link

or 

```shell
pip install gdown
gdown 1UOcu3ViEwylLxntVE8slYx3XQyfGFCBc
tar -xf data_set.tar.xz
rm data_set.tar.xz
gdown 1SVrPysESp6VTEd4USVWxIXbZa4IEosZ2
tar -xf offline_data_set.tar.xz
rm offline_data_set.tar.xz 
```

**Step 4.** Run experiments

#### Minigrid data experiments

```shell
python main.py --mode on_off_RL_from_observations --policy AWAC_GAE --max_iter 200 --data_set modified_human_expert --intrinsic_reward 0.01 --seed 0 --Train_encoder
```

```shell
python main.py --mode on_off_RL_from_observations --policy AWAC_Q_lambda_Peng --max_iter 200 --data_set modified_human_expert --intrinsic_reward 0.01 --seed 0 --Train_encoder
```

```shell
python main.py --mode on_off_RL_from_observations --policy AWAC_GAE --max_iter 200 --data_set modified_human_expert --intrinsic_reward 0.01 --domain_adaptation --seed 0 --Train_encoder
```

```shell
python main.py --mode on_off_RL_from_observations --policy AWAC_Q_lambda_Peng --max_iter 200 --data_set modified_human_expert --intrinsic_reward 0.01 --domain_adaptation --seed 0 --Train_encoder
```

#### Rodent data experiments

```shell
python main.py --mode on_off_RL_from_observations --policy AWAC_GAE --max_iter 200 --data_set rodent --intrinsic_reward 0.01 --domain_adaptation --seed 0 --Train_encoder
```

```shell
python main.py --mode on_off_RL_from_observations --policy AWAC_Q_lambda_Peng --max_iter 200 --data_set rodent --intrinsic_reward 0.01 --domain_adaptation --seed 0 --Train_encoder
```

```shell
python main.py --mode on_off_RL_from_observations --policy AWAC_GAE --max_iter 200 --data_set rodent --intrinsic_reward 0.005 --domain_adaptation --seed 0 --Train_encoder
```

```shell
python main.py --mode on_off_RL_from_observations --policy AWAC_Q_lambda_Peng --max_iter 200 --data_set rodent --intrinsic_reward 0.005 --domain_adaptation --seed 0 --Train_encoder
```


```shell
python main.py --mode on_off_RL_from_observations --policy AWAC_GAE --max_iter 200 --data_set rodent --intrinsic_reward 0.01 --seed 0 --Train_encoder
```

```shell
python main.py --mode on_off_RL_from_observations --policy AWAC_Q_lambda_Peng --max_iter 200 --data_set rodent --intrinsic_reward 0.01 --seed 0 --Train_encoder
```

#### RL

```shell
python main.py --mode RL --policy AWAC_GAE --max_iter 200 --Entropy --seed 0
```

```shell
python main.py --mode RL --policy AWAC_Q_lambda_Peng --Entropy --max_iter 200 --seed 0
```
