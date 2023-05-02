# Landmark-based mapping
Thank you for being interested in our work! This repository is a PyTorch implementation for paper ***[Learning Continuous Control Policies for Information-Theoretic Active Perception](https://arxiv.org/pdf/2209.12427.pdf)***
in ICRA 2023. Authors: [Pengzhi Yang](https://pengzhi1998.github.io/), 
[Yuhan Liu](https://jaysparrow.github.io/), [Shumon Koga](https://shumon0423.github.io/), [Arash Asgharivaskasi](https://arashasgharivaskasi-bc.github.io/), [Nikolay Atanasov](https://natanaso.github.io/).
If you are using the code for research work, please cite:
```
@inproceedings{yang2023icra,
  title={Learning Continuous Control Policies for Information-Theoretic Active Perception},
  author={Yang, Pengzhi and Liu, Yuhan and Koga, Shumon and Asgharivaskasi, Arash and Atanasov, Nikolay},
  booktitle={IEEE international conference on robotics and automation (ICRA)},
  year={2023}
}
```
## Brief Intro
This paper proposes a method for learning continuous control policies for active landmark localization and
exploration using an information-theoretic cost. We consider a mobile robot detecting landmarks within a limited sensing
range, and tackle the problem of learning a control policy that maximizes the mutual information between the landmark
states and the sensor observations. Here is a pipeline for our work:

<div style="text-align:center"><img src="https://github.com/JaySparrow/RL-for-active-mapping/blob/master/teaser.png" width = "600" height = "350"/></div>

## Instruction
1. First clone this repo and install the dependencies running the following commands in the terminal:
```
git clone https://github.com/JaySparrow/RL-for-active-mapping.git
git clone https://github.com/pengzhi1998/Landmark-based-mapping-Unity.git
conda create -n landmark_mapping python==3.7 -y
conda activate landmark_mapping
cd ./RL-for-active-mapping
pip install -r requirements.txt
cd ../
git clone --branch release_18 https://github.com/Unity-Technologies/ml-agents.git
cd ./ml-agents
pip install -e ./ml-agents-envs
pip install gym-unity==0.27.0
```
2. cd to the ```./RL-for-active-mapping/toy_active_mapping_ppo``` directory, then train and test the model:
```
python ./agent.py 
python ./agent_test.py  
```
There are multiple arguments for you to tune with for different scenarios. You could also test the model in Unity environment:
```
python ./agent_unity_test.py 
```
3. To play with Unity env, build it with the repo you have cloned:  
(1) Create a directory `./toy_active_mappking_ppo/Unity_envs` in your ```RL-for-active-mapping``` repo.  
(2) Build the environment with the cloned ```Landmark-based-mapping-Unity``` repo to ```Unity_envs``` just created.



