# ajing_marl
 This is a personal library that strives to implement various MARL algorithms. 
 
 The code style of this library from the openAI maddpg(https://github.com/openai/maddpg) and default environment refers to MPE(https://github.com/openai/multiagent-particle-envs), as well other libraries.
 
 The environment only integrates MPE, and the algorithm currently only has the form of CTDE and independent learning(IL) + DRL.
 

# Install
### Requirenments
python  v3.7.0+

Windows 10 or Linux x86\_84

### Install dependencies
You can install the latest version of the from a cloned Git repository:

git clone https://github.com/Gj-12222/ajing_marl

pip install -r requirements.txt

# run 
cd main/training

python train.py

### 注释
the environment config in envs/env_config.py