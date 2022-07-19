# ajing_marl
 This is a personal library that strives to implement various MARL algorithms. 
 
 The code style of this library from the openAI maddpg(https://github.com/openai/maddpg) and default environment refers to MPE(https://github.com/openai/multiagent-particle-envs), as well other libraries.
 
 The environment only integrates MPE, and the algorithm currently only has the form of CTDE and independent learning(IL) + DRL.
 
# Install
pip install -r requirements.txt

# run 
cd main/training

python train.py

# 注释
main文件是可运行程序

test文件是待测试程序