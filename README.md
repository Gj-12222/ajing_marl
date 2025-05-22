# MAGRL
MAGRL(**M**ulti-**A**gent **G**ame with **R**einforcement **L**earning)是一个以MAG为背景的MARL(**M**ulti-**A**gent **R**einforcement **L**earning算法工具包。
这个项目的初心是记录读研期间学习的MARL相关算法代码实践，以及给有需要的人参考借鉴。时隔三年，确立了 MAGRL。目标是在添加记录工作期间学习的RL相关算法和应用，同时规范代码编程能力。MAGRL 还是个粗糙的开源项目，欢迎多提issue。

*  MAGRL 包括两大部分：
  * Env： RL所需的交互环境，有各种Env，也支持自定义Env
  * Algo：(MA)RL的相关算法，分有CTDE和DTDE，也支持自定义Algorithms


## 特性
* (2022/07)添加**UAV Swarm Env**、tensorflow1.x版本
* (2025/01)重构框架结构、算法支持**pytorch**、新增多种算法、添加Multi-ATSC Env
* (2025/02)弃用tf_train.py，后续版本删除tf_train.py

## 安装
### 前提条件
```plantuml
python  v3.10.0+
Windows or Linux x86\_84
```

### 安装步骤

You can install the latest version of the form a cloned Git repository:
```commandline
git clone https://github.com/Gj-12222/ajing_marl.git
cd ajing_marl
pip install -r requirements.txt
```


### 简单运行
```commandline
cd magrl/train
python torch_train.py
```

## 算法性能报告
* 暂缺


## Todo List
* Env：
* Algo：
* Result：


## 参考项目
* MADDPG(https://github.com/openai/maddpg)
* MPE(https://github.com/openai/multiagent-particle-envs)
* DRLib(https://github.com/kaixindelele/DRLib)
* Spinning-up(https://github.com/openai/spinningup)


## 欢迎引用
如果您觉得我们的资源对您有帮助，欢迎引用我们的相关论文：
```
@misc{MAGRL,
  author       = {Gj-12222},
  title        = {{MAGRL}: Multi-Agent Game with Reinforcement Learning},
  year         = {2025},
  note         = {\texttt{https://github.com/Gj-12222/ajing_marl}},
}

```