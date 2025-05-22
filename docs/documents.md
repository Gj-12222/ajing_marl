# MAGRL中文文档

主要是为了梳理项目架构，以及介绍使用方法

### todo
1. 整合train(torch, ~~tf~~)√
2. 整理config
3. 重构绘制learning curve的代码
3. 梳理algorithm的算法
4. 梳理env
5. 整合utils
6. replaybuffer.py 合并
7. 不同环境应当对应不同安装依赖 magrl[UAV, ATSC, LLM]
8. 撰写test/xxx.py，给出主要特性的example

## magrl 框架
    
```plain
主要分三大部分： 环境Enversion， 算法包Algos， 训练方式Training

环境Enversion：
    可以单独安装某一环境的依赖包，并使用该环境
    也可以全部按转所有环境的依赖包，使用所有环境
    特别注意：UAV, ATSC与LLM环境相差较大，设计环境考虑如何兼容
    
算法包Algos：
    确保每个算法都有独立的代码文件，不同算法需要不同的调参方式，减少代码复用
    提供DTDE, CTCE, CTDE类型算法（可独立形成算法，也可超参控制）
    
训练方式Training：
    提供off-policy training，on-policy training（on-policy并不一定必须在每个episode结束才能更新）
    
```

## magrl 代码结构

```plain
│ algorithms                   # 算法库
│ config                       # 整体配置文件，包含算法参数，环境参数，训练参数等
│ envs                         # 环境框架
│ learning_curve               # 绘制训练/评估性能的相关方法
│ metrics                      # 训练保存的指标以及模型的checkpoints
│ tools                        # 一些基本算子和算法相关的utils
│ train                        # 执行训练主入口
```

## 运行训练介绍


## Env使用介绍
```plain
1. 支持多种环境，环境之间相互独立，避免依赖包和导入冲突
2. 提供环境基本格式，用户可以自定义环境，同时便于新增默认环境
```
## 环境的搭建
    1. 需要有有个环境基类（主要强调了一些必须实现的函数对象: init, reset, step, render, close）
    2. def step(self, ...)该函数主要计算了环境动力学，即环境的状态转移。
    3. 环境中的智能体Agent，需要有智能体基类（主要强调了init, action, update, save, load）
    4. def init与 def update中需要神经网络的参与（Actor/Critic Network）
    5. 神经网络需要继承自nn.Module，并实现forward函数
## 仿真引擎
    1. 需要有一个仿真引擎(如果使用第三方)
    2. 根据仿真引擎，定制环境类：init, reset, step, render, close
    3. 根据仿真引擎，定制Agent类
    4. 根据仿真引擎，定制Reward函数（在def step中使用）

## 支持的环境： 
### UAV Swarm
    这里是MARL的环境，UAV Swarm指的是无人机集群，主要以集群内协作集群间对抗的仿真场景。


### Multi-agent ATSC


### LLM-reasoning


## Algo使用介绍
    1. 支持多种算法，算法之间相互独立



