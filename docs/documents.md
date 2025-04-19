# MAGRL中文文档

主要是为了梳理项目架构，以及如何更方便的使用

### todo
1. 整合train
2. 梳理algorithm的算法
3. 梳理env
4. 整合utils
5. replaybuffer.py 合并
6. 不同环境应当对应不同安装依赖 magrl[UAVs, ATSC, LLM]
7. 撰写test/xxx.py，给出主要特性的example

## magrl 代码结构

```plain
│ algorithms                   # 算法库
│ envs                         # 环境框架
│ learning_curve               # 绘制训练/评估性能的相关方法
│ metrics                      # 训练保存的指标以及模型的checkpoints
│ tools                        # 一些基本算子和算法相关的utils
│ train                        # 执行训练主入口
```

## 运行训练介绍


## Env使用介绍

### UAV Swarm


### Multi-ATSC


### LLM-o1


## Algo使用介绍



