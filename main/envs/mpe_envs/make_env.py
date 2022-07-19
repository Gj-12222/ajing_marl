"""
Code for creating a multiagent environment with one of the scenarios listed
使用列出的场景之一创建多代理环境的代码
in ./scenarios/.
Can be called by using, for example:可以通过使用调用，例如:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.
在生成env对象之后，可以像OpenAI gym一样使用环境。
A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.

使用此环境的策略必须以所有代理的列表形式输出操作。列表中的每个元素都应该是一个大小相同的numpy数组
(env.world。dim_p + env.world。dim_c, 1)。
在这个数组中，物理动作先于通信动作。参见environment.py了解更多细节。
"""

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data是否需要生成基准测试数据
                            (usually only done during evaluation)通常只在评估期间进行

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from .multiagent.environment import MultiAgentEnv
    import envs.mpe_envs.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env
