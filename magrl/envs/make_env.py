"""
Code for creating a multiagent environment with one of the scenarios listed

in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.

(env.world。dim_p + env.world。dim_c, 1)。
在这个数组中，物理动作先于通信动作。参见environment.py了解更多细节。
"""

import imp
from magrl.config.model_config import envs_paths


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
    from magrl.envs.environment import MultiAgentEnv
    # load scenario from script
    scenario = scenario_load(scenario_name).Scenario()
    # create world
    world = scenario.make_world()
    render_callback, set_render_callback, setAction_callback = None, None, None
    if hasattr(scenario, 'render'):
        render_callback = scenario.render
    if hasattr(scenario, 'set_render'):
        set_render_callback = scenario.set_render
    if hasattr(scenario, 'setAction'):
        setAction_callback = scenario.setAction

    done_callback, terminal_callback = None, None
    if hasattr(scenario, "done"):
        done_callback = scenario.done
    if hasattr(scenario, "terminal"):
        terminal_callback = scenario.terminal

    # create multiagent environment
    env = MultiAgentEnv(world=world,
                        reset_callback=scenario.reset_world,
                        reward_callback=scenario.reward,
                        observation_callback=scenario.observation,
                        info_callback=None,
                        done_callback=done_callback,
                        terminal_callback=terminal_callback,
                        render_callback=render_callback,
                        set_render_callback=set_render_callback,
                        setAction_callback=setAction_callback)
    return env

def scenario_load(name):
    pathname = envs_paths.get(name, None)
    if pathname is None:
        print('no have env name', name, 'check envs_paths!')

    return imp.load_source('', pathname)
