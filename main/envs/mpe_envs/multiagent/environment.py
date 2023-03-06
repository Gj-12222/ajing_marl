import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from tools.multi_discrete import MultiDiscrete
import math
import copy

attack_angle = 60  # attack angle
defense_angle = 90 # defense angle
fire_range = 0.3  # fire range

jam_range = 0.6  # jam range

"""
   MultiAgentEnv input parms
   world=Scenario.make_world()
   reset_callback=scenario.reset_world, 
   reward_callback=scenario.reward,
   observation_callback= scenario.observation,
   info_callback=scenario.done
   done_callback=None
   shared_viewer=True
   """
# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        # environment parameters
        #self.discrete_action_space = True
        self.discrete_action_space = False
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # ① physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-1, high=+1, shape=(world.dim_c+world.dim_c+world.dim_c,),
                                            dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)

            # ② roll
            if self.discrete_action_space:
                r_action_space = spaces.Discrete(world.dim_f)
                total_action_space.append(r_action_space)
            # ③ jam
            if self.discrete_action_space:
                f_action_space = spaces.Discrete(world.dim_p)
                total_action_space.append(f_action_space)

            # ④communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
               c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if agent.silent:
                total_action_space.append(c_action_space)

            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                # print(act_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.obs_dim = obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents

        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])

        # advance world state
        self.world.step()
        # record observation for each agent

        first_death = []
        for i, agent in enumerate(self.agents):
            if agent.death:
                if agent.adversary:
                    self.agents[i].state.p_pos = np.array([20.0 for j in range(self.world.dim_p)])
                else:
                    self.agents[i].state.p_pos = np.array([-20.0 for j in range(self.world.dim_p)])

                self.agents[i].state.p_vel = np.zeros(self.world.dim_p)
                self.agents[i].state.c = np.zeros(self.world.dim_c)
                self.agents[i].state.f = np.zeros(self.world.dim_p)
                self.agents[i].state.p_roll = np.zeros(self.world.dim_c)
                first_death.append(i)

        for i, agent in enumerate(self.agents):
            if i in first_death:
                state_dim = self.obs_dim
                obs_n.append(np.zeros(state_dim))

                reward_n.append(0)
                done_n.append(True)
                if agent.adversary:
                 self.agents[i].state.p_pos = np.array([-10.0 for j in range(self.world.dim_p)])
                else:
                 self.agents[i].state.p_pos = np.array([-20.0 for j in range(self.world.dim_p)])

                self.agents[i].state.p_vel = np.zeros(self.world.dim_p)
                self.agents[i].state.c = np.zeros(self.world.dim_c)
                self.agents[i].state.f = np.zeros(self.world.dim_p)
                self.agents[i].state.p_roll = np.zeros(self.world.dim_c)
            else:
                obs_n.append(self._get_obs(agent))
                for j in range(3):  # action_dim = 3 for every agent
                    obs_n[i][-3 + j] = copy.deepcopy(action_n[i][j])

                reward_n.append(self._get_reward(agent))
                done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case

        reward = np.sum(reward_n)
        if self.shared_reward: # if agents are coopration in environment, sellf.shared_reward = True, else is False.
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n


    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        agent.action.f = np.zeros(1)
        agent.action.r = np.zeros(1)

        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:  #
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:  #
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    if math.isnan(action[0][0]):
                        action[0][0] = 1
                    # agent.action.u = (action[0][0] + 1.0) /2.0
                    agent.action.u = action[0][0]
                    sensitivity = 5.0
                    if agent.accel is not None:
                        sensitivity = agent.accel
                    agent.action.u *= sensitivity

                    if abs(action[0][1]) > 1:
                        action[0][1] = 1 / action[0][1]
                    agent.action.r = action[0][1] * 2.3 * math.pi / 180

                    if action[0][2] >= 0:
                        agent.action.f = 1
                    else:
                        agent.action.f = 0

        action = action[1:]


        #② roll
        if len(action) >0:
            if agent.movable:
                # physical action
                if self.discrete_action_input:
                    agent.action.r = np.zeros(self.world.dim_p)
                    # process discrete action
                    if action[0] == 1: agent.action.r[0] = -1.0
                    if action[0] == 2: agent.action.r[0] = +1.0
                    if action[0] == 3: agent.action.r[1] = -1.0
                    if action[0] == 4: agent.action.r[1] = +1.0
                else:
                    if self.force_discrete_action:
                        d = np.argmax(action[0])
                        action[0][:] = 0.0
                        action[0][d] = 1.0
                    if self.discrete_action_space:
                        d = np.argmax(action[0])
                        agent.action.r += action[0][d] * math.pi / 180
                    else:
                        agent.action.r += action[0]

                action = action[1:]

        # ③ jam
        if len(action) > 0:
            if self.discrete_action_space:
                d = np.argmax(action[0])
                action[0][:] = 0.0
                action[0][d] = 1.0
                agent.action.f = action[0][0] - action[0][1]
            else:
                agent.action.f = action[0]

            action = action[1:]

        # ④ communication
        if len(action) >0:
            if agent.silent:
                if self.discrete_action_input:
                    agent.action.c = np.zeros(self.world.dim_c)
                    agent.action.c[action[0]] = 1.0
                else:
                    agent.action.c = action[0]
                action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for i, agent in enumerate(self.world.agents):
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')

        for i in range(len(self.viewers)):

            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from envs.mpe_envs.multiagent import rendering
                self.viewers[i] = rendering.Viewer(800, 800)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from envs.mpe_envs.multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            # zgy  *************************
            self.render_lines = []
            for entity in self.world.entities:
                xform = rendering.Transform()  # init transform xform=[平移=0，旋转=0，尺度变换=1]
                if 'agent' in entity.name:
                     # uav
                    geom  = rendering.make_uav(entity.size)
                    # small forward_sector
                    geom_attact_sector = rendering.make_forward_sector(radius=fire_range, angle_start= -attack_angle/2, angle_end=attack_angle/2)  # 不需要等分360度，仅需等分attack_angle度
                    # forward_sector
                    geom_defence_sector = rendering.make_forward_sector(radius=fire_range,angle_start=180-defense_angle/2,angle_end=180+defense_angle/2)  # 不需要等分360度，仅需等分attack_angle度
                    # jam circle
                    geom_explore = rendering.make_circle(jam_range)

                    geom.set_color(*entity.color, alpha=0.5)
                    geom_attact_sector.set_color(*entity.color, alpha=0.3)
                    geom_defence_sector.set_color(*entity.color, alpha=0.05)

                    geom_explore.set_color(*entity.color, alpha= 0.01)

                else:
                    geom_explore = rendering.make_circle(2*entity.size)
                    geom_explore.set_color(*entity.color)

                geom.add_attr(xform)
                geom_attact_sector.add_attr(xform)
                geom_defence_sector.add_attr(xform)
                geom_explore.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms.append(geom_attact_sector)
                self.render_geoms.append(geom_defence_sector)
                self.render_geoms.append(geom_explore)
                self.render_geoms_xform.append(xform)

                # add geoms to viewer
                for viewer in self.viewers:
                    viewer.geoms = []
                    for geom in self.render_geoms:
                        viewer.add_geom(geom)
                    # for geom_attact_sector in self.render_geoms:
                    #     viewer.add_geom(geom_attact_sector)
                    # for geom_defence_sector in self.render_geoms:
                    #     viewer.add_geom(geom_defence_sector)
                    # for geom_explore in self.render_geoms:
                    #     viewer.add_geom(geom_explore)

        results = []

        # zgy
        agents = self.world.agents
        le = len(self.viewers)
        for i in range(len(self.viewers)):
            from envs.mpe_envs.multiagent import rendering
            # update bouds to center around agent
            cam_range = 2
            if self.shared_viewer:  # =true
                pos = np.zeros(self.world.dim_p)
            else:  # =false
                pos = self.agents[i].state.p_pos

            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                my_chi = np.zeros(1)
                if entity.death == True:
                    my_chi[0] = 0
                else:
                    my_chi[0] = agents[e].state.course_angle

                if entity.action.f == 1 and entity.state.f[0] >0:
                    self.render_geoms[(e+1)*4-1].set_color(*entity.color,alpha=0.5)
                else:
                    self.render_geoms[(e + 1) * 4 - 1].set_color(*entity.color, alpha=0.1)

                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                self.render_geoms_xform[e].set_rotation(my_chi[0])
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field

        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space

class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i + env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
