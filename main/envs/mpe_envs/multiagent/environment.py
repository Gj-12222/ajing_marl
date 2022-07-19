import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from tools.multi_discrete import MultiDiscrete
import math
import copy

attack_angle = 60  # 攻击区角度
defense_angle = 90 # 防守区角度
fire_range = 0.3  # 攻击距离
# comput_range = 0.6  # 计算距离-无效量
#guojing set
jam_range = 0.6  # 干扰距离

"""
该环境是 离散动作空间， 无离散动作输入
"""
"""
   MultiAgentEnv的输入：
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
        # self.discrete_jam_action_space = True # 干扰-改成连续
        # self.discrete_action_space = False  # 驱动力+滚转角+通信
        # self.discrete_jam_action_space = False  # 干扰-改成连续
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        ## liyuan修改部分******************
        self.num_adversaries = 0  # 敌方数量
        for agent in self.agents:
            if agent.adversary:  #如果agent里有adversary属性，则为敌方单位
                self.num_adversaries += 1  # +1
        ##  *****************************

        # configure spaces 配置空间
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # ① physical action space 动力动作空间
            if self.discrete_action_space:  # 如果是离散的动作
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:  # 是连续动作
                # 则连续动作的区间：[-1，1]
                u_action_space = spaces.Box(low=-1, high=+1, shape=(world.dim_c+world.dim_c+world.dim_c,),
                                            dtype=np.float32)
            if agent.movable:  # 可移动
                total_action_space.append(u_action_space)
            """#########guojing  set########### """
             # print('*****guojing set action is good *******')
            # ② 滚转角速度-默认连续
            if self.discrete_action_space:  # 如果是离散的动作
                r_action_space = spaces.Discrete(world.dim_f)
            #else:  # 是连续动作
            #     # 则连续动作的区间：[-1，1]
            #    r_action_space = spaces.Box(low=-agent.max_roll, high=+agent.max_roll, shape=(world.dim_c,),
            #                                dtype=np.float32)
            #if agent.movable:  # 可移动
            #    total_action_space.append(r_action_space)
            # ③ 干扰设计
            if self.discrete_action_space:  # 离散
                f_action_space = spaces.Discrete(world.dim_p)  # 干扰，不干扰
            #else:  # 连续
            #    f_action_space = spaces.Box(low=-1.0, high=1.0, shape=(world.dim_f,), dtype=np.float32)
            # total_action_space.append(f_action_space)  # 默认有干扰

            # ④communication action space  # 通信动作空间
            if self.discrete_action_space:  # 离散
                c_action_space = spaces.Discrete(world.dim_c)
            #else:  # 连续
            #    c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
           ##郭靖改-不参与通信********************##
            if not agent.silent:
            # if  agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:  # 有其他动作
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    # n是维度
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])

                else:
                    act_space = spaces.Tuple(total_action_space)
                # print(act_space)
                self.action_space.append(act_space)
            else:  # 无其他动作，只有运动
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
        # 输入动作
        # print('输入动作：',action_n)
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents

        # print("action_n: ",action_n)
        # for ag in self.agents:
        # print ("before pos: ",ag.state.p_pos,ag.state.p_vel)
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])

        # advance world state
        self.world.step()
        # for ag in self.agents:
        # print ("after pos: ",ag.state.p_pos,ag.state.p_vel)

        # record observation for each agent
        '''
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))
        '''
        ###changed by liyuan
        # 获取双方对战数量
        red_agent_num = 0
        green_agent_num = 0
        for agent in self.agents:
            if agent.adversary:
                red_agent_num += 1
            else:
                green_agent_num += 1

        first_death = []  # 第一次死亡？
        for i, agent in enumerate(self.agents):
            if agent.death:  # 如果agent死了
                # state_dim = 4+4*(self.n-1)+self.n+5
                # obs_n[i]=np.zeros(state_dim)
                # reward_n[i]=0
                # done_n[i]=False
                if agent.adversary:  # 红方
                    # 定义死亡的agent的位置是 (-10,-10)
                    self.agents[i].state.p_pos = np.array([20.0 for j in range(self.world.dim_p)])
                else:  # 绿方
                    # 定义死亡的agent的位置是(-20, -20)
                    self.agents[i].state.p_pos = np.array([-20.0 for j in range(self.world.dim_p)])
                # 死亡agent的速度是0
                self.agents[i].state.p_vel = np.zeros(self.world.dim_p)
                # 死亡agent的通信是0
                self.agents[i].state.c = np.zeros(self.world.dim_c)
                # guojing set
                self.agents[i].state.f = np.zeros(self.world.dim_p)  # 死亡干扰= 0
                # 滚转角= 0
                self.agents[i].state.p_roll = np.zeros(self.world.dim_c)  # 死亡 滚转角=0
                # 记录死亡agent序号
                first_death.append(i)
        # 打包[[1,agent1],[2,agnr2],[],...,[n,agnetn]]
        for i, agent in enumerate(self.agents):
            if i in first_death:  # i是死亡agent的序号
                """
                action_number=[np.zeros(5)]  # 2个位置+ 1个滚转角+1个作战 = 4
                pv = len(agent.state.p_vel)  # 2      2  自身位置
                pp = len(agent.state.p_pos)  # 2      2  自身速度
                ep = len(entity_pos)         # 2m     0  障碍物相对位置
                op = len(other_pos)          # 2(n-1) 22 其他agent相对距离
                ov = len(other_vel)          # 2(n-1) 22 其他agent的速度
                oc = len(our_chi)            # 1      1  自身角度
                ohc = len(other_chi)         # n-1    11 其他agent角度
                an = len(action_number)      # 3      3  动作数量=3        
                """
                # 状态维度是 2+2+1+1+0 +7*（n-1）+1+3 = 6+7(n-1)+1+3 = 6+7*(10-1)+1+3=6+63+1+3=73
                # state_dim = 4 + 4 * (self.n - 1) + self.n + 3
                # state_dim =  2 + 2 +  1 + 1 + 2 * (self.n - 1) + 2 * (self.n - 1) + 1 + self.n - 1 + self.n - 1 + self.n - 1 + 5
                #state_dim = 2 + 2 + 1 + 2 * (self.n - 1) + 2 * (self.n - 1) + 1 + self.n - 1 + 3
                state_dim = self.obs_dim
                obs_n.append(np.zeros(state_dim))  # 归0
                reward_n.append(0)  # 回报0
                #done_n.append(False)  # 不终止
                done_n.append(True)# 个人认为应该终止
                if agent.adversary:
                 self.agents[i].state.p_pos = np.array([-10.0 for j in range(self.world.dim_p)])
                else:
                 self.agents[i].state.p_pos = np.array([-20.0 for j in range(self.world.dim_p)])

                self.agents[i].state.p_vel = np.zeros(self.world.dim_p)
                self.agents[i].state.c = np.zeros(self.world.dim_c)
                # guojing set
                self.agents[i].state.f = np.zeros(self.world.dim_p)  # 死亡干扰= 0
                self.agents[i].state.p_roll = np.zeros(self.world.dim_c)  # 死亡 滚转角=0
            else: # i存活
                obs_n.append(self._get_obs(agent))  # 获取当前状态
                ####guojing set
                for j in range(3):  # 获取当前输入3个动作
                    obs_n[i][-3 + j] = copy.deepcopy(action_n[i][j])

                #for j in range(2):  # 获取当前输入5个动作的前2个
                #    obs_n[i][-5 + j] = copy.deepcopy(action_n[i][j])
                #for j in range(3):  # 后3个=0
                #    obs_n[i][-3 + j] = 0
                reward_n.append(self._get_reward(agent))
                done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))

        # for i, agent in enumerate(self.agents):
        # print("action: ",action_n[i])
        # print("obs: ",obs_n[i])
        # input()

        # print("reward_n: ",reward_n)
        # print("done_n: ",done_n)
        # input()
        # all agents get total reward in cooperative case

        reward = np.sum(reward_n)  # 总回报，在对抗环境中，博弈的纳什均衡-回报=C常数
        if self.shared_reward:  # 是共享回报
            reward_n = [reward] * self.n  # 对总回报乘以n
        """guojing MARL对抗环境
        if self.shard_reward:
            reward_adv=[reward_n[re] for re in red_agent_num]
            reward_good=np.sum(reward_n)-reward_adv
        reward_n=[reward_adv,rewad_good]
            """
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
    """action的选取：①选取力 ②选取通信-无，应该加入 ③功能选取-攻击，干扰"""
    # 动作定义：连续2维力+连续1维滚转角速度+离散2维作战功能+连续1维通信 =  4维连续+2维离散
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
            # print("discrete")
        else:
            # print("continues")
            action = [action]
        #print ('总action=', action)
        # ① 是驱动力
        if agent.movable:
            # physical action
            if self.discrete_action_input:  # 离散动作输入？？
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:  # 没有离散动作输入
                if self.force_discrete_action:  #
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:  # 离散
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:       # 连续动作
                    """状态方程在这已经完成转移"""
                    # ①推力-加速度
                    if math.isnan(action[0][0]):
                        action[0][0] = 1
                    # 无人机不应该悬停
                    # agent.action.u = (action[0][0] + 1.0) /2.0
                    agent.action.u = action[0][0]
                    # print(agent.action.u)
                    # 赋予加速度
                    sensitivity = 5.0
                    if agent.accel is not None:
                        sensitivity = agent.accel
                    agent.action.u *= sensitivity  # 只能是速度了
                    # ②滚转角速度-转弯角速度 # 滚转角速度<23°/s  1/rad
                    if math.isnan(action[0][1]):
                        action[0][1] = 1
                    if abs(action[0][1]) > 1:
                        action[0][1] = 1 / action[0][1]
                    agent.action.r = action[0][1] * 23 * math.pi / 180    # 本身是0.1s的时间粒度
                    # agent.action.r = action[0][1] *230
                    # ③干扰-0-1分布判断
                    if math.isnan(action[0][2]):
                        action[0][2] = 1
                    if action[0][2] >= 0:
                        agent.action.f = 1
                    else:
                        agent.action.f = 0

                    #agent.action.u = action[0]  # 第1个动作是 驱动力
                    #if np.random.random(1)>0.5:
                    #    agent.action.f =  1# 干扰
                    #else:
                    #    agent.action.f = 0
                    #agent.action.r += math.atan(9.8/np.sqrt(np.sum(np.square(agent.action.u))) / math.atan(agent.action.u[1] / agent.action.u[0]))
            #print('驱动力:', action[0])
            # print("ddd %s",agent.action.u)
            action = action[1:]

        """#②是滚转角速度
        if agent.movable:
            # physical action
            if self.discrete_action_input:  # 离散动作输入？？
                agent.action.r = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.r[0] = -1.0
                if action[0] == 2: agent.action.r[0] = +1.0
                if action[0] == 3: agent.action.r[1] = -1.0
                if action[0] == 4: agent.action.r[1] = +1.0
            else:  # 没有离散动作输入
                if self.force_discrete_action:  #
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    d = np.argmax(action[0])
                    agent.action.r += action[0][d] * math.pi / 180
                    # print(action[0])
                    # agent.action.r[0] += action[0][0] - action[0][1]
                    # agent.action.r[1] += action[0][3] - action[0][4]
                else:
                    agent.action.r += action[0]  # 第2个动作是 滚转角速度
            #print('滚转角:', action[0])
            action = action[1:]

            # print("ddd %s",agent.action.r)
            #sensitivity = 5.0
            #if agent.accel is not None:
            #    sensitivity = agent.accel
            # agent.action.u *= sensitivity * self.world.dt  # 只能是速度了

        # ③是干扰
        # 我在想，有没有移动都可以去攻击，干扰？ - -以后可以加入一些固定的防御装置
            #干扰，共3个。可以是 a[0,0] 表述为：a[2]
            #离散的选择可以这样：
        if self.discrete_action_space:  # 离散攻击干扰
            d = np.argmax(action[0])  # d=最大的动作值的索引
            action[0][:] = 0.0  # 所有位置=0
            action[0][d] = 1.0  # 再将该位置=1，用列的action[0][0,0,0,1,0,0]表示 argmax action[0]
            agent.action.f = action[0][0] - action[0][1]  #1是干扰，2是不干扰

        else:   # 连续输入
            agent.action.f = action[0]  # 连续输入-可以用beta分布，再在a中二分选择
            # 剔除用过的action[0]
        #print('干扰:', action[0])
        action = action[1:]
        """

        # ④是通信
        if not agent.silent:
        # if agent.silent:
            # communication action
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
                        # numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的索引值。
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary) 创建观测器
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(800, 800)

        # create rendering geometry 创建渲染几何
        if self.render_geoms is None:  # 若不存在
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []  # 预置
            self.render_geoms_xform = []  #预置
            # zgy  *************************
            self.render_lines = []  # 预置直线
            for entity in self.world.entities:  # 对于世界中的所有实体
                """
                geom = rendering.make_circle(entity.size)  # 制造圆形几何
                geom_line = rendering.make_line((0, 0), (fire_range, 0))  # 制造直线几何
                xform = rendering.Transform()  # 初始变换记录  xform=[平移=0，旋转=0，尺度变换=1]
                if 'agent' in entity.name:  # 如果是智能体
                    geom.set_color(*entity.color, alpha=0.5)  # 设置混合颜色
                    geom_line.set_color(*entity.color, alpha=0.5)  # 设置直线的混合颜色
                else:  # 如果是障碍物
                    geom.set_color(*entity.color)  # 设置纯色
                    geom_line.set_color(*entity.color)  # 设置纯色
                geom.add_attr(xform)  # 将渲染几何变换结果数据加入物体几何 列表形式 geom中
                geom_line.add_attr(xform)  # 加入直线几何geom_line中
                self.render_geoms.append(geom)  # 追加物体几何
                self.render_geoms.append(geom_line)  # 追加直线几何
                self.render_geoms_xform.append(xform)  # 追加变换后结果参数

                # add geoms to viewer添加几何形状到监视器
            for viewer in self.viewers:
                viewer.geoms = []  # 预置
                # 向viewer.geoms里追加图形
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for geom_line in self.render_geoms:
                    viewer.add_geom(geom_line)
                """
                """对于所有agent，五角星的飞行结构，小扇形的攻击区，大扇形的受威胁区， 等大圆形的通信区和侦察区"""

                xform = rendering.Transform()  # 初始变换记录  xform=[平移=0，旋转=0，尺度变换=1]
                if 'agent' in entity.name:  # 如果是智能体
                     # 飞行结构-五角星
                    geom  = rendering.make_uav(entity.size)
                    # 小扇形-攻击区
                    geom_attact_sector = rendering.make_forward_sector(radius=fire_range, angle_start= -attack_angle/2, angle_end=attack_angle/2)  # 不需要等分360度，仅需等分attack_angle度
                    # 大扇形-威胁区
                    geom_defence_sector = rendering.make_forward_sector(radius=fire_range,angle_start=180-defense_angle/2,angle_end=180+defense_angle/2)  # 不需要等分360度，仅需等分attack_angle度
                    # 圆形-侦察区-干扰区
                    geom_explore = rendering.make_circle(jam_range)  # 圆形几何

                    geom.set_color(*entity.color, alpha=0.5)  # 设置混合颜色
                    geom_attact_sector.set_color(*entity.color, alpha=0.3)  # 设置直线的混合颜色
                    geom_defence_sector.set_color(*entity.color, alpha=0.05)  # 设置直线的混合颜色

                    geom_explore.set_color(*entity.color, alpha= 0.01)  # 透明

                else:  # 如果是障碍物
                    geom_explore = rendering.make_circle(2*entity.size)  # 圆形几何
                    geom_explore.set_color(*entity.color)  # 设置纯色
                    # geom_line.set_color(*entity.color)  # 设置纯色
                geom.add_attr(xform)
                geom_attact_sector.add_attr(xform)
                geom_defence_sector.add_attr(xform)
                geom_explore.add_attr(xform)
                self.render_geoms.append(geom)  # 追加物体几何
                self.render_geoms.append(geom_attact_sector)  # 追加直线几何
                self.render_geoms.append(geom_defence_sector)  # 追加物体几何
                self.render_geoms.append(geom_explore)  # 追加直线几何
                self.render_geoms_xform.append(xform)   # 追加变换后结果参数

                # add geoms to viewer添加几何形状到监视器
                for viewer in self.viewers:
                    viewer.geoms = [] # 预置
                    # 向viewer.geoms里追加图形
                    for geom in self.render_geoms:
                        viewer.add_geom(geom)
                    for geom_attact_sector in self.render_geoms:
                        viewer.add_geom(geom_attact_sector)
                    for geom_defence_sector in self.render_geoms:
                        viewer.add_geom(geom_defence_sector)
                    for geom_explore in self.render_geoms:
                        viewer.add_geom(geom_explore)

        # self.render_geoms[3].attrs(*entity.color, alpha=0.1)  # 更新颜色

        results = []  # 渲染输出结果预置列表

        # zgy
        agents = self.world.agents  # 获取世界中智能体参数
        le = len(self.viewers)
        for i in range(len(self.viewers)):  # 对每个view
            from multiagent import rendering  # 导入rendering
            # update bouds to center around agent 以agent为中心点更新边界
            cam_range = 2 # 变大范围
            if self.shared_viewer:  # =true
                pos = np.zeros(self.world.dim_p)  # 预置位置0矩阵pos
            else:  # =false
                pos = self.agents[i].state.p_pos  # 获取agent的位置矩阵
            # 在监视器里添加边界条件
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions-更新几何位置
            for e, entity in enumerate(self.world.entities):
                my_chi = np.zeros(1)  # 预置0矩阵-角度值
                #if abs(agents[e].state.p_vel[0]) < 1e-6 and abs(agents[e].state.p_vel[1]) < 1e-6:
                #    my_chi[0] = 0  # 速度小于一定值，=0
                if entity.death == True:
                    my_chi[0] = 0
                else:  # 用反tan函数 atan2求出速度的角度-弧度值
                   #my_chi[0] = math.atan2(agents[e].state.p_vel[1], agents[e].state.p_vel[0])
                    my_chi[0] = agents[e].state.course_angle  # 角度
                    # print(e,"的航向角是： ",agents[e].state.course_angle)
                # 更新颜色
                if entity.action.f == 1 and entity.state.f[0] >0:  # 开干扰
                    self.render_geoms[(e+1)*4-1].set_color(*entity.color,alpha=0.05)
                else:   # 关干扰
                    self.render_geoms[(e + 1) * 4 - 1].set_color(*entity.color, alpha=0.01)

                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)  # 更新位置信息
                self.render_geoms_xform[e].set_rotation(my_chi[0])  # 更新旋转角度
            # render to display or array-渲染显示或数组
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    # 在局部坐标系中创建受体场位置
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        # 圆形的接受域
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin 添加的起点
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        # 栅格感受域
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
# 对一批多智能体环境进行矢量化包装
# 假设所有环境都有相同的观察和行动空间
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
