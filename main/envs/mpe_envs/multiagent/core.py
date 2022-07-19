import numpy as np
import math
"""需要调用的子类class"""
# physical/external base state of all entites-#所有实体的物理/外部基本状态
class EntityState(object):
    # 定义初始化
    def __init__(self):
        # physical position  物理位置 2
        self.p_pos = None
        # physical velocity  物理速度 2
        self.p_vel = None
        # physical roll      滚转角 1
        self.p_roll = None
        # physical course   航向角 1
        self.course_angle = None

# state of agents (including communication and internal/mental state)
# agent的状态(包括通信和友军状态/自身状态)
class AgentState(EntityState):
    # 定义初始化
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance-通信方式
        self.c = None
        # 作战功能-可干扰次数
        self.f = None



# action of the agent-智能体agent动作
class Action(object):
    # 定义初始化
    def __init__(self):
        # physical action-物理动作-驱动力 2
        self.u = None
        #######guojing  set
        # 定义滚转角速度  1
        self.r = None
        # 作战功能-干扰   1
        self.f = None
        # communication action-通信动作 1
        self.c = None

# properties and state of physical world entity
# 物理（真实）世界实体（障碍物）的属性和状态
class Entity(object):
    def __init__(self):
        # name agent名字
        self.name = ''
        # properties-属性（类似于 在物理世界所占的区域尺寸）
        self.size = 0.050
        # entity can move / be pushed-
        # 实体（障碍物+agent）可以移动/被推-状态movable--不可移动
        self.movable = False
        # entity collides with others
        # 实体（障碍物+agent）与其他实体发生冲突-状态collide
        self.collide = True
        # material density (affects mass)-（障碍物）材质密度density(影响质量)
        self.density = 25.0
        # color-颜色（区分红蓝对抗等）
        self.color = None
        # max speed and accel-最大速度max_speed和加速度accel
        self.max_speed = None
        self.accel = None
        self.max_roll = 23.0  # 最大滚转角
        self.max_course = 180.0  # 最大航向角
        # state- （实体）的状态
        self.state = EntityState()
        # mass-（实体）的质量
        self.initial_mass = 1.2
        # 存活
        """ guojing set """
        # agent.death
        self.death = False

    # property 负责把一个方法变成属性调用的
    @property
    # 把质量self.initial_mass变成属性调用-mass()
    def mass(self):
        return self.initial_mass

# properties of landmark entities-地标(目的，障碍物等)实体的属性
class Landmark(Entity):
    # 定义初始化
     def __init__(self):
        super(Landmark, self).__init__()  # 继承了调用父类Landmark的初始化

# properties of agent entities-#代理agent实体的属性
class Agent(Entity):
    # 定义初始化  super() 函数是用于调用父类(超类)的一个方法。
    def __init__(self):
        super(Agent, self).__init__()  # 继承了调用父类Agent的方法
        # agents are movable by default-默认情况下，智能体agent是可移动的
        self.movable = True
        # cannot send communication signals-无法发送通信信号
        self.silent = False
        # cannot observe the world-无法观测世界
        self.blind = False
        # physical motor noise amount-物理的电机噪音
        self.u_noise = None
        # communication noise amount-通信噪声
        self.c_noise = None
        # control range-控制范围
        self.u_range = 1.0
        # state-智能体状态
        self.state = AgentState()
        # action-智能体动作
        self.action = Action()
        # script behavior to execute-要执行的剧本（分配的任务）行为
        self.action_callback = None
        """*********guojing set variable*************"""
        # 侦察范围
        self.scoutUAV = 1.0
        # 作战有效范围
        self.combatUAV = 0.5
        # 电子干扰范围
        self.electronic_interference = 1.0
        # 通信认为是集群内完全通信，并共享侦察范围内的敌方的状态



# multi-agent world-多智能体世界
class World(object):
    # 定义初始化
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        # 代理agent和实体（障碍物）列表(可以在执行时更改!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality-通信通道维度
        self.dim_c = 1
        # position dimensionality-位置维度-2维
        self.dim_p = 2
        # 作战功能维度
        self.dim_f = 1 # 作战2个维度
        # color dimensionality-颜色维度-3维
        self.dim_color = 3
        # simulation timestep-离散化微分变量dt=0.1
        self.dt = 0.1
        # physical damping-物理衰减
        self.damping = 0.25
        # contact response parameters-接触的响应参数
        self.contact_force = 1e+2  # 接触的限定力
        self.contact_margin = 1e-3  # 接触的限定边界

    # return all entities in the world
    # 返回世界中的所有实体-障碍物+智能体
    """属性"""
    @property  # property 负责把一个方法变成属性调用的
    def entities(self):
        # 智能体agent+障碍物landmarks的总和
        return self.agents + self.landmarks

    # return all agents controllable by external policies-返回所有可被外部策略控制的代理agent策略
    # 即所有未被分配任务的agent列表
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts-返回世界脚本（任务）控制的所有agent
    # 即所有被分配任务的agent列表
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    """方法"""
    # update state of the world-更新世界状态
    # 单步更新
    def step(self):
        # set actions for scripted agents-为脚本代理(任务agent)设置动作
        for agent in self.scripted_agents:  # 有任务agent-NPC
            # agent的任务分配动作标志
            agent.action = agent.action_callback(agent, self)  # 分配任务

        # gather forces applied to entities-收集应用于实体的力量
        # 收集应用于实体（障碍物+agent）的力量，预置相同维度的一维None向量
        p_force = [None] * len(self.entities)  # [1,2,3,...,n] n个实体
        # apply agent physical controls-应用代理agent物理控制
        # 生成 agent_i的物理力量-动量F_i
        p_force  = self.apply_action_force(p_force)  # 根据agent的动作-力 +环境中的阻力-噪声 = 真实环境-反馈的力
        # apply environment forces-应用环境的力量
        # 实体a与b在环境中获得的接触力p_force[a],p_force[b]
        p_force = self.apply_environment_force(p_force)  # 驱动力+碰撞力
        # integrate physical state-整合物理状态
        #  通过实体的速度获得实体的位置状态 x(t) = x(t) + vx(t) * dt
        self.integrate_state(p_force)    # 更新实体的位置+速度
        # update agent state-更新每个agent状态
        for agent in self.agents:
            # 更新通信状态 通信动作*噪声 ==> 不知道通信动作/状态怎么表示通信？
            self.update_agent_state(agent)  # 更新agent的通信状态

    # gather agent action forces-收集agent动作力量
    """重新定义力为1维合力，不是2维分力"""
    def apply_action_force(self, p_force):

        # set applied forces-设置应用力量
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        for i, agent in enumerate(self.agents):  # 对agent编号 [ [0,agent0];[1,agent1];...;[n,agentn] ]
            u = np.zeros(2)
            epislon = 1e-6
            if agent.movable:  # 如果agent能移动
                #  if agent的物理噪音=True（意味着在物理世界中agent有噪声-干扰）
                #  根据agent的动作的物理动作空间的维度agent.action.u.shape，以元组形式（tuple）导入np.random.rand中
                #  噪声生成公式： noise = 随机数(agent的动作的物理动作空间的维度) * 物理噪声u_noise
                #         否则  noise = 0.0
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                # agent应用于环境的力量p_force = agent的物理动作action.u + 生成的噪声noise
                F = agent.action.u + noise      # 动作力 + 噪声 = 真实施加的力
                # 滚转角roll=默认是弧度值
                agent.state.p_roll += agent.action.r * self.dt
                # 滚转角roll约束
                if agent.max_roll is not None:
                    if abs(agent.state.p_roll) > agent.max_roll * math.pi / 180:
                        agent.state.p_roll[0] =(agent.state.p_roll / abs(agent.state.p_roll)) * agent.max_roll * math.pi / 180
                # 航向角速度-math.tan是根据弧度值计算
                # print('agent.action.u ',agent.action.u ,'agent.state.p_roll',agent.state.p_roll)
                agent_course_angular_velocity = 9.81 * 1.2 / ((agent.action.u + epislon) * self.dt) * math.tan(agent.state.p_roll)
                # 航向角-弧度值
                agent.state.course_angle += agent_course_angular_velocity * self.dt
                # 航向角约束- 弧度值
                if agent.max_course is not None:
                    if abs(agent.state.course_angle) > agent.max_course * math.pi /180:
                        agent.state.course_angle[0] = (agent.state.course_angle / abs(agent.state.course_angle)) * agent.max_course * math.pi /180
                # 计算Fx，Fy分力
                u[0] = F * math.sin(agent.state.course_angle)
                u[1] = F * math.cos(agent.state.course_angle)
                p_force[i] = u
                """
                if i==100:
                    print("滚转角速度：", agent.action.r / math.pi * 180)
                    print("滚转角：",agent.state.p_roll/ math.pi * 180)
                    print("航向角速度：", agent_course_angular_velocity/ math.pi * 180)
                    print("航向角：",agent.state.course_angle/ math.pi * 180)
                    print("合力-加速度力：", F)
                    print("x方向分力加速度力：", u[0])
                    print("y方向分力加速度力：", u[1])
                    print(i)
                """
        return p_force

    # gather physical forces acting on entities-收集作用于实体（agent+障碍物）的物理力量
    def apply_environment_force(self, p_force):  # p_force是 加入了自身动作力的环境力
        # simple (but inefficient) collision response-#简单(低效)的冲突响应
        # 对世界中所有实体（agent+障碍物）编号[[0, entities_0];[1, entities_1];...;[n, entities_n]]
        p = p_force
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                # a在b之前
                if(b <= a): continue  # 如果 b的序号小于a，跳过此次迭代
                # 获取两个实体(entity_a,entity_b)之间的任何接触的碰撞力
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)  # 给出ab的接触阻力
                if(f_a is not None):  # 如果f_a不是None
                    if(p_force[a] is None): p_force[a] = 0.0  # 如果p_force[a]是None。则赋0
                    p_force[a] = f_a + p_force[a]   # 累加：f_a + p_force[a]
                if(f_b is not None):  # 如果f_b不是None
                    if(p_force[b] is None): p_force[b] = 0.0  # 如果p_force[b]是None，则赋0
                    p_force[b] = f_b + p_force[b]  # 累加：f_b + p_force[b]
            # 给出真实的力产生的航向角
            # p[a]=p_force[a] - p[a]
           # entity_a.state.course_angle += math.atan(p[a][0] / p[a][1])
        #  返回 p_force 实体的物理力量
        # p_force是一种 自身驱动力 + 与其他agent的 弹性阻力-与最小距离有关

        return p_force

    # integrate physical state-集成物理状态

    def integrate_state(self, p_force):  # a是滚转角速度
        for i,entity in enumerate(self.entities):  # 实体编号：[0,entities_0],[1,entities_1 ],...,[ n , entities_n ]
            if not entity.movable: continue  # 如果实体entities_i不能移动，则此次循环跳过
            # 模拟空气阻力，没有额外力影响，则速度衰减
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)  # 实体状态-速度= 速度 * ( 1 - 物理衰减damping)
            if (p_force[i] is not None):  # 如果实体i的物理力量p_force[i]有值
                # 有额外力=驱动力+弹性阻力 dv =F/m·dt   使得: V <— V + dv 更新速度
                ###############gujing set ###############
                """theta_vel = 9.8 / entity.state.p_vel * math.tan(entity.state.p_roll)  # 航迹偏角速度
                theta += theta_vel * self.dt  # 航迹偏角
                entity_vel[0] = (p_force / entity.mass * self.dt)  # x方向的增速度
                entity_vel[1] = (p_force / entity.mass * self.dt)  # y方向的增速度
                entity.state.p_vel[0] += entity_vel[0] * math.sin(theta)   # 更新速度
                entity.state.p_vel[1] += entity_vel[1] * math.cos(theta)   # 更新速度
                """
                # ④航向角速度
                #if v_sum < 1e-4:
                #    dv_angle = 0
                #else:
                #    dv_angle = 9.8 / v_sum * math.tan(entity.state.p_roll)
                # 更新航向角-加入滚转角修正偏差
                #v_angle += dv_angle * self.dt
                # 更新速度分量
                #entity.state.p_vel[0] = v_sum * math.sin(v_angle)
                #entity.state.p_vel[1] = v_sum * math.cos(v_angle)


                entity.state.p_vel += p_force[i] / entity.mass * self.dt  # 实体i的动量F/该实体i的质量m * 微分变量dt + 实体i的速度p_vel
            if entity.max_speed is not None:  # 如果实体i有最大速度限制v_max
                # 计算速度和 v=sqrt(vx^2+vy^2)
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:  # 如果速度speed超过最大速度max_speed
                    # 归一化速度分量vx,vy作为最大速度的权重
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            #
            entity.state.course_angle[0] = math.atan2(entity.state.p_vel[1], entity.state.p_vel[0])
            # 获得位置坐标
            # last_pos = entity.state.p_pos
            entity.state.p_pos += entity.state.p_vel * self.dt  # x(t) = x(t) + vx(t) * dt
            # print(i, entity.state.p_pos)
            if entity.death == False:
                if abs(entity.state.p_pos[0]) > 2.0:
                    entity.state.p_pos[0] = 2.0 * (entity.state.p_pos[0] /abs(entity.state.p_pos[0]))
                if abs(entity.state.p_pos[1]) > 2.0:
                    entity.state.p_pos[1] = 2.0 * (entity.state.p_pos[1] /abs(entity.state.p_pos[1]))

            # 依次递减
            # 等于0不减了
            if entity.state.f[0] < 0:
                entity.state.f = np.zeros(1)
            else:
                entity.state.f -= entity.action.f

    # 更新agent的状态
    def update_agent_state(self, agent):
        # set communication state (directly for now)-设置通信状态(暂时直接)
        # 如果agent能进行通信
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)  # 通信状态为0矩阵
        else:  # 如果agent不能通信
            # 噪声 = 以agent通信动作维度（重组1行N列的元组形式）为随机种子产生随机数 * 智能体的通信噪声 c_noise
            # 如果没有噪声c_noise, 则 noise = 0
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise  # agent的通信状态 = agent的通信动作 + 噪声

    # get collision forces for any contact between two entities
    # 获取两个实体之间的接触的碰撞力
    def get_collision_force(self, entity_a, entity_b):  #
        # 判断实体entity_a和entity_b是否与其他实体碰撞
        if (not entity_a.collide) or (not entity_b.collide):  # 如果ab都不能碰撞， 返回None
            return [None, None]  # not a collider  没有碰撞，返回[None, None]
        if (entity_a is entity_b):  # 如果entity_a与entity_b是同一实体
            return [None, None]  # don't collide against itself 同一实体不会碰撞，返回[None, None]
        # compute actual distance between entities-计算实体之间的实际距离
        """欧式距离"""
        # 将entity_a的状态state中位置数据 与entity_b的状态state中位置数据做差得到delta_pos
        # 距离矢量
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        # 平方，在求和，最后求平方根，得到了真实距离dist-标量
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        epsilon_dist = 1e-6  # 防止NAN
        # minimum allowable distance-最小允许距离
        # 最小非碰撞距离dist_min = 实体a的尺寸 + 实体b的尺寸
        dist_min = entity_a.size + entity_b.size
        # softmax penetration-softmax规范化-对所有距离差取指数形式再计算归一化比例
        k = self.contact_margin  # contact_margin 接触的限定边界
        # np.logaddexp： 计算log(exp(x1) + exp(x2))
        # print('dist',dist,'dist_min',dist_min,'k',k)
        penetration = np.logaddexp(0, - (dist - dist_min) / k ) * k
        #  接触限定力 * 位置矢量 / 绝对距离 * penetration穿透力
        #if dist == 0:
        #    dist = 1
        force = self.contact_force * delta_pos / (dist+ epsilon_dist) * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        # 返回实体a与b的接触限定力
        return [force_a, force_b]