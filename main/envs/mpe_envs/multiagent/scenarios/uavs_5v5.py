import numpy as np
from envs.mpe_envs.multiagent.core import World, Agent, Landmark
from envs.mpe_envs.multiagent.scenario import BaseScenario
import math
import copy

attack_angle = 90  # 攻击区角度
defense_angle = 90 # 防守区角度
fire_range = 0.3  # 攻击距离
comput_range = 0.6  # 计算距离-无效量
#guojing set
jam_range = 0.7  # 干扰距离

class Scenario(BaseScenario):
    # 定义环境
    def make_world(self):
        print("**********guojing set scenario*************")
        world = World()  # 获取gym的粒子环境物理定义
        # 设定环境中智能体参数
        world.dim_c = 1  # 通信-干扰-滚转角维度
        world.dim_p = 2  # 位置维度
        world.dim_f = 1 # 46个滚转角速度-23——23
        num_red_agents = 4 # 红方agent数量
        num_blue_agents = 8 # 蓝方agent数量
        # num_good_agents = num_blue_agents
        # num_adversaries = num_red_agents
        # num_agents = num_adversaries + num_good_agents
        num_agents = num_red_agents + num_blue_agents  # 总无人机数
        num_landmarks = 0  # 障碍物

        # copy.deepcopy是常说的复制，即复制完全相同结构的对象，且是独立的个体。
        self.num_blue = copy.deepcopy(num_blue_agents)  # 赋
        self.num_red = copy.deepcopy(num_red_agents)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True # 与其他agent发生冲突
            agent.silent = True  # 与其他agnet通信
            agent.u_noise = True # 物理电机噪声
            agent.adversary = True if i < num_red_agents else False  # 前i个是red方，剩下是blue方
            agent.size = 0.04 if agent.adversary else 0.04  # 尺寸大小
            #agent.accel = 3.0 if agent.adversary else 2.0
            agent.accel = 3.0 if agent.adversary else 3.0 # 加速度
            #agent.accel = 20.0 if agent.adversary else 25.0
            #agent.max_speed = 1.5 if agent.adversary else 1.2
            agent.max_speed = 5.0 if agent.adversary else 5.0  # 最大速度
            #agent.max_speed = 1.0 if agent.adversary else 0.3  ###changed by liyuan
            """######guojing set agent"""
            agent.max_roll = 23.0   # 最大滚转角
            agent.max_course = 180.0  # 最大航向角


            # agent.chi = np.array([0.1,0])  # 初始固定角度-可以是随机的
            agent.chi = np.random.random([1,2])*0.5  # 随机角度
            if agent.adversary:
                agent.lock_num=[0 for j in range(num_blue_agents)]  # one-hot编码-集群内
            else:
                agent.lock_num=[0 for j in range(num_red_agents)]  # 同上
        # add landmarks  # 障碍物
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)  # 初始化环境
        return world

    # 重置环境，是重置了，颜色，以及随机初始状态

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            # agent颜色-蓝np.array([0.35, 0.35, 0.85]) 红np.array([0.85, 0.35, 0.35])
            agent.color = np.array([0.35, 0.35, 0.85]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])  # 黑色
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1.5, +1.5, world.dim_p)  # 随机初始位置
            agent.state.p_vel = np.zeros(world.dim_p)  # 初始0
            agent.state.c = np.zeros(world.dim_c) # 初始0
            # 航向角
            agent.state.course_angle = np.zeros(world.dim_c)  #
            agent.death = False  # 初始存活
            """guojing set"""
            # agent.state.p_roll = 0  # 滚转角
            agent.state.p_roll = np.zeros(world.dim_c)
            # 可干扰次数
            agent.state.f = np.array([15])  # 初始5

            # agent.course_angle = 0  # 航向角  没有参照物，定义飞机的前进方向和正北方向之间的夹角
            # 忽略侧偏角，并认为航向角与航迹偏角一致

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

        if agent.adversary:  # 重新定义one-hot编码
                agent.lock_num=[0 for j in range(self.num_blue)]
        else:
                agent.lock_num=[0 for j in range(self.num_red)]


    # 基准数据
    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:  # 红方
            collisions = 0  # 碰撞
            for a in self.good_agents(world):  # 蓝色
                # 判断agent与a的是否发生对抗
                # 蓝色攻击了红方，并且蓝色
                if self.attack_uav(a, agent) and a.death == False:
                    collisions += 1  # 可以看作是被集火次数  ./总敌数 是集火率
            # 返回集火次数
            return collisions
        else:  # 蓝方，####应该也被计算
            return 0

    '''
    def is_collision(self, agent1, agent2):
        if agent1.death or agent2.death:
            return False
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        #dist_min = agent1.size + agent2.size
        dist_min = 0.1
        return True if dist < dist_min else False
    '''

    ##liyuan: compute the number of locking number of the agent
    # 累计计算agent被敌方agent分别攻击次数-在总回合
    def entity_lock_num(self, agent, world):
        opponent = []
        if agent.adversary:  #红方
            opponent = self.good_agents(world)  # 红方的敌方是蓝方
        else:
            opponent = self.adversaries(world)  # 蓝方的敌方是红方

        for i, opp in enumerate(opponent):  # 敌方
            if self.attack_uav(opp,agent):  # 判断敌方是否攻击agent
                agent.lock_num[i] += 1  # 敌方第i个集火agent次数
            else:
                agent.lock_num[i] += 0

    ###############################################
    # 计算agent攻击了哪个敌方agent
    def attack_compute_num(self, agent, world):
        opponent = []

        if agent.adversary:  # 红方
            opponent = self.good_agents(world)  # 红方的敌方是蓝方
        else:
            opponent = self.adversaries(world)  # 蓝方的敌方是红方
        attack_num = []  # 预置
        for i, opp in enumerate(opponent):  # 敌方
            if self.attack_uav(agent, opp):  # 判断敌方是否攻击agent
                attack_num.append(1)
                # agent.lock_num[i] += 1  # 敌方第i个集火agent次数
            else:
                attack_num.append(0)
                # agent.lock_num[i] = 0
        return attack_num
    ###############################################
    # 计算agent受到敌方的集火
    def lock_compute_num(self, agent, world):
        opponent = []
        lock = []
        if agent.adversary:  #红方
            opponent = self.good_agents(world)  # 红方的敌方是蓝方
        else:
            opponent = self.adversaries(world)  # 蓝方的敌方是红方

        for i, opp in enumerate(opponent):  # 敌方
            if self.attack_uav(opp,agent):  # 判断敌方是否攻击agent
                lock.append(1)  # 敌方第i个集火agent次数
            else:
                lock.append(0)
        return lock
    ######################################################
    ###liyuan: True if agent1 win, False for others
    # 冲突判断-agent1攻击agent2的返回值
    ###guojing: 修改定义，认为是满足了操作的预要求，为适应所有操作，进行魔改，[0,0,0,0]进行判断
    """定义attack_uav是攻击的计算函数"""
    def attack_uav(self, agent1, agent2):
        if agent1.death or agent2.death:
            return False

        ###liyuan:judged by angle
        delta_pos = agent2.state.p_pos - agent1.state.p_pos  # 相对位置
        distance = np.sqrt(np.sum(np.square(delta_pos)))  # 欧氏距离
        if distance <= 1e-5:  # 距离太近，不符合物理定义，不能判断冲突
            return False
        
        agent1_chi = [agent1.state.p_vel[0],agent1.state.p_vel[1]]  # agent1的速度矢量
        # 速度默认水平方向
        if abs(agent1.state.p_vel[0]) < 1e-5 and abs(agent1.state.p_vel[1])<1e-5:
            agent1_chi[0] = 0.1
            agent1_chi[1] = 0
        agent2_chi = [agent2.state.p_vel[0],agent2.state.p_vel[1]]  # agent2的速度矢量
        # 速度太小默认是0
        if abs(agent2.state.p_vel[0]) < 1e-5 and abs(agent2.state.p_vel[1])<1e-5:
            agent2_chi[0] = 0.1
            agent2_chi[1] = 0

        agent1_chi_value = np.sqrt(np.sum(np.square(agent1_chi)))  # 求速度矢量的合速度标量
        # cross这块对！---
        """攻击防守角的计算，只需要通过agent1与2的位置矢量与agent1(2)的速度矢量计算向量夹角"""
        agent1_cross = (delta_pos[0]*agent1_chi[0]+delta_pos[1]*agent1_chi[1])/(distance*agent1_chi_value)
        """ guojing set """
        """
        攻击角的计算需要： 
        agent1的位置agent1.state.p_pos
        agent1与2的位置矢量delta_pos
        agent1与2的欧氏距离distance
        agent1的位置与(0,0)的欧式距离distance_origin = np.sqrt(np.sum(np.square(agent1.state.p_pos))) 
        1. agent1的位置与(0,0)的位置矢量与agent1与2的位置矢量的夹角
        agent1_angle_origin_value=(agent1.state.p_pos[0]*delta_pos[0]+agent1.state.p_pos[1]*delta_pos[1])/(distance_origin*delta_pos)
        agent1_angle_origin =  math.acos(agent1_angle_origin_value)
        2. agent1的位置与(0,0)的位置矢量与水平方向位置矢量(x,0)的夹角
        计算agent1的速度矢量与水平方向矢量(x,0)的夹角  
        agent1_course_ = agent1_chi 
        agent1_angle = math.acos( )     
        """


        if agent1_cross < -1:
           agent1_cross  = -1
        if agent1_cross > 1:
           agent1_cross = 1

        agent1_angle = math.acos(agent1_cross)  # acos() 返回x的反余弦弧度值

        agent2_chi_value = np.sqrt(np.sum(np.square(agent2_chi)))
        agent2_cross = (-delta_pos[0]*agent2_chi[0]-delta_pos[1]*agent2_chi[1])/(distance*agent2_chi_value)
        if agent2_cross < -1:
           agent2_cross  = -1
        if agent2_cross > 1:
           agent2_cross = 1
        agent2_angle = math.acos(agent2_cross)  # agent2的弧度值

        revised_defense = 180-defense_angle/2  # 平面的威胁区
        # 在火力范围内   受威胁角的角度值是大于防守角度-在威胁区内   攻击角的角度值小于攻击角度-在攻击区内
        if distance < fire_range and agent2_angle*180/math.pi>revised_defense and agent1_angle*180/math.pi<attack_angle/2:
            return True
        #elif distance < fire_range and agent2_angle*180/math.pi<attack_angle/2 and agent1_angle*180/math.pi>revised_defense:
            #return True,2
        #else:
        return False
    
    ###liyuan: True if agent1 win, False for others
    # 同样是冲突检测，不同的是可以是agent1的攻击范围hit_range不同
    """定义 will_hit是干扰的判定函数、
    干扰判定不需要角度，即干扰范围是一个圆，距离即可
    """
    def jam_uav(self, agent1, agent2, hit_range = jam_range):
        if agent1.death or agent2.death:
            return False

        ###liyuan:judged by angle
        delta_pos = agent2.state.p_pos - agent1.state.p_pos
        distance = np.sqrt(np.sum(np.square(delta_pos)))
        if distance <= 1e-5:
            return False
        if distance < hit_range:
            return True
        #elif distance < hit_range and agent2_angle*180/math.pi<attack_angle/2 and agent1_angle*180/math.pi>revised_defense:
            #return True,2
        #else:
        return False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    ##########guojing set##########
    # 奖励的设计，比较重要，基于移动，攻击和干扰进行动态计算奖励值
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.red_reward(agent, world) if agent.adversary else self.blue_reward(agent, world)
        return main_reward

    ############设定了蓝方的reward-主要包括了 位移奖励，攻击奖励，干扰奖励，次要是生存奖励(忽略)，边界限定
    def blue_reward(self, agent, world):
        rew = 0  # 奖励存储
        ####added by liyuan
        # ①生存奖励-死亡后所有奖励归0
        if agent.death == True:
            return 0
        #################################################
        # shape = True  #奖励与距离相关标志
        adversaries = self.adversaries(world)  # 敌方agent数据-red
        blue_agents = self.good_agents(world)  # blue方
        # ②攻击奖励-击中敌方agent获+奖励，（暂不考虑击中友方agent获-奖励）
        # self.compute_lock_num(agent, world)
        rew_attack = 0 # 预置0
        attack_red_num = []
        if agent.collide and agent.death == False:  # 冲突-碰撞？
            for i, bagents in enumerate(blue_agents):
                if bagents.death== False:
                    if bagents is agent:
                        agent_attack_red = self.attack_compute_num(agent,world)  # 攻击agent标志
                    # continue
                    else:
                        attack_red_num += self.attack_compute_num(bagents,world)  # 攻击agent标志
                else:
                    attack_red_num += [0 for _ in range(len(agent.lock_num))]  # 归0
            for i, age in enumerate(adversaries):
                # if agent_attack_red[i] != 0 and attack_red_num[i] != 0:  2人
                if agent_attack_red[i] !=0:  # 1人
                     rew_attack += 1
                   # 判断攻击几个人，给几倍的奖励
                # rew += rew_attack*5  # 2个blue共同攻击1个red才有攻击奖励
            rew += 20 * rew_attack

        # ④干扰奖励-干扰敌方获+奖励，默认不干扰友方
        if agent.action.f > 0:
            agent_jam = []  # 预置干扰位， 可以同时干扰在范围内的所有敌方
            if agent.collide and agent.death == False:  # 冲突-碰撞 且存活
                for i, a in enumerate(adversaries):
                    if self.jam_uav(agent,a) and agent.state.f[0] > 0:  # agent干扰a
                        # 如果在干扰范围内，是否选择干扰，因为同时满足干扰和攻击范围，取决于2个动作选择
                        agent_jam.append(1)
                    else:  #未干扰
                        agent_jam.append(0)
                # agent.state.f = agent.state.f - 1  # 每次干扰次数-1
                rew += sum(agent_jam)*0.5  # 干扰了sum个敌方，有几倍的奖励

        # for i in range(len(attack_red_num)):  #
        # ③被攻击奖励-无论是（暂不考虑友方）还是敌方攻击自己都获-奖励
        # 计算agent被敌方每个agent集火次数
        agent_lock = self.lock_compute_num(agent, world)
        if agent.collide and agent.death == False:  # 冲突-碰撞？
            # if sum(agent_lock) >= 2:  # 先判断集火agent的敌方是否超过2个
            if sum(agent_lock) >= 1:  # 1人
                for i, a in enumerate(adversaries):  # 再判断集火的agent是否都存活
                    ###changed by liyuan
                    # if self.is_collision(a, agent) and a.death == False:
                    ####guojing set    条件：敌方2个a攻击agent 才能击毁agent
                    if a.death == True :
                        agent_lock[i] = 0  # 死亡则归0
            # if sum(agent_lock) >= 2:  # 重新计算，存活a是否有2个集火
            if sum(agent_lock) >= 1:  # 1人
                rew -= sum(agent_lock) * 8  # """奖励大小应该调节"""
                # 击毁后即是奖励=0，但在st死亡，st+1的rt+1奖励应该是st，at的奖励，所以是有奖励
                agent.death = True
                return rew
        # ⑤被干扰奖励-被敌方干扰获-奖励
        agent_dejam = []
        if agent.collide and agent.death == False:  # 冲突-碰撞？
            for i, a in enumerate(adversaries):
                if self.jam_uav(a, agent) and a.action.f > 0 and  a.state.f[0] > 0:  # agent干扰a
                    # 如果在干扰范围内，是否选择干扰，因为同时满足干扰和攻击范围，取决于2个动作选择
                    agent_dejam.append(1)
                else:  #未干扰
                    agent_dejam.append(0)
            rew -= sum(agent_dejam)*0.2  # 被sum个敌方干扰

        # ⑥位移奖励-存活时间越长获+奖励越多，但若未击杀敌方苟活获极大-奖励
        # 暂定，没有位移奖励，这个很难确定如何奖励，
        # ⑦边界奖励
        # 定义边界bound子函数-巨大的边界惩罚
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.5:
                return 10
            return 100
        # 横纵坐标的边界限定 0, 10, [10,100]
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
            #if x > 1.5:
                # agent.death = True
                # agent.state.p_pos = np.random.uniform(-1.5, +1.5, 2)
            #    return rew
        

        #for p in range(world.dim_p):
        #    x = abs(agent.state.p_pos[p])
        #    if (x > 1.0):
        #        rew -= 20
        #       break
        ## Agents are negatively rewarded if caught by adversaries
        # 如果被对手抓住，agent就会得到负面奖励

        #shape = False

        '''
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                ###changed by liyuan
                if adv.death == True:
                    continue
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        '''

        

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries


        return rew

    # red方的奖励
    def red_reward(self, agent, world):
        ####added by liyuan
        # ①生存奖励
        if agent.death == True:
            return 0
        # Adversaries are rewarded for collisions with agents
        rew = 0
        #shape = False
        shape = True
        #
        agents = self.good_agents(world)  # blue方
        adversaries = self.adversaries(world)  # red方
        # ②距离奖励-red偏向于进攻
        if shape:
            dis = []
            for a in agents:
                if a.death == False:
                    dis.append(np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))))
            if len(dis) > 0:
                rew -= 0.1 * min(dis)
        
        # ③攻击奖励-red攻击blue的奖励
        red_attack = []
        agent_attack = []
        rew_attack = 0
        self.entity_lock_num(agent, world)  # 累计了blue方分别攻击agent的次数
        # red攻击blue的奖励
        if agent.collide and agent.death == False:  #
            for _,adv in enumerate(adversaries):  #
                ###########guojing set
                # 2个red-adv同时攻击1个blue-ag，攻击才有效
                if adv.death ==False:
                    if adv is agent:
                        agent_attack = self.attack_compute_num(adv,world)  # agent攻击了blue
                    # continue
                    else:
                        red_attack += self.attack_compute_num(adv,world)  # adv攻击了blue
                else:
                    red_attack += [0 for _ in range(len(agent.lock_num))]
            for i, red in enumerate(agents):  # 判断
                # if agent_attack[i] != 0 and red_attack[i] != 0:  # 2人
                if agent_attack[i] != 0:  # 1人
                    rew_attack += 1
            # rew += rew_attack*4 #攻击奖励
            rew += 20 * rew_attack

        #④被攻击奖励-被2个blue集火才算死亡
        # 先计算被集火次数
        red_agent_lock = self.lock_compute_num(agent,world)  #
        if agent.collide:
            # if sum(red_agent_lock)>=2:
            if sum(red_agent_lock) >= 1:  # 1人
                for i,ags in enumerate(agents):
                    if ags.death == True:
                        red_agent_lock[i] = 0
                # if sum(red_agent_lock)>=2:  #被2个blue集火才算死亡
                if sum(red_agent_lock) >= 1:  # 1人
                    agent.death = True # 死亡
                    rew -=  5 * sum(red_agent_lock)
                    return rew
                    #if self.is_collision(ag,adv) and ag.death == False and adv.death == False:
                    #if self.ag.lock_num[i]>=3 and ag.death == False and adv.death == False:
                    #    if not (adv is agent):
                    #       rew -= 2

        ###if the red agent is eatten
        #if agent.collide:
        #    for i,ag in enumerate(agents):
        #        if self.is_collision(ag, agent) and ag.death == False:
        #        #if ag.death == False and agent.lock_num[i]>=3:
        #            agent.death = True
        #            rew -= 4
        #            break
        # ④干扰奖励-干扰敌方获+奖励，默认不干扰友方
        if agent.action.f > 0:
            agent_jam = []  # 预置干扰位， 可以同时干扰在范围内的所有敌方
            if agent.collide and agent.death == False:  # 冲突-碰撞 且存活
                for i, ag in enumerate(agents):
                    if self.jam_uav(agent, ag) and agent.state.f[0] > 0:  # agent干扰a
                        # 如果在干扰范围内，是否选择干扰，因为同时满足干扰和攻击范围，取决于2个动作选择
                        agent_jam.append(1)
                    else:  # 未干扰
                        agent_jam.append(0)
                # 干扰次数在环境中转移了
                # agent.state.f = agent.state.f - 1
                rew += sum(agent_jam) * 0.5   # 干扰了sum个敌方，有几倍的奖励

        # ⑤被干扰奖励-被敌方干扰获-奖励
        agent_dejam = []
        if agent.collide and agent.death == False:  # 冲突-碰撞？
            for i, ag in enumerate(agents):
                if self.jam_uav(ag, agent) and ag.action.f > 0 and ag.state.f[0] > 0:  # agent干扰a
                    # 如果在干扰范围内，是否选择干扰，因为同时满足干扰和攻击范围，取决于2个动作选择
                    agent_dejam.append(1)
                else:  # 未干扰
                    agent_dejam.append(0)
            rew -= sum(agent_dejam) * 0.2  # 被sum个敌方干扰
        # ⑤边界设定奖励
        for adv in adversaries:  # 红方
            if adv.death == False:
                exceed = False
                for p in range(world.dim_p):
                    x = abs(adv.state.p_pos[p])
                    if (x > 1.5):
                        exceed = True
                        break
                if adv is agent and exceed == True:
                    rew -= 100
                    # agent.state.p_pos = np.random.uniform(-1.5, +1.5, 2)
                    break
                else:
                    rew -= 0

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        other_chi = []
        other_roll = []
        our_chi = []
        # our_roll = []
        # 干扰次数
        other_jam = []

        my_chi = np.zeros(1)
        if abs(agent.state.p_vel[0])<1e-5 and abs(agent.state.p_vel[1])<1e-5:
            my_chi[0] = 0
        else:
            my_chi[0] = math.atan2(agent.state.p_vel[1],agent.state.p_vel[0])
        our_chi.append(my_chi)         # 航向角

        # our_chi = np.array(our_chi)
        # our_roll= # 滚转角
        # our_roll = np.array(our_roll)
        temp_agents=[]  # temp_agents排序：先红方后蓝方
        for agent_i in world.agents:
            if agent_i.adversary == agent.adversary:
                temp_agents.append(agent_i)
        for agent_i in world.agents:
            if agent_i.adversary != agent.adversary:
                temp_agents.append(agent_i)

        for other in temp_agents:
            if other is agent: continue            
            ###changed by liyuan
            if other.death:  # 其他agent
                comm.append(np.zeros(world.dim_c))
                other_pos.append(np.zeros(world.dim_p))
                other_vel.append(np.zeros(world.dim_p))
                tmp_chi = np.zeros(1)
                other_chi.append(tmp_chi)
                other_roll.append(np.zeros(1))
                other_jam.append(np.zeros(1))
            else:
                comm.append(other.state.c)  # 通信状态
                other_pos.append(other.state.p_pos - agent.state.p_pos)  # 相对于agent相对位置矢量
                #if not other.adversary:
                other_vel.append(other.state.p_vel)  # 速度

                tmp_chi = np.zeros(1)  #
                if abs(other.state.p_vel[0])<1e-5 and abs(other.state.p_vel[1])<1e-5:
                    tmp_chi[0] = 0
                else:
                    tmp_chi[0] = math.atan2(other.state.p_vel[1],other.state.p_vel[0])
                other_chi.append(tmp_chi)  # 航向角
                other_roll.append(other.state.p_roll) # 滚转角
                other_jam.append(other.state.f) # 干扰次数
        # other_roll = np.array(other_roll)

        #action_number = [np.zeros(5)]
        action_number=[np.zeros(3)]  # 1个力+1个滚转角速度+ 1个干扰 = 3
        pv = len(agent.state.p_vel)  # 2      2  自身位置
        pp = len(agent.state.p_pos)  # 2      2  自身速度
        proll=len(agent.state.p_roll)# 1      1  自身滚转角
        our_jam = len(agent.state.f) # 1      1  自身干扰次数
        ep = len(entity_pos)         # 2m     0  障碍物相对位置
        op = len(other_pos)          # 2(n-1) 22 其他agent相对距离
        ov = len(other_vel)          # 2(n-1) 22 其他agent的速度
        oc = len(our_chi)            # 1      1  自身航向角度
        ohc = len(other_chi)         # n-1    11 其他agent航向角度
        ###########
        ohroll = len(other_roll)     # n-1    11 其他agent滚转角度
        ohjam = len(other_jam)       # n-1       其他agent干扰次数
        an = len(action_number)      # 5      5  动作数量=5
        ##########
        """# 以前的状态：满状态  2 + 2 + 1 + 1 + 0 + 2*(n -1 + n -1) +n -1 + n-1 + 1 + n -1 + 3 = 
                              2 + 2 +  1 + 1 + 2 * (self.n - 1) + 2 * (self.n - 1) + 1 + self.n - 1 + self.n - 1 + self.n - 1 + 3 =   
        """
        # all_shape = np.concatenate(
        #                            [agent.state.p_vel] +
        #                            [agent.state.p_pos] +
        #                            [agent.state.p_roll] +
        #                            [agent.state.f] +
        #                            entity_pos +
        #                            other_pos +
        #                            other_vel +
        #                            other_roll +
        #                            other_jam +
        #                            our_chi +
        #                            other_chi +
        #                            action_number)


        all_shape = np.concatenate([agent.state.p_vel] +
                                   [agent.state.p_pos] +
                                   [agent.state.f] +
                                   entity_pos +
                                   other_pos +
                                   other_vel +
                                   our_chi +
                                   other_chi +
                                   action_number)
        """ash = all_shape.shape[0]
        print('pv=', pv)
        print('pp=', pp)
        print('ep=', ep)
        print('op=', op)
        print('ov=', ov)
        print('oc=', oc)
        print('ohc=', ohc)
        print('an=', an)
        print('all_shape=',all_shape)
        print('ash=', ash)"""
        #comm.append(other.state.c)
            #other_pos.append(other.state.p_pos - agent.state.p_pos)
            #if not other.adversary:
                #other_vel.append(other.state.p_vel)
        return all_shape
# m个障碍物 n个agent
# 2+2+2m+2(n-1)+2(n-1)+1+(n-1)+5 = 6n-6+1+5+4+2m = 2m+5n+5 = 2*0+5*12+5= 65
    ##added by liyuan: if all green nodes die, this epsoid is over.
    ##added by liyuan: if all green nodes die, this epsoid is over.
    # 如果所有的绿色节点都死了，这个插曲就结束了。
    def done(self, agent, world):  # 终止条件判定
        allDie = False  # agent未死
        if agent.death == True:
            allDie = True
        #agents = self.good_agents(world)  # 蓝色-友方
        #for agent in agents:
        #    if agent.death == True:
        #        allDie = True  # 若有绿色存活
        #        break
        # 只要有1个绿色存活 allDie = False
        return allDie
