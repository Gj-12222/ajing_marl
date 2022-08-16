import numpy as np
from envs.mpe_envs.multiagent.core import World, Agent, Landmark
from envs.mpe_envs.multiagent.scenario import BaseScenario
import math
import copy
from training.Config import Config



class Scenario(BaseScenario):
    def __init__(self):
        # get config
        self.cfg = Config()

    def make_world(self):
        print("**********set scenario*************")

        world = World()

        world.dim_c = 1  # communication
        world.dim_p = 2  # location XY
        world.dim_f = 1 #
        num_red_agents = 4
        num_blue_agents = 8

        num_agents = num_red_agents + num_blue_agents
        num_landmarks = 0

        self.num_blue = copy.deepcopy(num_blue_agents)
        self.num_red = copy.deepcopy(num_red_agents)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = False
            agent.u_noise = True
            agent.adversary = True if i < num_red_agents else False
            agent.size = 0.04 if agent.adversary else 0.04

            agent.accel = 3.0 if agent.adversary else 3.0
            agent.max_speed = 5.0 if agent.adversary else 5.0
            agent.max_roll = 23.0
            agent.max_course = 180.0

            agent.chi = np.random.random([1,2])*0.5
            if agent.adversary:
                agent.lock_num=[0 for j in range(num_blue_agents)]
            else:
                agent.lock_num=[0 for j in range(num_red_agents)]
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1.5, +1.5, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

            agent.state.course_angle = np.zeros(world.dim_c)
            agent.death = False
            agent.state.p_roll = np.zeros(world.dim_c)
            agent.state.f = np.array([15])

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

        if agent.adversary:
                agent.lock_num=[0 for j in range(self.num_blue)]
        else:
                agent.lock_num=[0 for j in range(self.num_red)]



    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):

                if self.attack_uav(a, agent) and a.death == False:
                    collisions += 1

            return collisions
        else:
            return 0


    # compute the number of locking number of the agent
    def entity_lock_num(self, agent, world):
        opponent = []
        if agent.adversary:
            opponent = self.good_agents(world)
        else:
            opponent = self.adversaries(world)

        for i, opp in enumerate(opponent):
            if self.attack_uav(opp,agent):
                agent.lock_num[i] += 1
            else:
                agent.lock_num[i] += 0


    # compute who attacked from agent in opponent team agents.
    def attack_compute_num(self, agent, world):
        opponent = []

        if agent.adversary:
            opponent = self.good_agents(world)
        else:
            opponent = self.adversaries(world)
        attack_num = []
        for i, opp in enumerate(opponent):
            if self.attack_uav(agent, opp):
                attack_num.append(1)

            else:
                attack_num.append(0)

        return attack_num

    # compute times for the agent is attacked from opponent team agents.
    def lock_compute_num(self, agent, world):
        opponent = []
        lock = []
        if agent.adversary:
            opponent = self.good_agents(world)
        else:
            opponent = self.adversaries(world)
        for i, opp in enumerate(opponent):
            if self.attack_uav(opp,agent):
                lock.append(1)
            else:
                lock.append(0)
        return lock

    # True if agent1 win, False for others

    def attack_uav(self, agent1, agent2):
        if agent1.death or agent2.death:
            return False

        delta_pos = agent2.state.p_pos - agent1.state.p_pos
        distance = np.sqrt(np.sum(np.square(delta_pos)))
        if distance <= 1e-5:
            return False
        
        agent1_chi = [agent1.state.p_vel[0],agent1.state.p_vel[1]]

        if abs(agent1.state.p_vel[0]) < 1e-5 and abs(agent1.state.p_vel[1])<1e-5:
            agent1_chi[0] = 0.1
            agent1_chi[1] = 0
        agent2_chi = [agent2.state.p_vel[0],agent2.state.p_vel[1]]

        if abs(agent2.state.p_vel[0]) < 1e-5 and abs(agent2.state.p_vel[1])<1e-5:
            agent2_chi[0] = 0.1
            agent2_chi[1] = 0

        agent1_chi_value = np.sqrt(np.sum(np.square(agent1_chi)))
        agent1_cross = (delta_pos[0]*agent1_chi[0]+delta_pos[1]*agent1_chi[1])/(distance*agent1_chi_value)

        if agent1_cross < -1:
           agent1_cross  = -1
        if agent1_cross > 1:
           agent1_cross = 1

        agent1_angle = math.acos(agent1_cross)
        agent2_chi_value = np.sqrt(np.sum(np.square(agent2_chi)))
        agent2_cross = (-delta_pos[0]*agent2_chi[0]-delta_pos[1]*agent2_chi[1])/(distance*agent2_chi_value)
        if agent2_cross < -1:
           agent2_cross  = -1
        if agent2_cross > 1:
           agent2_cross = 1
        agent2_angle = math.acos(agent2_cross)

        revised_defense = 180-self.cfg.defense_angle/2

        if distance < self.cfg.fire_range and agent2_angle*180/math.pi>revised_defense and agent1_angle*180/math.pi<self.cfg.attack_angle/2:
            return True

        return False
    
    # True if agent1 win, False for others

    def jam_uav(self, agent1, agent2):
        if agent1.death or agent2.death:
            return False

        # judged by angle
        delta_pos = agent2.state.p_pos - agent1.state.p_pos
        distance = np.sqrt(np.sum(np.square(delta_pos)))
        if distance <= 1e-5:
            return False
        if distance < self.cfg.jam_range:
            return True

        return False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.red_reward(agent, world) if agent.adversary else self.blue_reward(agent, world)
        return main_reward


    def blue_reward(self, agent, world):
        rew = 0  # 奖励存储

        if agent.death == True:
            return 0

        adversaries = self.adversaries(world)
        blue_agents = self.good_agents(world)

        # attack reward
        rew_attack = 0
        attack_red_num = []
        if agent.collide and agent.death == False:
            for i, bagents in enumerate(blue_agents):
                if bagents.death== False:
                    if bagents is agent:
                        agent_attack_red = self.attack_compute_num(agent,world)
                    # continue
                    else:
                        attack_red_num += self.attack_compute_num(bagents,world)
                else:
                    attack_red_num += [0 for _ in range(len(agent.lock_num))]
            for i, age in enumerate(adversaries):

                if agent_attack_red[i] !=0:
                     rew_attack += 1

            rew += 20 * rew_attack

        # jam reward
        if agent.action.f > 0:
            agent_jam = []
            if agent.collide and agent.death == False:
                for i, a in enumerate(adversaries):
                    if self.jam_uav(agent,a) and agent.state.f[0] > 0:

                        agent_jam.append(1)
                    else:
                        agent_jam.append(0)

                rew += sum(agent_jam)*0.5

        agent_lock = self.lock_compute_num(agent, world)
        if agent.collide and agent.death == False:

            if sum(agent_lock) >= 1:
                for i, a in enumerate(adversaries):

                    if a.death == True :
                        agent_lock[i] = 0

            if sum(agent_lock) >= 1:
                rew -= sum(agent_lock) * 8

                agent.death = True
                return rew
        #  jamed reward
        agent_dejam = []
        if agent.collide and agent.death == False:
            for i, a in enumerate(adversaries):
                if self.jam_uav(a, agent) and a.action.f > 0 and  a.state.f[0] > 0:

                    agent_dejam.append(1)
                else:
                    agent_dejam.append(0)
            rew -= sum(agent_dejam)*0.2

        # bound reward

        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.5:
                return 10
            return 100

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)


        return rew


    def red_reward(self, agent, world):

        if agent.death == True:
            return 0
        # Adversaries are rewarded for collisions with agents
        rew = 0

        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # ② distance reward
        if shape:
            dis = []
            for a in agents:
                if a.death == False:
                    dis.append(np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))))
            if len(dis) > 0:
                rew -= 0.1 * min(dis)
        
        # attack reward
        red_attack = []
        agent_attack = []
        rew_attack = 0
        self.entity_lock_num(agent, world)

        if agent.collide and agent.death == False:
            for _,adv in enumerate(adversaries):

                if adv.death ==False:
                    if adv is agent:
                        agent_attack = self.attack_compute_num(adv,world)

                    else:
                        red_attack += self.attack_compute_num(adv,world)
                else:
                    red_attack += [0 for _ in range(len(agent.lock_num))]
            for i, red in enumerate(agents):

                if agent_attack[i] != 0:  # 1人
                    rew_attack += 1

            rew += 20 * rew_attack

        # attacked reward
        red_agent_lock = self.lock_compute_num(agent,world)  #
        if agent.collide:

            if sum(red_agent_lock) >= 1:
                for i,ags in enumerate(agents):
                    if ags.death == True:
                        red_agent_lock[i] = 0

                if sum(red_agent_lock) >= 1:
                    agent.death = True
                    rew -=  5 * sum(red_agent_lock)
                    return rew


        # jam reward
        if agent.action.f > 0:
            agent_jam = []
            if agent.collide and agent.death == False:
                for i, ag in enumerate(agents):
                    if self.jam_uav(agent, ag) and agent.state.f[0] > 0:

                        agent_jam.append(1)
                    else:
                        agent_jam.append(0)

                rew += sum(agent_jam) * 0.5

        # jamed reward
        agent_dejam = []
        if agent.collide and agent.death == False:
            for i, ag in enumerate(agents):
                if self.jam_uav(ag, agent) and ag.action.f > 0 and ag.state.f[0] > 0:

                    agent_dejam.append(1)
                else:
                    agent_dejam.append(0)
            rew -= sum(agent_dejam) * 0.2
        # bound reward
        for adv in adversaries:
            if adv.death == False:
                exceed = False
                for p in range(world.dim_p):
                    x = abs(adv.state.p_pos[p])
                    if (x > 1.5):
                        exceed = True
                        break
                if adv is agent and exceed == True:
                    rew -= 100

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

        other_jam = []

        my_chi = np.zeros(1)
        if abs(agent.state.p_vel[0])<1e-5 and abs(agent.state.p_vel[1])<1e-5:
            my_chi[0] = 0
        else:
            my_chi[0] = math.atan2(agent.state.p_vel[1],agent.state.p_vel[0])
        our_chi.append(my_chi)

        temp_agents=[]
        for agent_i in world.agents:
            if agent_i.adversary == agent.adversary:
                temp_agents.append(agent_i)
        for agent_i in world.agents:
            if agent_i.adversary != agent.adversary:
                temp_agents.append(agent_i)

        for other in temp_agents:
            if other is agent: continue            

            if other.death:
                comm.append(np.zeros(world.dim_c))
                other_pos.append(np.zeros(world.dim_p))
                other_vel.append(np.zeros(world.dim_p))
                tmp_chi = np.zeros(1)
                other_chi.append(tmp_chi)
                other_roll.append(np.zeros(1))
                other_jam.append(np.zeros(1))
            else:
                comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)

                other_vel.append(other.state.p_vel)

                tmp_chi = np.zeros(1)  #
                if abs(other.state.p_vel[0])<1e-5 and abs(other.state.p_vel[1])<1e-5:
                    tmp_chi[0] = 0
                else:
                    tmp_chi[0] = math.atan2(other.state.p_vel[1],other.state.p_vel[0])
                other_chi.append(tmp_chi)  #
                other_roll.append(other.state.p_roll)
                other_jam.append(other.state.f)

        action_number=[np.zeros(3)]  # 3 acc, roll_a, jam

        all_shape = np.concatenate([agent.state.p_vel] +
                                   [agent.state.p_pos] +
                                   [agent.state.f] +
                                   entity_pos +
                                   other_pos +
                                   other_vel +
                                   our_chi +
                                   other_chi +
                                   action_number)

        return all_shape

    def done(self, agent, world):
        allDie = False
        if agent.death == True:
            allDie = True
        return allDie
