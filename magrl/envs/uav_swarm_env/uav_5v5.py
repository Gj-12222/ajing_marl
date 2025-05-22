import numpy as np

from .baseEnv import BaseWorld, Agent, Landmark
from magrl.envs.mpe_env.multiagent.scenario import BaseScenario
import math
import copy

from magrl.utils.multi_discrete import MultiDiscrete
from magrl.config.magrl_config import TrainConfig

from magrl.config.env_config.uav_swarm_config.env_config import EnvConfig
ECG = EnvConfig()


class Scenario(BaseScenario):
    def __init__(self):
        # get config
        self.cfg = TrainConfig()

    def make_world(self):
        print("**********set scenario*************")

        world = BaseWorld()

        world.dim_c = 1  # communication
        world.dim_p = 2  # location XY
        world.dim_f = 1  #
        num_red_agents = self.cfg.num_adversaries
        num_blue_agents = self.cfg.num_agents

        num_agents = num_red_agents + num_blue_agents
        num_landmarks = 0

        self.num_blue = copy.deepcopy(num_blue_agents)
        self.num_red = copy.deepcopy(num_red_agents)
        # add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'uav %d' % i
            agent.collide = True
            agent.silent = False
            agent.u_noise = True
            agent.adversary = True if i < num_red_agents else False
            agent.size = 0.04 if agent.adversary else 0.04

            agent.accel = 3.0 if agent.adversary else 3.0
            agent.max_speed = 5.0 if agent.adversary else 5.0
            agent.max_roll = 23.0
            agent.max_course = 180.0

            agent.chi = np.random.random([1, 2]) * 0.5
            if agent.adversary:
                agent.lock_num = [0 for _ in range(num_blue_agents)]
            else:
                agent.lock_num = [0 for _ in range(num_red_agents)]
        # add landmarks
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        self.world = world
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.45, 1]) if not agent.adversary else np.array([1, 0.45, 0.45])
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

            if agent.adversary:
                agent.lock_num = [0 for _ in range(self.num_blue)]
            else:
                agent.lock_num = [0 for _ in range(self.num_red)]

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):

                if self.attack_uav(a, agent) and not a.death:
                    collisions += 1

            return collisions
        else:
            return 0

    # compute the number of locking number of the agent
    def entity_lock_num(self, agent, world):
        if agent.adversary:
            opponent = self.good_agents(world)
        else:
            opponent = self.adversaries(world)

        for i, opp in enumerate(opponent):
            if self.attack_uav(opp, agent):
                agent.lock_num[i] += 1
            else:
                agent.lock_num[i] += 0

    # compute who attacked from agent in opponent team agents.
    def attack_compute_num(self, agent, world):
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
        lock = []
        if agent.adversary:
            opponent = self.good_agents(world)
        else:
            opponent = self.adversaries(world)
        for i, opp in enumerate(opponent):
            if self.attack_uav(opp, agent):
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

        agent1_chi = [agent1.state.p_vel[0], agent1.state.p_vel[1]]

        if abs(agent1.state.p_vel[0]) < 1e-5 and abs(agent1.state.p_vel[1]) < 1e-5:
            agent1_chi[0] = 0.1
            agent1_chi[1] = 0
        agent2_chi = [agent2.state.p_vel[0], agent2.state.p_vel[1]]

        if abs(agent2.state.p_vel[0]) < 1e-5 and abs(agent2.state.p_vel[1]) < 1e-5:
            agent2_chi[0] = 0.1
            agent2_chi[1] = 0

        agent1_chi_value = np.sqrt(np.sum(np.square(agent1_chi)))
        agent1_cross = (delta_pos[0] * agent1_chi[0] + delta_pos[1] * agent1_chi[1]) / (distance * agent1_chi_value)

        if agent1_cross < -1:
            agent1_cross = -1
        if agent1_cross > 1:
            agent1_cross = 1

        agent1_angle = math.acos(agent1_cross)
        agent2_chi_value = np.sqrt(np.sum(np.square(agent2_chi)))
        agent2_cross = (-delta_pos[0] * agent2_chi[0] - delta_pos[1] * agent2_chi[1]) / (distance * agent2_chi_value)
        if agent2_cross < -1:
            agent2_cross = -1
        if agent2_cross > 1:
            agent2_cross = 1
        agent2_angle = math.acos(agent2_cross)

        revised_defense = 180 - ECG.defense_angle / 2

        if distance < ECG.fire_range and agent2_angle * 180 / math.pi > revised_defense and \
                agent1_angle * 180 / math.pi < ECG.attack_angle / 2:
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
        if distance < ECG.jam_range:
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
        rew = 0  # 即时奖励
        if agent.death:
            return 0

        adversaries = self.adversaries(world)
        blue_agents = self.good_agents(world)

        # attack reward
        rew_attack = 0
        attack_red_num = []
        if agent.collide and not agent.death:
            for b_agents in blue_agents:
                if not b_agents.death:
                    if b_agents is agent:
                        agent_attack_red = self.attack_compute_num(agent, world)
                        for i, age in enumerate(adversaries):
                            if agent_attack_red[i] != 0:
                                rew_attack += 1
                    else:
                        attack_red_num += self.attack_compute_num(b_agents, world)
                else:
                    attack_red_num += [0 for _ in range(len(agent.lock_num))]

            rew += 20 * rew_attack

        # jam reward
        if agent.action.f > 0:
            agent_jam = []
            if agent.collide and not agent.death:
                for i, a in enumerate(adversaries):
                    if self.jam_uav(agent, a) and agent.state.f[0] > 0:
                        agent_jam.append(1)
                    else:
                        agent_jam.append(0)

                rew += 0.5 * sum(agent_jam)

        agent_lock = self.lock_compute_num(agent, world)
        if agent.collide and not agent.death:
            if sum(agent_lock) >= 1:
                for i, a in enumerate(adversaries):
                    if a.death:
                        agent_lock[i] = 0

            if sum(agent_lock) >= 1:
                rew -= 8 * sum(agent_lock)
                return rew

        # jammed reward
        agent_jammed = []
        if agent.collide and not agent.death:
            for i, a in enumerate(adversaries):
                if self.jam_uav(a, agent) and a.action.f > 0 and a.state.f > 0:
                    agent_jammed.append(1)
                else:
                    agent_jammed.append(0)

            rew -= 0.2 * sum(agent_jammed)

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
        rew = 0
        if agent.death:
            return rew
        # Adversaries are rewarded for collisions with agents
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # ② distance reward
        if shape:
            dis = []
            for a in agents:
                if not a.death:
                    dis.append(np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))))
            if len(dis) > 0:
                rew -= 0.1 * min(dis)

        # attack reward
        red_attack = []
        agent_attack = []
        rew_attack = 0
        self.entity_lock_num(agent, world)

        if agent.collide and not agent.death:
            for _, adv in enumerate(adversaries):

                if not adv.death:
                    if adv is agent:
                        agent_attack = self.attack_compute_num(adv, world)

                    else:
                        red_attack += self.attack_compute_num(adv, world)
                else:
                    red_attack += [0 for _ in range(len(agent.lock_num))]
            for i, red in enumerate(agents):

                if agent_attack[i] != 0:  # 1人
                    rew_attack += 1

            rew += 20 * rew_attack

        # attacked reward
        red_agent_lock = self.lock_compute_num(agent, world)  #
        if agent.collide:

            if sum(red_agent_lock) >= 1:
                for i, ags in enumerate(agents):
                    if not ags.death:
                        red_agent_lock[i] = 0

                if sum(red_agent_lock) >= 1:
                    agent.death = True
                    rew -= 5 * sum(red_agent_lock)
                    return rew

        # jam reward
        if agent.action.f > 0:
            agent_jam = []
            if agent.collide and not agent.death:
                for i, ag in enumerate(agents):
                    if self.jam_uav(agent, ag) and agent.state.f[0] > 0:

                        agent_jam.append(1)
                    else:
                        agent_jam.append(0)

                rew += sum(agent_jam) * 0.5

        # jammed reward
        agent_jammed = []
        if agent.collide and not agent.death:
            for i, ag in enumerate(agents):
                if self.jam_uav(ag, agent) and ag.action.f > 0 and ag.state.f[0] > 0:

                    agent_jammed.append(1)
                else:
                    agent_jammed.append(0)
            rew -= sum(agent_jammed) * 0.2
        # bound reward
        for adv in adversaries:
            if not adv.death:
                exceed = False
                for p in range(world.dim_p):
                    x = abs(adv.state.p_pos[p])
                    if x > 1.5:
                        exceed = True
                        break
                if adv is agent and exceed:
                    rew -= 100
                    break

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
        if abs(agent.state.p_vel[0]) < 1e-5 and abs(agent.state.p_vel[1]) < 1e-5:
            my_chi[0] = 0
        else:
            my_chi[0] = math.atan2(agent.state.p_vel[1], agent.state.p_vel[0])
        our_chi.append(my_chi)

        temp_agents = []
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
                if abs(other.state.p_vel[0]) < 1e-5 and abs(other.state.p_vel[1]) < 1e-5:
                    tmp_chi[0] = 0
                else:
                    tmp_chi[0] = math.atan2(other.state.p_vel[1], other.state.p_vel[0])
                other_chi.append(tmp_chi)  #
                other_roll.append(other.state.p_roll)
                other_jam.append(other.state.f)

        action_number = [np.zeros(3)]  # 3 acc, roll_a, jam

        # pv = len(agent.state.p_vel)  # 2      2  自身位置
        # pp = len(agent.state.p_pos)  # 2      2  自身速度
        # proll=len(agent.state.p_roll)# 1      1  自身滚转角
        # our_jam = len(agent.state.f) # 1      1  自身干扰次数
        # ep = len(entity_pos)         # 2m     0  障碍物相对位置
        # op = len(other_pos)          # 2(n-1) 22 其他agent相对距离
        # ov = len(other_vel)          # 2(n-1) 22 其他agent的速度
        # oc = len(our_chi)            # 1      1  自身航向角度
        # ohc = len(other_chi)         # n-1    11 其他agent航向角度
        #
        # ohroll = len(other_roll)     # n-1    11 其他agent滚转角度
        # ohjam = len(other_jam)       # n-1       其他agent干扰次数
        # an = len(action_number)      # 5      5  动作数量=5

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

    # added by GuoJing: if all green nodes die, this episode is over.
    def done(self, agent):
        allDie = False
        if agent.death:
            allDie = True
        return allDie

    def terminal(self, agent, world):
        agent_index = self.world.agents.index(agent)
        allDies = []
        if agent_index < self.num_red:
            for _agent in self.adversaries(world):
                allDie = False
                if agent.death:
                    allDie = True
                allDies.append(allDie)
        else:
            for _agent in self.good_agents(world):
                allDie = False
                if agent.death:
                    allDie = True
                allDies.append(allDie)

        return all(allDies)


    def render(self, rendering, viewers, render_geoms, render_geoms_xform):
        for entity in self.world.entities:
            xform = rendering.Transform()  # init transform xform=[平移=0，旋转=0，尺度变换=1]
            # uav
            geom = rendering.make_uav(entity.size)
            # small forward_sector
            geom_attack_sector = rendering.make_forward_sector(radius=ECG.fire_range,
                                                               angle_start=- ECG.attack_angle / 2,
                                                               angle_end=ECG.attack_angle / 2)  # 不需要等分360度，仅需等分attack_angle度
            # forward_sector
            geom_defence_sector = rendering.make_forward_sector(radius=ECG.fire_range,
                                                                angle_start=180 - ECG.defense_angle / 2,
                                                                angle_end=180 + ECG.defense_angle / 2)  # 不需要等分360度，仅需等分attack_angle度
            # jam circle
            geom_explore = rendering.make_circle(ECG.jam_range)

            geom.set_color(*entity.color, alpha=0.5)
            geom_attack_sector.set_color(*entity.color, alpha=0.3)
            geom_defence_sector.set_color(*entity.color, alpha=0.05)

            geom_explore.set_color(*entity.color, alpha=0.01)
            geom.add_attr(xform)
            geom_attack_sector.add_attr(xform)
            geom_defence_sector.add_attr(xform)
            geom_explore.add_attr(xform)
            render_geoms.append(geom)
            render_geoms.append(geom_attack_sector)
            render_geoms.append(geom_defence_sector)
            render_geoms.append(geom_explore)
            render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in viewers:
                viewer.geoms = []
                for geom in render_geoms:
                    viewer.add_geom(geom)

    def update_render(self, viewers, shared_viewer, render_geoms, render_geoms_xform, mode):
        results = []
        for i in range(len(viewers)):
            # update bounds to center around agent
            cam_range = 2
            if shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.world.agents[i].state.p_pos

            viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                if 'uav' in entity.name:
                    my_chi = 0
                    if not entity.death:
                        my_chi = entity[e].state.course_angle

                    if entity.action.f == 1 and entity.state.f[0] > 0:
                        render_geoms[(e + 1) * 4 - 1].set_color(*entity.color, alpha=0.5)
                    else:
                        render_geoms[(e + 1) * 4 - 1].set_color(*entity.color, alpha=0.1)
                    render_geoms_xform[e].set_rotation(my_chi)

                render_geoms_xform[e].set_translation(*entity.state.p_pos)

            # render to display or array
            results.append(viewers[i].render(return_rgb_array=mode == 'rgb_array'))

            return results

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
                # 1. acceleration
                agent.action.u = action[0][0]
                sensitivity = 5.0
                if agent.accel is not None:
                    sensitivity = agent.accel
                agent.action.u *= sensitivity
                # 2. obstruction
                if abs(action[0][1]) > 1:
                    action[0][1] = 1 / action[0][1]
                # 3. roll angular velocity
                agent.action.r = action[0][1] * 2.3 * math.pi / 180

                if action[0][2] >= 0:
                    agent.action.f = 1
                else:
                    agent.action.f = 0

        action = action[1:]

        # ② roll
        if len(action) > 0:
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
        if len(action) > 0:
            if agent.silent:
                if self.discrete_action_input:
                    agent.action.c = np.zeros(self.world.dim_c)
                    agent.action.c[action[0]] = 1.0
                else:
                    agent.action.c = action[0]
                action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0
