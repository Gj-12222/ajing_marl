r'''

CityflowEnv

'''
import math
import os
import cityflow
import json
import gym
import numpy as np

from .agent import TSCAgent, Road


def list_with_unique_element(original_list):
    new_list = []
    for elem in original_list:
        if elem not in new_list:
            new_list.append(elem)
    return new_list


class CityFlowWorld():
    def __init__(self, config: dict):
        self.cfg = config
        self.seed = config['seed']
        self.env_cfg = config['env_config']
        self.agent_cfg = config['agent_config']
        self.algo_cfg = config['algo_config']
        self.device = config['device']

        self.save_play = self.env_cfg['save_play']
        self.road_file = self.env_cfg['roadnet']
        self.flow_file = self.env_cfg['flow']
        self.roadID_file = self.env_cfg['roadID']
        self.virtualAgent_file = self.env_cfg['virtualAgent_file']
        self.roadIDs = None

        self.eng = cityflow.Engine(self.env_cfg['config'], thread_num=self.env_cfg['thread_num'])
        self.eng.set_save_replay(self.save_play)
        self.max_step = self.algo_cfg['max_episode_len']  # 3600s
        self.setup()

    def setup(self):
        self._setup_world()
        self._setup_agent()

        self.time_step = 0
        self.reset()

    def step(self):
        # 最小决策间隔
        for j in range(self.agent_cfg['action_interval']):  # 20s
            # set tls
            for agent in self.agents:
                agent.step(self.interval, j)
            # world step
            for i in range(self.interval):  # s
                self.eng.next_step()
                self.time_step += 1

        # self._get_global_state()
        self.current_action_n = [agent.current_phase for agent in self.agents]
        self.last_action_n = [agent.last_phase for agent in self.agents]

    def eval_step(self):
        # 最小决策间隔
        # set tls
        for agent in self.agents:
            agent.eval_step(self.interval)
        self.eng.next_step()  # world step
        self.time_step += 1

    def reset(self):
        self._vehicle_waiting_time = {}
        self._vehicle_trajectory = {}
        self._vehicle_trajectory_last_update_time = -1
        self._cache_queue_length = [0, 0, 0]
        self._cache_average_delay = [0, 0, 0]
        self._cache_throughput = [0, 0, 0, 0]
        self.current_action_n = None
        self.last_action_n = None

        self.eng.reset(seed=self.seed)

        for i, agent in enumerate(self.agents):
            agent.reset()
            agent.action_callback = None
            agent.action = None
        for virtualAgentID in self.virtualAgent:
            self.eng.set_tl_phase(virtualAgentID, 0)

        self.time_step = 0
        self.eng.set_save_replay(self.save_play)
        # self._get_global_state()

    def _update_average_queue_length(self):
        for agent in self.agents:
            lane2waiting_vehicle = self._lane_2_num_waiting_vehicle(agent)
            for waiting_vehicle in lane2waiting_vehicle:
                self._cache_queue_length[0] += 1
                self._cache_queue_length[1] += waiting_vehicle
                self._cache_queue_length[2] += waiting_vehicle * waiting_vehicle

    # def _update_average_throughput(self):
    #     current_throughput = self.eng.get_finished_vehicle_cnt()
    #     throughput_this_minute = current_throughput - self._cache_throughput[3]
    #     self._cache_throughput[0] += 1
    #     self._cache_throughput[1] += throughput_this_minute
    #     self._cache_throughput[2] += throughput_this_minute * throughput_this_minute
    #     self._cache_throughput[3] = current_throughput

    def _update_average_delay(self):
        for agent in self.agents:
            inlane2delay = self._inlane_2_delay(agent)
            for delay in inlane2delay:
                self._cache_average_delay[0] += 1
                self._cache_average_delay[1] += delay
                self._cache_average_delay[2] += delay * delay

    def _seed(self, seed=None):
        self.eng.set_random_seed(seed)

    @property
    def world_time(self):
        return self.eng.get_current_time()

    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    def reward(self, agent: TSCAgent):
        reward = np.array(0.)
        for reward_feature, reward_weight in zip(self.algo_cfg['reward_list'], self.algo_cfg['reward_weight']):
            feature = self._info_functions[reward_feature](agent)
            reward = reward + feature * reward_weight

        # 仅考虑当前agent的reward
        # lane_waiting_vehicles_dict = self.eng.get_lane_waiting_vehicle_count()
        # vehicle_num = np.zeros(1, dtype=np.float32)
        # for edgeID, vehNum in lane_waiting_vehicles_dict.items():
        #     if edgeID in self.roadDict[agent.intersection_id]:
        #         vehicle_num += vehNum
        # TODO: 团队协同的reward(未考虑)

        return reward

    def observation(self, agent: TSCAgent):
        obs = []
        for observation_feature in self.algo_cfg['obs_list']:
            feature = self._info_functions[observation_feature](agent)
            obs.append(feature)
        if self.agent_cfg['algo_name'] == 'TinyLight':
            obs = [_obs.reshape(1, -1) for _obs in obs]
            return obs
        else:
            obs = np.concatenate(obs, axis=-1, dtype=np.float32)
            return obs

    def done(self, agent):
        # TODO: 如果该交叉口出现了锁死, 该如何判断状态
        if self.time_step >= self.max_step:
            return True
        else:
            return False

    def terminal(self):
        # if self.time_step >= self.max_step:
        #     return True
        # else:
        #     return False

        return False

    def info(self, agent):
        info = {}
        for metric_feature in self.algo_cfg['metric_list']:
            info[metric_feature] = self._info_functions[metric_feature]()
        return info

    def _setup_world(self):
        with open(self.env_cfg['config'], encoding='utf-8') as fp:
            self.engConfigDict = json.load(fp)
        with open(self.road_file, encoding='utf-8') as fp:
            self.road_dict = json.load(fp)
        with open(self.flow_file, encoding='utf-8') as fp:
            self.flow_dict = json.load(fp)
        if not os.path.exists(self.roadID_file) is None:
            with open(self.roadID_file, encoding='utf-8') as fp:
                self.roadIDs = json.load(fp)
        if not os.path.exists(self.virtualAgent_file) is None:
            with open(self.virtualAgent_file, encoding='utf-8') as fp:
                self.virtualAgent = json.load(fp)

        for virtualAgentID in self.virtualAgent:
            self.eng.set_tl_phase(virtualAgentID, 0)

        # self.inters, self.road_laneID = {}, {}
        # self.eng_state = {}

        if self.engConfigDict['interval'] < 1:
            self.interval = int(1 / self.engConfigDict['interval'])
        else:
            self.interval = self.engConfigDict['interval']

        # info function dict
        self._info_functions = {
            'agent_2_index': self._agent_2_index,
            'lane_2_num_vehicle': self._lane_2_num_vehicle,
            'lane_2_num_waiting_vehicle': self._lane_2_num_waiting_vehicle,
            'lane_2_sum_waiting_time': self._lane_2_sum_waiting_time,
            'lane_2_delay': self._lane_2_delay,
            'lane_2_num_vehicle_seg_by_k': self._lane_2_num_vehicle_seg_by_k,
            ######################
            'lane_2_max_num_waiting_vehicle': self._lane_2_max_num_waiting_vehicle,
            'lane_2_sum_num_waiting_vehicle': self._lane_2_sum_num_waiting_vehicle,
            #######################
            'inlane_2_num_vehicle': self._inlane_2_num_vehicle,
            'inlane_2_num_waiting_vehicle': self._inlane_2_num_waiting_vehicle,
            'inlane_2_sum_waiting_time': self._inlane_2_sum_waiting_time,
            'inlane_2_delay': self._inlane_2_delay,
            'inlane_2_pressure': self._inlanelink_2_pressure,
            'inlane_2_num_vehicle_seg_by_k': self._inlane_2_num_vehicle_seg_by_k,

            'outlane_2_num_vehicle': self._outlane_2_num_vehicle,
            'outlane_2_num_waiting_vehicle': self._outlane_2_num_waiting_vehicle,
            'outlane_2_sum_waiting_time': self._outlane_2_sum_waiting_time,
            'outlane_2_delay': self._outlane_2_delay,
            'outlane_2_num_vehicle_seg_by_k': self._outlane_2_num_vehicle_seg_by_k,

            'phase_2_num_vehicle': self._phase_2_num_vehicle,
            'phase_2_num_waiting_vehicle': self._phase_2_num_waiting_vehicle,
            'phase_2_sum_waiting_time': self._phase_2_sum_waiting_time,
            'phase_2_delay': self._phase_2_delay,
            'phase_2_pressure': self._phase_2_pressure,

            #########################
            'phase_2_max_num_vehicle': self._phase_2_max_num_vehicle,
            'phase_2_max_num_waiting_vehicle': self._phase_2_max_num_waiting_vehicle,
            ########################
            'inroad_2_num_vehicle': self._inroad_2_num_vehicle,
            'inroad_2_num_waiting_vehicle': self._inroad_2_num_waiting_vehicle,
            'inroad_2_sum_waiting_time': self._inroad_2_sum_waiting_time,
            'inroad_2_delay': self._inroad_2_delay,
            ##########################
            'inroad_2_sum_num_waiting_vehicle': self._inroad_2_sum_num_waiting_vehicle,
            'inroad_2_max_num_waiting_vehicle': self._inroad_2_max_num_waiting_vehicle,
            'inroad_2_other_sum_num_vehicle': self._inroad_2_other_sum_num_vehicle,
            ##########################
            'inter_2_num_vehicle': self._inter_2_num_vehicle,
            'inter_2_num_waiting_vehicle': self._inter_2_num_waiting_vehicle,
            'inter_2_sum_waiting_time': self._inter_2_sum_waiting_time,
            'inter_2_delay': self._inter_2_delay,
            'inter_2_pressure': self._inter_2_pressure,
            'inter_2_vehicle_position_image': self._inter_2_vehicle_position_image,
            'inter_2_current_phase': self._inter_2_current_phase,
            'inter_2_current_phase_duration': self._inter_2_current_phase_duration,
            'inter_2_next_phase': self._inter_2_next_phase,
            'inter_2_phase_has_changed': self._inter_2_phase_has_changed,
            'inter_2_num_passed_vehicle_since_last_action': self._inter_2_num_passed_vehicle_since_last_action,
            'inter_2_sum_travel_time_since_last_action': self._inter_2_sum_travel_time_since_last_action,

            'lanelink_2_pressure': self._lanelink_2_pressure,
            'lanelink_2_num_vehicle': self._lanelink_2_num_vehicle,

            # other agent current phase
            'other_inter_2_current_phase': self._other_inter_2_current_phase,
            'other_inter_2_current_phase_duration': self._other_inter_2_current_phase_duration,
            # evaluation metric
            'world_2_average_travel_time': self._world_2_average_travel_time,
            'world_2_average_queue_length': self._world_2_average_queue_length,
            'world_2_average_throughput': self._world_2_average_throughput,
            'world_2_average_delay': self._world_2_average_delay,
        }
        self._vehicle_waiting_time = {}
        self._vehicle_trajectory = {}
        self._vehicle_trajectory_last_update_time = -1
        self._cache_queue_length = [0, 0, 0]  # number of sample, sum of sample, sum of sample^2
        self._cache_average_delay = [0, 0, 0]  # number of sample, sum of sample, sum of sample^2
        self._cache_throughput = [0, 0, 0, 0]  # number of sample, sum of sample, sum of sample^2, last throughput

        # parsing roads
        self.id2road = {}
        for road_dict in self.road_dict['roads']:
            road_id = road_dict['id']
            self.id2road[road_id] = Road(road_id, road_dict)

        # relevant parameters
        self._seed(self.seed)

    def _setup_agent(self):
        # parsing intersections
        self.id2agent = {}
        self.agents = []
        self.tlsRoadIDs = filter(
            lambda inter_: (not inter_['virtual']) and (len(inter_['trafficLight']['lightphases']) > 1),
            self.road_dict['intersections'])

        for tls_idx, tls_dict in enumerate(self.tlsRoadIDs):
            if tls_dict['id'] in self.virtualAgent: continue

            agent_id = tls_dict['id']
            agent = TSCAgent(tls_idx, agent_id, tls_dict, self)
            self.id2agent[agent_id] = agent
            self.agents.append(agent)
        # find out the neighbors of each intersection
        for agent_idx, agent_i in enumerate(self.agents):
            for agent_jdx, agent_j in enumerate(self.agents):
                if agent_idx == agent_jdx: continue
                if not set(agent_i.n_out_road).isdisjoint(set(agent_j.n_in_road)):
                    agent_i.n_neighbor_idx.append(agent_j.inter_idx)

        self.n = len(self.agents)
        # match TinyLight
        for agent in self.agents:
            obs = self.observation(agent)
            if self.agent_cfg['algo_name'] == 'TinyLight':
                obs_shape = []
                for inter_obs in obs:
                    obs_shape.append(inter_obs.shape[1:])  # dim 0 is batch
                agent.obs_dim = obs_shape
            else:
                agent.obs_dim = obs.shape[0]

        # 配合baseline算法
        self.n_action_space = [gym.spaces.Discrete(len(agent.n_phase)) for agent in self.agents]

    def _get_global_state(self):
        self.global_state = {
            'total_run_vehicle_num': self.eng.get_vehicle_count(),  # 总运行车辆数
            'total_vehicle_id': self.eng.get_vehicles(include_waiting=True),  # 所有车辆id
            'total_run_vehicle_id': self.eng.get_vehicles(include_waiting=False),  # 仅运行车辆id
            'total_vehicle_speed': self.eng.get_vehicle_speed(),  # 所有车辆的speed
            'total_vehicle_distance_lane': self.eng.get_vehicle_distance(),  # 所有车辆的在当前车道的行驶距离
            'vehicle_num_lane': self.eng.get_lane_vehicle_count(),  # 每条车道的运行车辆数量
            'vehicle_id_lane': self.eng.get_lane_vehicles(),  # 每条车道的车辆id
            'waiter_vehicle_num_lane': self.eng.get_lane_waiting_vehicle_count(),  # 每条车道的等待车辆数量
            'sim_time': self.eng.get_current_time(),  # 仿真运行时间
            'average_travel_time': self.eng.get_average_travel_time(),  # 车辆平均旅行时间
            'tsc_last_actions': np.array([self._get_eng_tls_(agent) for agent in self.agents])  # 所有路口的last action
            # vehicle_info = self.eng.get_vehicle_info(veh_id)  # 给定 veh_id 车辆信息 {speed distance drivable road intersection route}
            # leader_vehicle_id = self.eng.get_leader(veh_id)  # veh_id 前面的车辆 id
        }

        return self.global_state

    # create dict of controllable inters and number of light phases
    def _parse_roadnet(self):
        for i in range(len(self.road_dict['intersections'])):
            # check if intersection is controllable
            if self.road_dict['intersections'][i]['virtual'] == False:
                # for each roadLink in intersection store incoming lanes, outgoing lanes and direction in lists
                incomingLanes = []
                outgoingLanes = []
                directions = []
                totalRoadIDs = []
                inputRoadIDs = []
                outputRoadIDs = []
                for j in range(len(self.road_dict['intersections'][i]['roadLinks'])):
                    incomingRoads = []
                    outgoingRoads = []
                    directions.append(self.road_dict['intersections'][i]['roadLinks'][j]['direction'])
                    for k in range(len(self.road_dict['intersections'][i]['roadLinks'][j]['laneLinks'])):
                        incomingRoads.append(self.road_dict['intersections'][i]['roadLinks'][j]['startRoad'] +
                                             '_' +
                                             str(self.road_dict['intersections'][i]['roadLinks'][j]['laneLinks'][k][
                                                     'startLaneIndex']))
                        outgoingRoads.append(self.road_dict['intersections'][i]['roadLinks'][j]['endRoad'] +
                                             '_' +
                                             str(self.road_dict['intersections'][i]['roadLinks'][j]['laneLinks'][k][
                                                     'endLaneIndex']))
                    incomingLanes.append(incomingRoads)
                    outgoingLanes.append(outgoingRoads)
                    inputRoadIDs.append(self.road_dict['intersections'][i]['roadLinks'][j]['startRoad'])
                    outputRoadIDs.append(self.road_dict['intersections'][i]['roadLinks'][j]['endRoad'])
                for laneID in self.road_dict['intersections'][i]['roads']:
                    totalRoadIDs.append(laneID)
                inputRoadIDs = list_with_unique_element(inputRoadIDs)
                outputRoadIDs = list_with_unique_element(outputRoadIDs)
                # add intersection to dict where key = intersection_id
                # value = phase_num, phase_dict, incoming lane names, outgoing lane names, lane_directions车道组方向
                phase_num = [len(self.road_dict['intersections'][i]['trafficLight']['lightphases'])]
                phase_dict = self.road_dict['intersections'][i]['trafficLight']['lightphases']

                self.inters[self.road_dict['intersections'][i]['id']] = {'phaseNum': phase_num,
                                                                         'pahseDict': phase_dict,
                                                                         'totalRoadIDs': totalRoadIDs,
                                                                         'inputRoadIDs': inputRoadIDs,
                                                                         'outputRoadIDs': outputRoadIDs,
                                                                         'inputLaneIDs': incomingLanes,
                                                                         'outputLaneIDs': outgoingLanes,
                                                                         'laneDirections': directions}

    # 当前agent的位置index
    def _agent_2_index(self, agent: TSCAgent):
        agent_indexs = np.zeros(self.n)
        agent_indexs[agent.inter_idx] = 1.0
        return agent_indexs

    # 当前agent 的 进口lane + 出口lane 的存在车辆数
    def _lane_2_num_vehicle(self, agent: TSCAgent) -> np.array:
        lane_2_num_vehicle = self.eng.get_lane_vehicle_count()
        return self._concat_by_unit(lane_2_num_vehicle, agent, unit='lane', inroad_only=False, outroad_only=False)

    # 当前agent 的 进口lane + 出口lane 的等待车辆数
    def _lane_2_num_waiting_vehicle(self, agent: TSCAgent) -> np.array:
        lane_2_num_waiting_vehicle = self.eng.get_lane_waiting_vehicle_count()
        return self._concat_by_unit(lane_2_num_waiting_vehicle, agent, unit='lane', inroad_only=False,
                                    outroad_only=False)

    # 当前agent 的 进口lane + 出口lane 中 最大等待车辆数
    def _lane_2_max_num_waiting_vehicle(self, agent: TSCAgent) -> np.array:
        lane_2_num_waiting_vehicle = self._lane_2_num_waiting_vehicle(agent)
        return np.max(lane_2_num_waiting_vehicle, keepdims=1)

    # 当前agent 的 进口lane + 出口lane 中 最大等待车辆数
    def _lane_2_sum_num_waiting_vehicle(self, agent: TSCAgent) -> np.array:
        lane_2_num_waiting_vehicle = self._lane_2_num_waiting_vehicle(agent)
        return np.sum(lane_2_num_waiting_vehicle, keepdims=1)

    # 当前agent的所有phase里的允许通过进口lane + 出口lane的每个lane都划分为K份，每份的存在车辆数
    def _lane_2_num_vehicle_seg_by_k(self, agent: TSCAgent):
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_distance = self.eng.get_vehicle_distance()
        k = self.algo_cfg['K']

        n_lane_2_num_vehicle_seg_by_k = []
        for road in agent.n_road:
            for lane_id in road.n_lane_id:
                lane_2_num_vehicle_seg_by_k = [0.0 for _ in range(k)]
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                for vehicle_id in n_vehicle_id:
                    vehicle_distance = vehicle_2_distance[vehicle_id]
                    idx = math.floor(vehicle_distance / (road.length / k))
                    idx = max(min(idx, k - 1), 0)
                    lane_2_num_vehicle_seg_by_k[idx] += 1.0
                n_lane_2_num_vehicle_seg_by_k.extend(lane_2_num_vehicle_seg_by_k)
        return np.array(n_lane_2_num_vehicle_seg_by_k)

    # 当前agent的 出口lane +  进口lane 的总等待时间
    def _lane_2_sum_waiting_time(self, agent: TSCAgent):
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_waiting_time = self._get_vehicle_waiting_time()

        n_lane_waiting_time = []
        for road in agent.n_road:
            for lane_id in road.n_lane_id:
                lane_waiting_time = 0.
                for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                    lane_waiting_time += vehicle_2_waiting_time[vehicle_id]
                n_lane_waiting_time.append(lane_waiting_time)
        n_lane_waiting_time = np.array(n_lane_waiting_time)
        return n_lane_waiting_time

    # 当前agent的 出口lane +  进口lane的 平均车辆延误率
    def _lane_2_delay(self, agent):
        # delay of each lane = 1. - lane_avg_speed / speed_limit
        # by default the speed limit is 11.11
        speed_limit = 11.11
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_speed = self.eng.get_vehicle_speed()
        n_lane_delay = []

        for road in agent.n_road:
            for lane_id in road.n_lane_id:
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                vehicle_speed_sum = 0
                for vehicle_id in n_vehicle_id:
                    vehicle_speed_sum += vehicle_2_speed[vehicle_id]
                if len(n_vehicle_id) == 0:
                    lane_avg_speed = speed_limit
                else:
                    lane_avg_speed = vehicle_speed_sum * 1.0 / len(n_vehicle_id)
                n_lane_delay.append(1. - lane_avg_speed / speed_limit)

        n_lane_delay = np.array(n_lane_delay)
        return n_lane_delay

    # 当前agent 的 出口lane 的存在车辆数
    def _outlane_2_num_vehicle(self, agent: TSCAgent):
        lane_2_num_waiting_vehicle = self.eng.get_lane_waiting_vehicle_count()
        n_outlane_2_num_vehicle = []
        for road in agent.n_out_road:
            for lane_id in road.n_lane_id:
                n_outlane_2_num_vehicle.append(lane_2_num_waiting_vehicle[lane_id])
        n_outlane_2_num_vehicle = np.array(n_outlane_2_num_vehicle)
        return n_outlane_2_num_vehicle

    # 当前agent的 出口lane的车辆总等待时间
    def _outlane_2_sum_waiting_time(self, agent: TSCAgent):
        # the sum of waiting times of vehicles on the lane since their last halt
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_waiting_time = self._get_vehicle_waiting_time()

        n_lane_waiting_time = []
        for out_road in agent.n_out_road:
            for lane_id in out_road.n_lane_id:
                lane_waiting_time = 0.
                for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                    lane_waiting_time += vehicle_2_waiting_time[vehicle_id]
                n_lane_waiting_time.append(lane_waiting_time)
        n_lane_waiting_time = np.array(n_lane_waiting_time)
        return n_lane_waiting_time

    # 当前agent的 出口lane 的 平均车辆延误率
    def _outlane_2_delay(self, agent):
        # delay of each lane = 1. - lane_avg_speed / speed_limit
        # by default the speed limit is 11.11
        speed_limit = 11.11
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_speed = self.eng.get_vehicle_speed()
        n_lane_delay = []

        for out_road in agent.n_out_road:
            for lane_id in out_road.n_lane_id:
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                vehicle_speed_sum = 0
                for vehicle_id in n_vehicle_id:
                    vehicle_speed_sum += vehicle_2_speed[vehicle_id]
                if len(n_vehicle_id) == 0:
                    lane_avg_speed = speed_limit
                else:
                    lane_avg_speed = vehicle_speed_sum * 1.0 / len(n_vehicle_id)
                n_lane_delay.append(1. - lane_avg_speed / speed_limit)

        n_lane_delay = np.array(n_lane_delay)
        return n_lane_delay

    # 当前agent 的 出口lane 的等待车辆数
    def _outlane_2_num_waiting_vehicle(self, agent):
        lane_2_num_waiting_vehicle = self.eng.get_lane_waiting_vehicle_count()
        return self._concat_by_unit(lane_2_num_waiting_vehicle, agent, unit='lane', inroad_only=False,
                                    outroad_only=True)

    # 当前agent 的 进口lane + 出口lane 中 最大等待车辆数
    def _outlane_2_max_num_waiting_vehicle(self, agent: TSCAgent) -> np.array:
        outlane_2_num_waiting_vehicle = self._outlane_2_num_waiting_vehicle(agent)
        return np.max(outlane_2_num_waiting_vehicle, keepdims=1)

    # 当前agent的 出口lane的每个lane划分为K份，每份的存在车辆数
    def _outlane_2_num_vehicle_seg_by_k(self, agent: TSCAgent):
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_distance = self.eng.get_vehicle_distance()
        k = self.algo_cfg['K']

        n_outlane_2_num_vehicle_seg_by_k = []
        for road in agent.n_out_road:
            for lane_id in road.n_lane_id:
                outlane_2_num_vehicle_seg_by_k = [0.0 for _ in range(k)]
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                for vehicle_id in n_vehicle_id:
                    vehicle_distance = vehicle_2_distance[vehicle_id]
                    idx = math.floor(vehicle_distance / (road.length / k))
                    idx = max(min(idx, k - 1), 0)
                    outlane_2_num_vehicle_seg_by_k[idx] += 1.0
                n_outlane_2_num_vehicle_seg_by_k.extend(outlane_2_num_vehicle_seg_by_k)
        return np.array(n_outlane_2_num_vehicle_seg_by_k)

    # 当前agent的 进口lane 的 存在车辆数
    def _inlane_2_num_vehicle(self, agent):
        lane_2_num_vehicle = self.eng.get_lane_vehicle_count()
        return self._concat_by_unit(lane_2_num_vehicle, agent, unit='lane', inroad_only=True, outroad_only=False)

    # 当前agent的 进口lane 的 等待车辆数
    def _inlane_2_num_waiting_vehicle(self, agent):
        lane_2_num_waiting_vehicle = self.eng.get_lane_waiting_vehicle_count()
        return self._concat_by_unit(lane_2_num_waiting_vehicle, agent, unit='lane', inroad_only=True,
                                    outroad_only=False)

    # 当前agent的 进口lane的车辆总等待时间
    def _inlane_2_sum_waiting_time(self, agent: TSCAgent):
        # the sum of waiting times of vehicles on the lane since their last halt
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_waiting_time = self._get_vehicle_waiting_time()

        n_lane_waiting_time = []
        for in_road in agent.n_in_road:
            for lane_id in in_road.n_lane_id:
                lane_waiting_time = 0.
                for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                    lane_waiting_time += vehicle_2_waiting_time[vehicle_id]
                n_lane_waiting_time.append(lane_waiting_time)
        n_lane_waiting_time = np.array(n_lane_waiting_time)
        return n_lane_waiting_time

    # 当前agent的 进口lane 的 平均车辆延误率
    def _inlane_2_delay(self, agent):
        # delay of each lane = 1. - lane_avg_speed / speed_limit
        # by default the speed limit is 11.11
        speed_limit = 11.11
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_speed = self.eng.get_vehicle_speed()
        n_lane_delay = []

        for in_road in agent.n_in_road:
            for lane_id in in_road.n_lane_id:
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                vehicle_speed_sum = 0
                for vehicle_id in n_vehicle_id:
                    vehicle_speed_sum += vehicle_2_speed[vehicle_id]
                if len(n_vehicle_id) == 0:
                    lane_avg_speed = speed_limit
                else:
                    lane_avg_speed = vehicle_speed_sum * 1.0 / len(n_vehicle_id)
                n_lane_delay.append(1. - lane_avg_speed / speed_limit)

        n_lane_delay = np.array(n_lane_delay)
        return n_lane_delay

    # 当前agent的所有phase里的允许通过进口lane的每个lane划分为K份，每份的存在车辆数
    def _inlane_2_num_vehicle_seg_by_k(self, agent: TSCAgent):
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_distance = self.eng.get_vehicle_distance()
        k = self.algo_cfg['K']

        n_inlane_2_num_vehicle_seg_by_k = []
        for road in agent.n_in_road:
            for lane_id in road.n_lane_id:
                inlane_2_num_vehicle_seg_by_k = [0.0 for _ in range(k)]
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                for vehicle_id in n_vehicle_id:
                    vehicle_distance = vehicle_2_distance[vehicle_id]
                    idx = math.floor(vehicle_distance / (road.length / k))
                    idx = max(min(idx, k - 1), 0)
                    inlane_2_num_vehicle_seg_by_k[idx] += 1.0
                n_inlane_2_num_vehicle_seg_by_k.extend(inlane_2_num_vehicle_seg_by_k)
        return np.array(n_inlane_2_num_vehicle_seg_by_k)

    # 当前agent的 进口lane的 车道压力 == 两条通过交叉口的链接两端道路的车辆差值
    def _inlanelink_2_pressure(self, agent: TSCAgent):
        lane_2_vehicle_count = self.eng.get_lane_vehicle_count()

        n_lane_pressure = []
        for in_road in agent.n_in_road:
            for lane_id in in_road.n_lane_id:
                lane_pressure = 0.0
                for road_link in agent.n_roadlink:
                    if road_link.startroad_id != in_road.road_id:
                        continue
                    for lane_link in road_link.n_lanelink_id:
                        if lane_link[0] == lane_id:
                            lane_pressure += lane_2_vehicle_count[lane_link[0]]
                            lane_pressure -= lane_2_vehicle_count[lane_link[1]]
                n_lane_pressure.append(lane_pressure)
        return np.array(n_lane_pressure)

    # 当前agent的 进口lanelink + 出口lanelink 的 车道压力
    def _lanelink_2_pressure(self, agent: TSCAgent):
        lane_2_vehicle_count = self.eng.get_lane_vehicle_count()
        n_lanelink_pressure = []
        for roadlink in agent.n_roadlink:
            for lanelink in roadlink.n_lanelink_id:
                lanelink_pressure = lane_2_vehicle_count[lanelink[0]] - lane_2_vehicle_count[lanelink[1]]
                n_lanelink_pressure.append(lanelink_pressure)
        return np.array(n_lanelink_pressure)

    # 当前agent的 进口lanelink + 出口lanelink 的 总车辆数
    def _lanelink_2_num_vehicle(self, agent: TSCAgent):
        lane_2_vehicle_count = self.eng.get_lane_vehicle_count()
        n_lanelink_num_vehicle = []
        for roadlink in agent.n_roadlink:
            for lanelink in roadlink.n_lanelink_id:
                inlane_id, outlane_id = lanelink[0], lanelink[1]
                lanelink_num_vehicle = lane_2_vehicle_count[inlane_id] + lane_2_vehicle_count[outlane_id]
                n_lanelink_num_vehicle.append(lanelink_num_vehicle)
        return np.array(n_lanelink_num_vehicle)

    #  当前agent的所有phase里的允许通过进口lane的 feature
    def _represent_feature_from_inlane_to_phase(self, inlane_feature, agent: TSCAgent):
        phase_2_passable_lane = agent.phase_2_passable_lane_idx  # 所有phase的允许通过lane的one-hot
        lane_2_applicable_phase = np.transpose(phase_2_passable_lane)  # 2维里相当于转置.T
        # 矩阵乘法 选取对应lane的 feature
        phase_feature = np.matmul(inlane_feature, lane_2_applicable_phase)
        return phase_feature

    # 当前agent的所有phase里的允许通过进口lane的 存在车辆数
    def _phase_2_num_vehicle(self, agent: TSCAgent):
        inlane_2_num_vehicle = self._inlane_2_num_vehicle(agent)
        return self._represent_feature_from_inlane_to_phase(inlane_2_num_vehicle, agent)

    # 当前agent的所有phase里的允许通过进口lane的 存在车辆数 之和
    def _phase_2_sum_num_vehicle(self, agent: TSCAgent):
        inlane_2_num_vehicle = self._phase_2_num_vehicle(agent)
        return np.sum(inlane_2_num_vehicle, keepdims=True)

    # 当前agent的所有phase里的允许通过进口lane的 等待车辆数
    def _phase_2_num_waiting_vehicle(self, agent: TSCAgent):
        inlane_2_num_waiting_vehicle = self._inlane_2_num_waiting_vehicle(agent)
        return self._represent_feature_from_inlane_to_phase(inlane_2_num_waiting_vehicle, agent)

    # 当前agent的所有phase里的允许通过进口lane的 存在车辆的最大值
    def _phase_2_max_num_vehicle(self, agent: TSCAgent):
        phase_2_num_vehicle = self._phase_2_num_vehicle(agent)
        return np.max(phase_2_num_vehicle, axis=-1, keepdims=True)

    # 当前agent的当前phase里的允许通过进口lane的 存在车辆的总数
    def _current_phase_2_sum_num_vehicle(self, agent: TSCAgent):
        # 所有phase下的 sum vehicle number
        phase_2_num_vehicle = self._phase_2_num_vehicle(agent)
        # 当前phase的 index
        phase_index = self._inter_2_current_phase(agent, index=True)
        # 获取当前phase的所有lane的 vehicle number
        phase_2_num_vehicle = phase_2_num_vehicle[phase_index]
        # get sum
        return np.sum(phase_2_num_vehicle, keepdims=True)

    # 当前agent的所有phase里的允许通过进口lane的 等待车辆数的最大值
    def _phase_2_max_num_waiting_vehicle(self, agent: TSCAgent):
        phase_2_num_waiting_vehicle = self._phase_2_num_waiting_vehicle(agent)
        return np.max(phase_2_num_waiting_vehicle, axis=-1, keepdims=True)

    # 当前agent的所有phase里的允许通过进口lane的 车辆等待时间
    def _phase_2_sum_waiting_time(self, agent: TSCAgent):
        inlane_2_sum_waiting_time = self._inlane_2_sum_waiting_time(agent)
        return self._represent_feature_from_inlane_to_phase(inlane_2_sum_waiting_time, agent)

    # 当前agent的所有phase里的允许通过lane的 车辆延误率
    def _phase_2_delay(self, agent: TSCAgent):
        inlane_2_delay = self._inlane_2_delay(agent)
        return self._represent_feature_from_inlane_to_phase(inlane_2_delay, agent)

    # 当前agent的所有phase里的允许通过进口lane的lanelink 压力
    def _phase_2_pressure(self, agent: TSCAgent):
        inlane_2_pressure = self._inlanelink_2_pressure(agent)
        return self._represent_feature_from_inlane_to_phase(inlane_2_pressure, agent)

    # 当前agent的进口edge(road)的存在车辆数
    def _inroad_2_num_vehicle(self, agent):
        lane_2_num_vehicle = self.eng.get_lane_vehicle_count()
        return self._concat_by_unit(lane_2_num_vehicle, agent, unit='road', inroad_only=True, outroad_only=False)

    # 当前agent的进口edge(road)的存在车辆数之和
    def _inroad_2_sum_num_vehicle(self, agent: TSCAgent):
        inroad_2_num_vehicle = self._inroad_2_num_vehicle(agent)
        return np.sum(inroad_2_num_vehicle, axis=-1, keepdims=True)

    # 当前agent的进口edge(road)的平均等待车辆数
    def _inroad_2_num_waiting_vehicle(self, agent):
        lane_2_num_waiting_vehicle = self.eng.get_lane_waiting_vehicle_count()
        return self._concat_by_unit(lane_2_num_waiting_vehicle, agent, unit='road', inroad_only=True,
                                    outroad_only=False)

    # 当前agent的进口edge(road)的平均等待车辆数之和
    def _inroad_2_sum_num_waiting_vehicle(self, agent):
        inroad_2_num_waiting_vehicle = self._inroad_2_num_waiting_vehicle(agent)
        return np.sum(inroad_2_num_waiting_vehicle, keepdims=True)

    # 当前agent的进口edge(road)的平均等待车辆数的最大值
    def _inroad_2_max_num_waiting_vehicle(self, agent):
        inroad_2_num_waiting_vehicle = self._inroad_2_num_waiting_vehicle(agent)
        return np.max(inroad_2_num_waiting_vehicle, keepdims=True)

    # 当前agent的所有进口edge(road)排除当前agent的当前phase的in lane的最大车辆数的车辆数之和
    def _inroad_2_other_sum_num_vehicle(self, agent: TSCAgent):
        # return self._inroad_2_sum_num_vehicle(agent) - self._phase_2_max_num_vehicle(agent)
        return self._inroad_2_sum_num_vehicle(agent) - self._current_phase_2_sum_num_vehicle(agent)

    # 当前agent的进口edge(road)的平均车辆等待时间
    def _inroad_2_sum_waiting_time(self, agent):
        # the sum of waiting time of vehicles on the road since their last halt
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_waiting_time = self._get_vehicle_waiting_time()

        n_road_waiting_time = []
        for in_road in agent.n_in_road:
            road_waiting_time = 0.
            for lane_id in in_road.n_lane_id:
                for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                    road_waiting_time += vehicle_2_waiting_time[vehicle_id]
            n_road_waiting_time.append(road_waiting_time)
        n_road_waiting_time = np.array(n_road_waiting_time)
        return n_road_waiting_time

    # 当前agent的进口edge(road)的平均车辆延误率
    def _inroad_2_delay(self, agent):
        # delay of each road = 1. - road_avg_speed / speed_limit
        # by default the speed_limit is 11.11
        speed_limit = 11.11
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_speed = self.eng.get_vehicle_speed()
        n_road_delay = []

        for in_road in agent.n_in_road:
            vehicle_speed_sum = 0.
            vehicle_num = 0
            for lane_id in in_road.n_lane_id:
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                vehicle_num += len(n_vehicle_id)
                for vehicle_id in n_vehicle_id:
                    vehicle_speed_sum += vehicle_2_speed[vehicle_id]
            if vehicle_num == 0:
                road_avg_speed = speed_limit
            else:
                road_avg_speed = vehicle_speed_sum * 1.0 / vehicle_num
            n_road_delay.append(1. - road_avg_speed / speed_limit)

        n_road_delay = np.array(n_road_delay)
        return n_road_delay

    # 当前agent的inter的所有edge(road)平均存在车辆数
    def _inter_2_num_vehicle(self, agent):
        lane_2_num_vehicle = self.eng.get_lane_vehicle_count()
        return self._concat_by_unit(lane_2_num_vehicle, agent, unit='intersection', inroad_only=True,
                                    outroad_only=False)

    # 当前agent的inter的所有edge(road)平均等待车辆数
    def _inter_2_num_waiting_vehicle(self, agent):
        lane_2_num_waiting_vehicle = self.eng.get_lane_waiting_vehicle_count()
        return self._concat_by_unit(lane_2_num_waiting_vehicle, agent, unit='intersection', inroad_only=True,
                                    outroad_only=False)

    # 当前agent的inter的总进口edge(road)车辆等待时间
    def _inter_2_sum_waiting_time(self, agent):
        # the sum of waiting times of vehicles on the intersection since their last halt
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_waiting_time = self._get_vehicle_waiting_time()

        inter_waiting_time = 0.
        for in_road in agent.n_in_road:
            for lane_id in in_road.n_lane_id:
                for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                    inter_waiting_time += vehicle_2_waiting_time[vehicle_id]
        inter_waiting_time = np.array([inter_waiting_time])
        return inter_waiting_time

    # 当前agent的inter的平均进口edge(road)车辆延误率
    def _inter_2_delay(self, agent):
        # delay of agent = 1. - inter_avg_speed / speed_limit
        # by default the speed_limit is 11.11
        speed_limit = 11.11
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_speed = self.eng.get_vehicle_speed()

        vehicle_speed_sum = 0.
        vehicle_num = 0

        for in_road in agent.n_in_road:
            for lane_id in in_road.n_lane_id:
                n_vehicle_id = lane_2_n_vehicle_id[lane_id]
                vehicle_num += len(n_vehicle_id)
                for vehicle_id in n_vehicle_id:
                    vehicle_speed_sum += vehicle_2_speed[vehicle_id]

        if vehicle_num == 0:
            inter_avg_speed = speed_limit  # follow the implementation of GeneraLight
        else:
            inter_avg_speed = vehicle_speed_sum * 1.0 / vehicle_num
        inter_delay = np.array([1. - inter_avg_speed / speed_limit])
        return inter_delay

    # 当前agent的inter的总进口edge(road)车道压力
    def _inter_2_pressure(self, agent):
        lane_2_vehicle_count = self.eng.get_lane_vehicle_count()
        pressure = 0.
        for in_road in agent.n_in_road:
            for lane_id in in_road.n_lane_id:
                pressure += lane_2_vehicle_count[lane_id]
        for out_road in agent.n_out_road:
            for lane_id in out_road.n_lane_id:
                pressure -= lane_2_vehicle_count[lane_id]
        return np.array([pressure])

    # 当前agent的inter可视化图形
    def _inter_2_vehicle_position_image(self, agent: TSCAgent, grid_height=4, grid_width=4):
        # USE WITH CAUTION: this implementation follows IntelliLight and is only applicable for squared intersection
        area_height, area_width = 600, 600
        map_of_car = np.zeros((1, area_height // grid_height, area_width // grid_width))

        inter_x, inter_y = agent.inter_dict["point"]["x"], agent.inter_dict["point"]["y"]
        vehicle_2_distance = self.eng.get_vehicle_distance()
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()

        for road in agent.n_road:
            for lane_id in road.n_lane_id:
                start_x, start_y, norm_x, norm_y = self._get_lane_start_location(road.road_dict,
                                                                                 int(lane_id.split('_')[-1]))
                for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                    vehicle_distance = vehicle_2_distance[vehicle_id]
                    vehicle_x, vehicle_y = start_x + vehicle_distance * norm_x, start_y + vehicle_distance * norm_y
                    transform_x, transform_y = int((vehicle_x - inter_x + 300) / grid_width), int(
                        (vehicle_y - inter_y + 300) / grid_height)
                    transform_x = max(min(transform_x, int(area_width / grid_width) - 1), 0)
                    transform_y = max(min(transform_y, int(area_height / grid_height) - 1), 0)

                    flip_y = max(min(area_height // grid_height - transform_y, int(area_height / grid_height) - 1), 0)
                    map_of_car[0, flip_y, transform_x] += 1
        return map_of_car

    # 获得进口lane的位置XY
    def _get_lane_start_location(self, road_dict, lane_idx):
        road_start_point = (road_dict["points"][0]["x"], road_dict["points"][0]["y"])
        road_end_point = (road_dict["points"][1]["x"], road_dict["points"][1]["y"])
        delta_x, delta_y = road_end_point[0] - road_start_point[0], road_end_point[1] - road_start_point[1]
        norm = math.sqrt(delta_x ** 2 + delta_y ** 2)
        delta_x, delta_y = delta_x / norm, delta_y / norm

        lane_width = 0.
        for lane_jdx in range(lane_idx):
            lane_width += road_dict["lanes"][lane_jdx]["width"]
        lane_width += road_dict["lanes"][lane_idx]["width"] / 2
        bias_x, bias_y = delta_y * lane_width, -1. * delta_x * lane_width  # 90 degree clockwise rotation
        return road_start_point[0] + bias_x, road_start_point[1] + bias_y, delta_x, delta_y

    # 当前agent的inter的current phase one-hot编码
    def _inter_2_current_phase(self, agent: TSCAgent, index=False):
        if index:
            return agent.current_phase

        phase_one_hot = np.zeros(len(agent.n_phase))
        phase_one_hot[agent.current_phase] = 1.0
        return phase_one_hot
        # current_phase = agent.current_phase
        # return np.array([current_phase])

    # 当前agent的inter的current phase duration
    def _inter_2_current_phase_duration(self, agent: TSCAgent):
        current_phase_duration = np.array([agent.current_phase_time])
        return current_phase_duration

    # 当前agetn的下一个phase oneh-hot编码
    def _inter_2_next_phase(self, agent: TSCAgent):
        phase_one_hot = np.zeros(len(agent.n_phase))
        phase_one_hot[(agent.current_phase + 1) % len(agent.n_phase)] = 1.0
        return phase_one_hot

    # 当前agent的inter是否被切换phase one-hot编码
    def _inter_2_phase_has_changed(self, agent: TSCAgent):
        if self.last_action_n is None or self.current_action_n is None:
            return np.zeros(1)
        inter_idx = agent.inter_idx
        if self.last_action_n[inter_idx] == self.current_action_n[inter_idx]:
            return np.zeros(1)
        else:
            return np.ones(1)

    # 当前agent的inter中 1个 min phase duration 通过的 车辆数
    def _inter_2_num_passed_vehicle_since_last_action(self, agent: TSCAgent):
        n_passed_vehicle_id = self._get_n_vehicle_id_passed_since_last_action(agent)
        return np.array(len(n_passed_vehicle_id))

    # 当前agent的inter中 1个 min phase duration 通过的总车辆行驶时间
    def _inter_2_sum_travel_time_since_last_action(self, agent: TSCAgent):
        n_passed_vehicle_id = self._get_n_vehicle_id_passed_since_last_action(agent)
        sum_travel_time = 0

        for passed_vehicle_id in n_passed_vehicle_id:
            sum_travel_time += self._vehicle_trajectory[passed_vehicle_id][-2]["time_on_lane"]
        return np.array(sum_travel_time)

    # 当前agent的inter中 1个 min phase duration 通过的 车辆id
    def _get_n_vehicle_id_passed_since_last_action(self, agent: TSCAgent):
        vehicle_2_trajectory = self._get_vehicle_trajectory()
        n_passed_vehicle_id = []

        for vehicle_id, trajectory in vehicle_2_trajectory.items():
            if len(trajectory) < 2:
                continue

            if trajectory[-2]["lane_id"] in agent.n_in_lane_id \
                    and trajectory[-1]["lane_id"] in agent.n_out_lane_id \
                    and trajectory[-1]["time_on_lane"] < self.interval:
                n_passed_vehicle_id.append(vehicle_id)

        return n_passed_vehicle_id

    # other agent current_phase
    def _other_inter_2_current_phase(self, agent: TSCAgent):
        other_current_phase = []
        for other_agent in self.agents:
            if other_agent.inter_idx == agent.inter_idx: continue
            current_phase = self._inter_2_current_phase(other_agent)
            other_current_phase.append(current_phase)

        return np.concatenate(other_current_phase, axis=-1)

    # other agent current_phase_duration
    def _other_inter_2_current_phase_duration(self, agent: TSCAgent):
        other_current_phase_duration = []
        for other_agent in self.agents:
            if other_agent.inter_idx == agent.inter_idx: continue
            current_phase_duration = self._inter_2_current_phase_duration(other_agent)
            other_current_phase_duration.append(current_phase_duration)

        return np.concatenate(other_current_phase_duration, axis=-1)

    # cityflow的车辆平均运行时间
    def _world_2_average_travel_time(self):
        return self.eng.get_average_travel_time()

    # cityflow的车辆平均等待长度
    def _world_2_average_queue_length(self):
        sample_number = self._cache_queue_length[0]
        mean = (self._cache_queue_length[1] / sample_number) if sample_number > 0 else 0
        std = math.sqrt(self._cache_queue_length[2] / sample_number - mean * mean) if sample_number > 1 else 0
        return mean, std

    # cityflow的车辆平均吞吐量(通行能力)
    def _world_2_average_throughput(self):
        sample_number = self._cache_throughput[0]
        mean = (self._cache_throughput[1] / sample_number) if sample_number > 0 else 0
        std = math.sqrt(self._cache_throughput[2] / sample_number - mean * mean) if sample_number > 1 else 0
        return mean, std

    # cityflow 的车辆平均延误
    def _world_2_average_delay(self):
        sample_number = self._cache_average_delay[0]
        mean = (self._cache_average_delay[1] / sample_number) if sample_number > 0 else 0
        std = math.sqrt(self._cache_average_delay[2] / sample_number - mean * mean) if sample_number > 1 else 0
        return mean, std

    # get speed < 0.1m/s 的 车辆等待时间， 可以根据step每次调用该函数来累计waite time
    def _get_vehicle_waiting_time(self):
        n_vehicle_id = self.eng.get_vehicles()
        vehicle2speed = self.eng.get_vehicle_speed()
        for vehicle_id in n_vehicle_id:
            if vehicle_id not in self._vehicle_waiting_time.keys():  # vehicle appears for the first time
                self._vehicle_waiting_time[vehicle_id] = 0
            elif vehicle2speed[vehicle_id] < 0.1:  # vehicle is waiting
                self._vehicle_waiting_time[vehicle_id] += 1
            else:  # vehicle is moving
                self._vehicle_waiting_time[vehicle_id] = 0
        return self._vehicle_waiting_time

    # get 每个step下的 vehicle的轨迹信息
    def _get_vehicle_trajectory(self):
        cur_time = self.eng.get_current_time()
        if cur_time <= self._vehicle_trajectory_last_update_time:
            return self._vehicle_trajectory

        self._vehicle_trajectory_last_update_time = cur_time
        vehicle_2_lane = self._get_vehicle_2_lane()  # {veh_id: lane_id}
        n_vehicle_id = self.eng.get_vehicles(include_waiting=False)
        for vehicle_id in n_vehicle_id:
            if vehicle_id not in self._vehicle_trajectory.keys():  # vehicle appears for the first time
                self._vehicle_trajectory[vehicle_id] = [{
                    "lane_id": vehicle_2_lane[vehicle_id],
                    "enter_time": int(cur_time),
                    "time_on_lane": 0
                }]
            else:
                if vehicle_id not in vehicle_2_lane.keys():
                    continue
                if vehicle_2_lane[vehicle_id] == self._vehicle_trajectory[vehicle_id][-1]["lane_id"]:  # on last lane
                    self._vehicle_trajectory[vehicle_id][-1]["time_on_lane"] += 1
                else:  # on a new lane
                    self._vehicle_trajectory[vehicle_id].append({
                        "lane_id": vehicle_2_lane[vehicle_id],
                        "enter_time": int(cur_time),
                        "time_on_lane": 0
                    })
        return self._vehicle_trajectory

    # get {lane_id:[veh_id,....]} --> {veh_id: lane_id}
    def _get_vehicle_2_lane(self):
        lane_2_n_vehicle_id = self.eng.get_lane_vehicles()
        vehicle_2_lane = {}
        for lane_id in lane_2_n_vehicle_id.keys():
            for vehicle_id in lane_2_n_vehicle_id[lane_id]:
                vehicle_2_lane[vehicle_id] = lane_id
        return vehicle_2_lane

    # 辅助函数
    def _concat_by_unit(self, lane_stat_dict, agent: TSCAgent, unit, inroad_only, outroad_only):
        assert unit in ['intersection', 'road', 'lane']

        result = []
        roadSet = agent.n_road
        if inroad_only:
            roadSet = agent.n_in_road
        elif outroad_only:
            roadSet = agent.n_out_road
        # for road in (agent.n_in_road if inroad_only else agent.n_road):
        for road in roadSet:
            result_by_road = []
            for lane_id in road.n_lane_id:
                result_by_road.append(lane_stat_dict[lane_id])
            if unit == 'lane':
                result_by_road = np.array(result_by_road)
            elif unit in ['road', 'intersection']:
                result_by_road = np.mean(result_by_road)
            result.append(result_by_road)

        if unit == 'lane':
            result = np.concatenate(result)
        elif unit == 'road':
            result = np.array(result)
        elif unit == 'intersection':
            result = np.array([np.mean(result)])
        return result
