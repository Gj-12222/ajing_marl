
# for easy processing tsc in cityflow
import os
import pandas as pd
from math import atan2, pi, sqrt
from typing import List, Tuple
from Env.utils import list_with_unique_element


class TSCAgent(object):
    def __init__(self, inter_idx, inter_id, inter_dict, env):
        self.current_phase = None
        self.phase_num = None

        self.obs_dim = None
        self.action_dim = None
        self.tls_type = None
        self.action = None

        self.action_callback = None
        self.death = False

        self.inter_idx = inter_idx
        self.inter_id = inter_id
        self.inter_dict = inter_dict
        self.env = env
        self.eng = env.eng
        self.last_phase = None
        self.current_phase = 0
        self.current_phase_time = 0
        self.yellow_phase = -1
        self.current_phase_yellow = 0
        self._yellow_time = 3



        self.n_road = []  # type: List[Road]
        self.n_in_road = []  # type: List[Road]
        self.n_out_road = []  # type: List[Road]
        self.n_lane_id = []  # type: List[str]
        self.n_in_lane_id = []  # type: List[str]
        self.n_out_lane_id = []  # type: List[str]

        # scan all road
        for road_id in self.inter_dict['roads']:
            road = self.env.id2road[road_id]
            self.n_road.append(road)

        # scan all roadlink
        self.n_roadlink = []  # type: List[RoadLink]
        self.n_num_lanelink = []  # type: List[int]
        for roadlink_dict in self.inter_dict['roadLinks']:
            roadlink = RoadLink(roadlink_dict, self)
            self.n_roadlink.append(roadlink)
            self.n_num_lanelink.append(len(roadlink.n_lanelink_id))

        # scan all in_road and out_road by roadlink
        for roadlink in self.n_roadlink:
            self.n_in_road.append(self.env.id2road[roadlink.startroad_id])
            self.n_out_road.append(self.env.id2road[roadlink.endroad_id])
        self.n_in_road = list_with_unique_element(self.n_in_road)
        self.n_out_road = list_with_unique_element(self.n_out_road)

        # fill in lane_id
        for road in self.n_road:
            self.n_lane_id.extend(road.n_lane_id)
        for in_road in self.n_in_road:
            self.n_in_lane_id.extend(in_road.n_lane_id)
        for out_road in self.n_out_road:
            self.n_out_lane_id.extend(out_road.n_lane_id)

        # scan all phase
        self.n_phase = []  # type: List[Phase]
        for phase_idx, phase_dict in enumerate(self.inter_dict['trafficLight']['lightphases']):
            if len(phase_dict['availableRoadLinks']) > 0:
                self.n_phase.append(Phase(phase_idx, phase_dict, self))

        self.n_neighbor_idx = [self.inter_idx]  # this will be determined in TSCEnv once all intersections are scanned
        self.phase_2_passable_lane_idx = self._get_phase_2_passable_lane_idx()
        self.phase_2_passable_lanelink_idx = self._get_phase_2_passable_lanelink_idx()

        self.phase_num = len(self.n_phase)
        self.action_callback = None

        self.tls_type = env.agent_cfg['tls_type']
        if self.tls_type == 'shift':
            self.action_type = 'discrete'
            self.action_dim = [2]  # [0,1]  切换相位
        elif self.tls_type == 'select':
            self.action_type = 'discrete'
            self.action_dim = [self.phase_num]  # 8个相位 4个组合
        elif self.tls_type == 'adjust':
            self.action_type = 'continuous'
            self.action_dim = [2]   # [duration, phase]
            self.action_max = env.agent_cfg['action_max']   # 最大绿灯时间
            self.action_min = env.agent_cfg['duration_min']  # 满足最小绿灯时间 + 黄灯时间
        else:
            raise print('Error: tls_type is no match! ')
        self.obs_dim = None

        # special tsc:
        self.special_tsc = None
        if "N0110" == self.inter_id:
            self.special_tsc = [{'phase_index': 5, 'phase_time': 3, 'need_yellow_phase': False}]  # N0110 3s西全红

    # 所有phase的允许通过的lane的 one-hot编码
    def _get_phase_2_passable_lane_idx(self):
        phase_2_passable_lane_idx = []
        for phase_idx in range(len(self.n_phase)):
            n_lane = [0 for _ in range(len(self.n_in_lane_id))]
            # 每个相位允许通过的车道index
            for pass_lane_id in self.n_phase[phase_idx].n_available_startlane_id:
                lane_idx = self.n_in_lane_id.index(pass_lane_id)
                n_lane[lane_idx] = 1
            phase_2_passable_lane_idx.append(n_lane)
        return phase_2_passable_lane_idx

    # 所有phase的允许通过的lane的lanelink的 one-hot编码
    def _get_phase_2_passable_lanelink_idx(self):
        phase_2_passable_lanelink_idx = []
        for phase_idx in range(len(self.n_phase)):
            n_lanelink = []
            for roadlink_idx, roadlink in enumerate(self.n_roadlink):
                if roadlink_idx in self.n_phase[phase_idx].n_available_roadlink_idx:
                    n_lanelink.extend([1 for _ in range(self.n_num_lanelink[roadlink_idx])])
                else:
                    n_lanelink.extend([0 for _ in range(self.n_num_lanelink[roadlink_idx])])
            phase_2_passable_lanelink_idx.append(n_lanelink)
        return phase_2_passable_lanelink_idx

    def step(self, interval, action_index):
        self.last_phase = self.current_phase

        if self.tls_type == 'shift':  # 表示 [否，是] 切换当前 phase  one-hot编码[0,0]
            if self.current_phase_yellow == self.yellow_phase:  # -1  -1
                if self.current_phase_time < self._yellow_time: # 1< 3
                    self.current_phase_time += interval  # 1+1+1+1
                else:
                    self.current_phase_time = interval  # 1
                    self.current_phase_yellow = - self.current_phase_yellow  # 取反  1
                    self.eng.set_tl_phase(self.inter_id, self.current_phase)
            elif self.action[action_index] == 0:  # 满足3s黄灯结束后，如果action不变
                self.current_phase_time += interval
                # 检查 special tsc
                self._check_special_tsc()

            elif self.action[action_index] == 1:  # match index
                self.current_phase = (self.current_phase + 1) % self.phase_num   # 按顺序切换
                self.current_phase_time = interval  # 1
                self.current_phase_yellow = self.yellow_phase  # -1
        elif self.tls_type == 'select':  # 表示 从[pahse_0,phase_1,...,phase_4] 4个
            # one-out编码[0, 0, 0, 0, 0, 0, 0, 0]
            # Set each trafficlight phase to specified action
            if self.current_phase_yellow == self.yellow_phase:
                if self.current_phase_time < self._yellow_time:
                    self.current_phase_time += interval
                else:
                    self.eng.set_tl_phase(self.inter_id, self.n_phase[self.action].phase_idx)
                    self.current_phase_yellow = - self.current_phase_yellow  # 取反
                    self.current_phase = self.action  # # 按顺序切换
                    self.current_phase_time = interval
            elif self.action == self.current_phase:
                self.current_phase_time += interval
                # 检查 special tsc
                self._check_special_tsc()
            else:
                self.current_phase_yellow = self.yellow_phase
                self.current_phase_time = interval
        # TODO:
        elif self.tls_type == 'adjust':  # 表示调整相位duration时间，有界[duration_min, duration_max]
            self.eng.set_tl_phase(self.inter_id, self.action)

        if self.env.save_play:
            self._save_play()

    def eval_step(self, interval):
        duration_total_time = 0.0
        for phase in self.n_phase:
            duration_total_time += phase.duration_time
        current_time = self.env.world_time
        if current_time <= 3600:  # 6-7点
            duration_time = current_time % duration_total_time
            phase_time = 0.0
            for phase in self.n_phase:
                phase_time += phase.duration_time
                if duration_time - phase_time < 0:
                    self.eng.set_tl_phase(self.inter_id, phase.phase_idx)
                    print(f"duration_time:{duration_time}, phase_id:{phase.phase_idx}")
                    break
        else:
            print('超过3600， 终止！')

    def reset(self):
        self.current_phase = 0
        self.current_phase_time = 0
        self.eng.set_tl_phase(self.inter_id, self.n_phase[self.current_phase].phase_idx)

    def _save_play(self):
        save_path = os.path.join(self.env.cfg['save_log_path'], self.env.agent_cfg['algo_name'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path_to_log_file = os.path.join(save_path, "signal_inter_{0}.txt".format(self.inter_id))
        df = [self.env.world_time, self.current_phase]
        df = pd.DataFrame(df)
        df = df.transpose()
        df.to_csv(path_to_log_file, mode='a', header=False, index=False)

    def _check_special_tsc(self):
        if self.special_tsc:
            for special_tsc in self.special_tsc:
                # 触发特殊 phase
                if self.current_phase == special_tsc['phase_index']:
                    # 超过特殊 phase 的持续时间，强制切换相位
                    if self.current_phase_time >= special_tsc['phase_time']:
                        self.current_phase = (self.current_phase + 1) % self.phase_num  # 按顺序切换
                        # 需要进入 3s yellow phase 再切
                        if special_tsc['need_yellow_phase']:
                            self.current_phase_yellow = self.yellow_phase  # -1
                        else:  # 不需要yellow phase，直接切
                            self.eng.set_tl_phase(self.inter_id, self.current_phase)

    def __str__(self):
        return self.inter_id


class Phase:
    def __init__(self, phase_idx, phase_dict, intersection):
        self.phase_idx = phase_idx
        self.intersection = intersection
        self.duration_time = phase_dict['time']
        self.n_available_roadlink_idx = phase_dict['availableRoadLinks']  # type: List[int]
        self.phase_dict = phase_dict
        self.n_available_lanelink_id = []  # type: List[Tuple[str, str]]
        self.n_available_startlane_id = []  # type: List[str]

        for available_roadlink_idx in self.n_available_roadlink_idx:
            roadlink = self.intersection.n_roadlink[available_roadlink_idx]
            self.n_available_lanelink_id.extend(roadlink.n_lanelink_id)
            self.n_available_startlane_id.extend(roadlink.n_startlane_id)
        self.n_available_startlane_id = list_with_unique_element(self.n_available_startlane_id)

    def __str__(self):
        return str({
            'phase_idx': self.phase_idx,
            'available_roadlink': self.n_available_roadlink_idx
        })


class RoadLink:
    def __init__(self, roadlink_dict, intersection):
        self.intersection = intersection

        self.startroad_id = roadlink_dict['startRoad']
        self.endroad_id = roadlink_dict['endRoad']

        # scan each lane link
        self.n_lanelink_id = []  # type: List[Tuple[str, str]]
        self.n_startlane_id = []  # type: List[str]
        for lanelink_dict in roadlink_dict['laneLinks']:
            startlane_id = '{}_{}'.format(self.startroad_id, lanelink_dict['startLaneIndex'])
            endlane_id = '{}_{}'.format(self.endroad_id, lanelink_dict['endLaneIndex'])
            self.n_lanelink_id.append((startlane_id, endlane_id))
            self.n_startlane_id.append(startlane_id)

        self.n_startlane_id = list_with_unique_element(self.n_startlane_id)

    def __str__(self):
        return str({'startroad_id': self.startroad_id,
                    'endroad_id': self.endroad_id,
                    'n_lanelink_id': self.n_lanelink_id})

class Road:
    def __init__(self, road_id, road_dict):
        self.road_id = road_id
        self.road_dict = road_dict

        self.road_direction = self._get_road_direction()
        self.n_lane_id = []  # type: List[str]
        self.length = self._get_road_length()
        for lane_idx in range(len(self.road_dict['lanes'])):
            self.n_lane_id.append('{}_{}'.format(road_dict['id'], lane_idx))

    def _get_road_direction(self):
        # 0.00 -> east, 1/2 * pi -> north, pi -> west, 3/2 * pi -> south
        delta_x = self.road_dict['points'][1]['x'] - self.road_dict['points'][0]['x']
        delta_y = self.road_dict['points'][1]['y'] - self.road_dict['points'][0]['y']
        direction = atan2(delta_y, delta_x)
        return direction if direction >= 0 else (direction + 2 * pi)

    def _get_road_length(self):
        delta_x = self.road_dict['points'][1]['x'] - self.road_dict['points'][0]['x']
        delta_y = self.road_dict['points'][1]['y'] - self.road_dict['points'][0]['y']
        return sqrt(delta_x ** 2 + delta_y ** 2)

    def __str__(self):
        return str({'road_id': self.road_id,
                    'num_lanes': len(self.n_lane_id)})
