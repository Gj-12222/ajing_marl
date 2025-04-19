

#
# class EnvConfig:
#     attack_angle = 60  # attack angle
#     defense_angle = 90  # defense angle
#     fire_range = 0.3  # fire range
#     jam_range = 0.6  # jam range

class EnvConfig(object):
    def __init__(self):
        self.env_name = 'uav_5v5'
        # 智能体参数
        self.num_adversaries = 5
        self.num_agents = 5
        self.display = True
        self.attack_angle = 90  # attack angle
        self.defense_angle = 90 # defense angle
        self.fire_range = 0.3  # fire range
        self.compute_range = 0.7  #
        self.jam_range = 0.6  # jam range

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
