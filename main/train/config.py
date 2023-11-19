## 配置文件
import torch


# 定义训练所需的参数：
#     环境相关参数
#     训练用的超参数定义
#     checkpointing（用于存储数据和模型）
#     测试阶段的参数
# Environment环境相关参数

class TrainConfig(object):
    def __init__(self):
        self.scenario_name = "uavs_5v5"  # 场景
        self.num_adversaries = 5 # red
        self.num_agents = 5  # blue
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 算法配置参数
        self.algo_name = 'maddpg'
        self.network_structure = 'share' if 'share' in self.algo_name else 'no_share'
        self.update_step = 10
        self.learn_type = 'off-policy'
        self.algo_config = {'red': None, 'blue': None}
        self.adv_policy = "CTDE"
        self.good_policy = "CTDE"
        self.adv_algorithm = "masac"
        self.good_algorithm = "masac"

        self.seed = 251  # 幸运种子
        self.hidden_dim = 256  # 隐藏层
        self.batch_size = 512  # off-policy data size of each batch
        self.max_episode_len = 300  #
        self.episodes = 20000
        self.epochs = 1
        self.td_lammda = 0.8
        self.lr = 1e-4
        self.gamma = 0.99  # 56, 114, 229, 459 的max_episode_len 对应 0.96, 0.98, 0.99, 0.995 的gamma

        # 保存配置参数
        self.load_model = False
        self.save_name = f"{self.num_adversaries}v{self.num_agents}_{self.scenario_name}"
        self.save_model_dir = "./metrics/model/save/"
        self.save_data_dir = "./metrics/data/learning_curves/"
        self.save_rate = 2
        self.load_dir = "./metrics/model/load/"

        self.data_file_dir = 'data_name.pkl'
        self.create_xslx_dir = 'data_name.xlsx'
        self.plot_dir = "./plot_learning_curve/"

        # 特殊算法固有的参数
        self.init_alpha = 0.02
        self.fix_alpha = True
        self.use_target_actor = True
        self.epsilon = 1e-6


class EnvConfig(object):
    def __init__(self):
        self.env_name = 'uav_5v5'
        # 智能体参数
        self.num_adversaries = 5
        self.num_agents = 5
        self.display = True
        self.attack_angle = 90
        self.defense_angle = 90
        self.fire_range = 30
        self.comput_range = 0.7
        self.jam_range = 0.6
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
