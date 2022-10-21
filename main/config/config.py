## 配置文件

# 定义训练所需的参数：
#     环境相关参数
#     训练用的超参数定义
#     checkpointing（用于存储数据和模型）
#     测试阶段的参数
# Environment环境相关参数

class Config(object):
    def __init__(self):
        self.scenario = "uavs_5v5"
        self.max_episode_len = 300
        self.num_episodes = 20000
        self.td_lammda = 0.8
        self.lr = 1e-4
        self.gamma = 0.99  # 56, 114, 229, 459 的max_episode_len 对应 0.96, 0.98, 0.99, 0.995 的gamma

        self.seed = 251  # 幸运种子
        self.batch_size = 512
        self.num_units = 256
        # 智能体参数
        self.uav_hper_parms()
        self.num_adversaries = 5
        self.num_agents = 5
        self.adv_policy = "CTDE"
        self.good_policy = "CTDE"
        self.adv_algorithm = "masac"
        self.good_algorithm = "masac"
        self.display = True
        # 保存模型数据的配置参数
        self.save_name = "4v8_uav"
        self.save_model_dir = "./training/model/save/"
        self.save_data_dir = "./training/data/learning_curves/"
        self.save_rate = 2
        self.load_dir = "./training/model/load/"
        self.load_model = False

        self.data_file_dir = 'data_name.pkl'
        self.create_xslx_dir = 'data_name.xlsx'
        self.plot_dir = "./plot_learning_curve/"
        # 特殊算法固有的参数
        self.init_alpha = 0.02
        self.fix_alpha = True
        self.use_target_actor = True
        self.epsilon = 1e-6
    def uav_hper_parms(self):
        self.attack_angle =90
        self.defense_angle = 90
        self.fire_range = 30
        self.comput_range = 0.7
        self.jam_range = 0.6