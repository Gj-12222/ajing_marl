## 配置文件

class Config(object):
    def __init__(self):
        self.scenario = "uavs_5v5"
        self.max_episode_len = 300
        self.num_episodes = 20000
        self.td_lammda = 0.8
        self.lr = 1e-4
        self.gamma = 0.99  # 56, 114, 229, 459 的max_episode_len 对应 0.96, 0.98, 0.99, 0.995 的gamma

        self.seed = 3407  # 幸运种子
        self.batch_size = 512
        self.num_units = 128
        # 智能体参数
        self.num_adversaries = 5
        self.num_agents = 10
        self.adv_policy = "CTDE"
        self.good_policy = "CTDE"
        self.adv_algorithm = "maddpg"
        self.good_algorithm = "maddpg"
        # 保存模型数据的配置参数
        self.save_name = "4v8_uav"
        self.save_model_dir = "./training/model/save/"
        self.save_data_dir = "./training/data/learning_curves/"
        self.save_rate = 2
        self.load_dir = "./training/model/load/"
        self.restore = False
        self.display = False
        # self.plot_dir = "./plot_learning_curve/"
        # 特殊算法固有的参数
        self.init_alpha = 0.02
        self.fix_alpha = True
        self.use_target_actor = True