r'''

rl algorithm config file

'''


# 余下为算法/train的超参
# 简单解读：idqn - independence deep q-network

idqn_config = dict(
    learn_type='off_policy',
    action_select='sample',  # sample, distribution
    algo_type='value_based',  # value_based, policy_based
    obs_list=[],
    reward_list=[],
    metric_list=[],
    reward_weight=[],
    network_structure='no share',
    epochs=1,
    episodes=200,
    max_episode_len=3600,
    batch_size=128,
    hidden_dim=128,
    activate_fn='None',
    buffer_size=1e6,
    epsilon=0.8,
    epsilon_min=0.000001,
    epsilon_decay=0.95,
    gamma=0.99,
    tau=5e-5,
    critic_lr=4e-4,
    update_step=10)

irainbow_config = dict(
    learn_type='off_policy',
    action_select='sample',  # sample, distribution
    algo_type='value_based',  # value_based, policy_based
    obs_list=[],
    reward_list=[],
    metric_list=[],
    reward_weight=[],
    network_structure='no share',
    atom=51,  # default to c51
    epochs=1,
    episodes=200,
    max_episode_len=3600,
    batch_size=128,
    hidden_dim=128,
    activate_fn='softmax',
    buffer_size=1e6,
    epsilon=0.8,
    epsilon_min=0.000001,
    epsilon_decay=0.95,
    gamma=0.99,
    tau=5e-5,
    q_lr=4e-4,
    adv_lr=4e-5,
    update_step=100)

masac_discrete_config = dict(
    learn_type='off_policy',
    action_select='distribution',  # sample, distribution
    algo_type='policy_based',  # value_based, policy_based
    obs_list=[],
    reward_list=[],
    metric_list=[],
    reward_weight=[],
    epochs=1,
    episodes=200,
    max_episode_len=3600,
    network_structure='no_share',
    batch_size=128,
    hidden_dim=128,
    activate_fn='None',
    buffer_size=1e6,
    gamma=0.99,
    tau=5e-5,
    critic_lr=4e-4,
    actor_lr=5e-4,
    alpha_lr=5e-4,
    update_step=100)

qmix_config = dict(
    learn_type='on_policy',
    action_select='sample',  # sample, distribution
    algo_type='value_based',  # value_based, policy_based
    obs_list=[],
    reward_list=[],
    metric_list=[],
    reward_weight=[],
    epochs=1,
    episodes=200,
    max_episode_len=3600,
    network_structure='no_share',
    batch_size=128,
    hidden_dim=128,
    activate_fn='None',
    buffer_size=1e6,
    gamma=0.99,
    tau=5e-5,
    critic_lr=4e-4,
    actor_lr=5e-4,
    update_step=100)

masac_config = dict(
    learn_type='off_policy',
    action_select='distribution',  # sample, distribution
    algo_type='policy_based',  # value_based, policy_based
    obs_list=[],
    reward_list=[],
    metric_list=[],
    reward_weight=[],
    epochs=1,
    episodes=200,
    max_episode_len=3600,
    network_structure='no_share',
    batch_size=128,
    hidden_dim=128,
    activate_fn='None',
    distribution_fn="softCategorical",
    buffer_size=1e6,
    gamma=0.99,
    tau=5e-5,
    critic_lr=4e-4,
    actor_lr=5e-4,
    alpha_lr=5e-4,
    update_step=100)

maddpg_config = dict(
    learn_type='off_policy',
    action_select='distribution',  # sample, distribution
    algo_type='policy_based',  # value_based, policy_based
    obs_list=[],
    reward_list=[],
    metric_list=[],
    reward_weight=[],
    epochs=1,
    episodes=200,
    max_episode_len=3600,
    network_structure='no_share',
    batch_size=128,
    hidden_dim=128,
    distribution_fn='softCategorical',  # softCategorical, GumbelSoftCategorical
    buffer_size=1e6,
    gamma=0.99,
    tau=5e-5,
    critic_lr=4e-4,
    actor_lr=5e-4,
    soft_update_freq=100,
    update_step=100)

matd3_config = dict(
    learn_type='off_policy',
    action_select='distribution',  # sample, distribution
    algo_type='policy_based',  # value_based, policy_based
    obs_list=[],
    reward_list=[],
    metric_list=[],
    reward_weight=[],
    epochs=1,
    episodes=200,
    max_episode_len=3600,
    network_structure='no_share',
    batch_size=128,
    hidden_dim=128,
    activate_fn='None',
    buffer_size=1e6,
    gamma=0.99,
    tau=5e-5,
    critic_lr=4e-4,
    actor_lr=5e-4,
    soft_update_freq=100,
    delay_update_freq=200,
    update_step=100)

mappo_config = dict(
    learn_type='on_policy',
    action_select='sample',  # sample, distribution
    algo_type='policy_based',  # value_based, policy_based
    # env params
    obs_list=[],
    reward_list=[],
    metric_list=[],
    reward_weight=[],
    # ppo params
    epochs=1,
    episodes=200,
    max_episode_len=3600,
    use_clip_valueLoss=True,
    clip_param=0.2,
    repeat_epoch=10,  # repeat times
    mini_batch=1,
    entropy_coef=0.01,
    valueLoss_coef=0.5,
    use_max_gradNorm=True,
    max_gradNorm=10.0,
    use_gae=True,
    gae_lambda=0.95,
    use_proper_time_limits=False,
    use_huber_loss=False,
    use_value_active_masks=True,
    use_policy_active_masks=True,
    huber_delta=10.0,
    gamma=0.99,
    tau=5e-5,
    soft_update_freq=100,
    update_step=100,
    # network params
    network_structure='no_share',
    batch_size=128,
    hidden_dim=256,
    use_popart=False,
    use_valueNorm=True,
    use_featureNorm=True,
    use_orthogonal=True,
    last_action_layer_gain=0.01,
    activate_fn='None',
    # rnn params
    use_rnn_policy=True,
    rnn_layer_dim=2,
    timestep=1,
    # replay buffer params
    buffer_size=1e6,
    # optimizer params
    critic_lr=4e-4,
    actor_lr=5e-4
    # optimizer_epsilon=1e-5,
    # weight_delay=0
    )

dqn_share_config = dict(
    learn_type='off_policy',
    action_select='sample',  # sample, distribution
    algo_type='value_based',  # value_based, policy_based
    obs_list=['agent_2_index'] + [],
    reward_list=[],
    metric_list=[],
    reward_weight=[],
    network_structure='share',
    epochs=1,
    episodes=200,
    max_episode_len=3600,
    batch_size=128,
    hidden_dim=128,
    activate_fn='None',
    buffer_size=1e6,
    epsilon=0.8,
    epsilon_min=0.000001,
    epsilon_decay=0.95,
    gamma=0.99,
    tau=5e-5,
    critic_lr=4e-4,
    soft_update_freq=100,
    update_step=100)

maddpg_share_config = dict(
    learn_type='off_policy',
    action_select='distribution',  # sample, distribution
    algo_type='policy_based',  # value_based, policy_based
    obs_list=['agent_2_index'] + [],
    reward_list=[],
    metric_list=[],
    reward_weight=[],
    epochs=1,
    episodes=200,
    max_episode_len=3600,
    network_structure='share',
    batch_size=128,
    hidden_dim=128,
    distribution_fn='softCategorical',
    buffer_size=1e6,
    gamma=0.99,
    tau=5e-5,
    critic_lr=4e-4,
    actor_lr=5e-4,
    soft_update_freq=100,
    update_step=100)

masac_discrete_share_config = dict(
    learn_type='off_policy',
    action_select='sample',  # sample-从分布中采样的值, distribution-直接输出分布 eg. softmax
    algo_type='policy_based',  # value_based, policy_based
    obs_list=['agent_2_index'] + [],
    reward_list=[],
    metric_list=[],
    reward_weight=[],
    epochs=1,
    episodes=50,
    max_episode_len=3600,
    network_structure='share',
    batch_size=128,
    hidden_dim=128,
    activate_fn='None',
    buffer_size=1e6,
    gamma=0.99,
    tau=5e-5,
    critic_lr=4e-4,
    actor_lr=5e-4,
    alpha_lr=5e-3,
    update_step=100)

mappo_share_config = dict(
    learn_type='on_policy',
    action_select='sample',  # sample, distribution
    algo_type='policy_based',  # value_based, policy_based
    # env params
    obs_list=['agent_2_index'] + [],
    reward_list=[],
    metric_list=[],
    reward_weight=[],
    # ppo params
    epochs=1,
    episodes=200,
    max_episode_len=3600,
    use_clip_valueLoss=True,
    clip_param=0.2,
    repeat_epoch=10,
    mini_batch=5,  # repeat times
    entropy_coef=0.01,
    valueLoss_coef=1,
    use_max_gradNorm=True,
    max_gradNorm=10.0,
    use_gae=True,
    gae_lambda=0.95,
    use_proper_time_limits=False,
    use_huber_loss=True,
    use_value_active_masks=True,
    use_policy_active_masks=True,
    huber_delta=10.0,
    gamma=0.99,
    tau=5e-5,
    soft_update_freq=100,
    update_step=100,
    # network params
    network_structure='share',
    batch_size=128,
    hidden_dim=256,
    use_popart=False,
    use_valueNorm=True,
    use_featureNorm=True,
    use_orthogonal=True,
    last_action_layer_gain=0.01,
    activate_fn='None',
    # rnn params
    use_rnn_policy=False,
    rnn_layer_dim=3,
    timestep=10,
    # replay buffer params
    buffer_size=1e6,
    # optimizer params
    critic_lr=4e-4,
    actor_lr=5e-4,
    # optimizer_epsilon=1e-5,
    # weight_delay=0)
    )

# ---------------------------------------------------------------
#  baseline algorithms
# ---------------------------------------------------------------

TinyLight_config = {
    'learn_type': 'off_policy',
    'action_select': 'sample',  # sample, distribution
    'algo_type': 'value_based',  # value_based, policy_based
    "obs_list": [
        "phase_2_num_vehicle",
        "phase_2_num_waiting_vehicle",
        "phase_2_sum_waiting_time",
        "phase_2_delay",
        "phase_2_pressure",
        "inlane_2_num_vehicle",
        "inlane_2_num_waiting_vehicle",
        "inlane_2_sum_waiting_time",
        "inlane_2_delay",
        "inlane_2_pressure",
        "inter_2_current_phase"
    ],
    "reward_list": ["inter_2_pressure"],
    "reward_weight": [-1.0],
    "metric_list": [
        "world_2_average_travel_time",
        "world_2_average_queue_length",
        # "world_2_average_throughput",
        "world_2_average_delay"
    ],
    "n_layer_1_dim": [16, 18, 20, 22, 24],
    "n_layer_2_dim": [16, 18, 20, 22, 24],
    "learning_rate": 0.001,
    "epsilon": 0.1,
    "buffer_size": 100000,
    "batch_size": 32,
    "gamma": 0.9,
    "tau": 0.1,
    "max_episode_len": 3600,
    "epochs":1,
    "episodes":200,
    "network_structure": 'no_share',
    "update_step":100,
}
