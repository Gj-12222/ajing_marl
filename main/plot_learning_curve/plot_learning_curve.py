"""

plot_loss.py
Only support to export the data(.pkl) of default env(uav_env) of library
注意事项：
*在转化为txt时，应该选择带文本标识符的txt文件 UTF-8
*选择文件路径：change reward_target_logdir
*选择列命名文件：change reward_names 文件内容 = 数据的列命名
*选择绘制图像：更改 243行以及之后的代码


if you want to plot yourself data, you will read this code in .py to coding appropriate code.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from training.Config import Config
# 获取路径
args = Config()
target_logdir = [r"D:"]

sum_reward_target_logdir = [r"D:"]
agent_reward_target_logdir = [r"D:"]
red_blue_sum_rewards_logdir = [r"D:"]

save_logdir = [r"D:"]
test_logdir= [r"D:"]
exp_idx = 0
units = dict

# 列数命名
def names():
    new_names = ['update_times']
    for i in range(12):
        agent_q1_loss_str = 'agent' + str(i) + '_q1_loss'
        new_names.append(agent_q1_loss_str)
        agent_q2_loss_str = 'agent' + str(i) + '_q2_loss'
        new_names.append(agent_q2_loss_str)
        agent_actor_loss_str = 'agent' + str(i) + '_actor_object'
        new_names.append(agent_actor_loss_str)
        agent_alpha_loss_str = 'agent' + str(i) + '_alpha_loss'
        new_names.append(agent_alpha_loss_str)
        agent_mean_y_target_str = 'agent' + str(i) + '_mean(Y_target)'
        new_names.append(agent_mean_y_target_str)
        agent_mean_rew_str = 'agent' + str(i) + '_mean(rew)'
        new_names.append(agent_mean_rew_str)
        agent_mean_target_min_q_str = 'agent' + str(i) + '_mean(target_min_q)'
        new_names.append(agent_mean_target_min_q_str)
        agent_std_y_target_str = 'agent' + str(i) + '_std(Y_target)'
        new_names.append(agent_std_y_target_str)
    return new_names
new_names = names()
condition_names = names()

reward_names = ['average_rewards']
red_blue_names = ['red_rewards', 'blue_rewards','sum_rewards']
agent_reward_names = ['agent0_rewards','agent1_rewards','agent2_rewards','agent3_rewards','agent4_rewards',
                'agent5_rewards','agent6_rewards','agent7_rewards','agent8_rewards','agent9_rewards','agent10_rewards','agent11_rewards']
# 提取所有文件路径
logdirs= []
for logdir in target_logdir:
    if os.path.isdir(logdir) and logdir[-1] == os.sep:
        logdirs += [logdir]
    else:
        basedir = os.path.dirname(logdir)
        fulldir = lambda x: os.path.join(basedir,x)
        prefix = logdir.split(os.sep)[-1]
        print('basedir=',basedir)
        listdir = os.listdir(basedir)
        logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

# 获取data
def make_data(logdir, condition =None):
    global exp_idx
    global units
    exp_names = []
    roots = []
    sub_data = []
    for root, _, files in os.walk(logdir):
        if 'MASAC_VS_MASAC.txt' in files:
            exp_name = None
            try:
                exp_name = files[0][:-4]
                exp_names.append(exp_name)
                roots.append(root)
            except Exception as e:
                print(e)

    roots_names_dict = {exp_names[index]: roots for index in range(len(exp_names))}
    for key, value in roots_names_dict.items():
        print(key, value)
    # 排序
    roots_names_list = sorted(roots_names_dict.items(), key=lambda x:x[0])
    roots_names_dict = {tup[0]:tup[1] for tup in roots_names_list}

    for file, roots in roots_names_dict.items():
        for root in roots:
            log = os.path.join(root, 'MASAC_VS_MASAC.txt')  #
            print(log)
            try:
                data = pd.read_table(log)
            except:
                print('Could not read from %s' % os.path.join(root, 'MASAC_VS_MASAC.txt'))
                continue
            # 在这里加入所有的 ondition_names
            for i, condition_name in enumerate(condition_names):
                condition1 = condition or condition_name or 'exp'
                condition2 = condition1 + '-' + str(exp_idx)
                if condition1 not in units:
                    units[condition1] = 0
                unit = units[condition1]
                units[condition1] += 1
                # performance = 'agent0_q1_loss'
                data.insert(len(data.columns), 'Unit' + str(i), unit)
                data.insert(len(data.columns), 'Condition1'+ str(i), condition1)
                data.insert(len(data.columns), 'Condition2'+ str(i), condition2)
            exp_idx += 1
    sub_data.append(data)
    return sub_data
datas = []
for sub_logdir in logdirs: # 对每个数据文件提取数据
    datas += make_data(logdir=sub_logdir)

# 绘制图像
def plot_data(datas, data_name, xaxis_name,yaxis_name,
              condition1="Condtition1",
              condition2="Condtition1",
              unit="Unit1",
              color_index='deep',
              smooth =1,
              linewidth = 4,
              rank = True,
              performance = True,
              **kwargs):

    performance_rank_dict = {}
    condition2_list = []
    y = np.ones(smooth)
    for i,datum in enumerate(datas):
        condition2_list.append(datum[condition2].values[0])
        row, columns = datum.shape
        # 对每一列都求滑动平均
        x = np.asarray(datum[yaxis_name])
        z = np.ones(len(x))
        smooth_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')  # 这是numpy函数中的卷积函数库
        datum[yaxis_name] = smooth_x
        # 平均性能
        if datum[condition1].values[0] not in performance_rank_dict.keys():
            performance_rank_dict[datum[condition1].values[0]] = np.mean(smooth_x[-len(smooth_x)//10:])
        else:
            performance_rank_dict[datum[condition1].values[0]] += np.mean(smooth_x[-len(smooth_x)//10:])
    # 多个随机种子取平均
    for key in performance_rank_dict.keys():
        seed_num = sum([1 for _ in condition2_list if key in _])
        performance_rank_dict[key] /= seed_num
        print('yaxis_name=', yaxis_name, 'seed_num=', seed_num)
    # 性能排序
    performance_value_list = []
    performance_rank_keys = []
    for key, value in performance_rank_dict.items():
        print(key, value)
        performance_rank_keys.append(key)
        performance_value_list.append(value)

    # 获取列表排序序号
    performance_rank_list = np.argsort(np.argsort(- np.array(performance_value_list)))
    performance_rank_sort_dict = {performance_rank_keys[index]: performance_rank_list[index]
                                  for index in range(len(performance_rank_list))}
    print(performance_rank_list)
    # 拼接图像
    if isinstance(datas, list):
        datas = pd.concat(datas, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.75,)

    # datas 按着lenged排序
    datas.sort_values(by=condition1,axis=0)
    # 生成图像
    """这是各种 seaborn 的颜色 code 需要自取
    ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 
    'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r',
     'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 
     'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 
     'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r',
      'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 
      'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 
      'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',
       'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 
       'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 
       'flag', 'flag_r', 'gist_earth', 'gist_earth_r']
    """
    sns.tsplot(data=datas,
               time=xaxis_name,
               value=yaxis_name,
               unit=unit,
               condition=condition1,
               ci='sd',
               linewidth=linewidth,
               color=sns.color_palette(color_index, len(datas)),
               **kwargs)
    # 设定图像位置
    plt.legend(loc='upper right',
               ncol=1,
               framealpha=0.2,  # 不透明度，0.2表明有20%的不透明度
               handlelength=6,
               borderaxespad=0.,
               )
    # loc='upper center', 'lower right',  'upper left',  'upper left'          mode="expand",
    xscale = np.max(np.asarray(datas[xaxis_name])) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout(pad=0.5)

# 绘制图像
smooth= 50  # 滑动窗口尺寸
estimator = 'mean'
linewidth = 4
rank = True  # 排序
performance = True  # 性能
data_name = 'MASAC_VS_MASAC_'
condition1 = 'Condition1' # 未加随机种子
condition2 = 'Condition2' # 加入随机种子
unit = 'Unit'
condition1_index = None
condition2_index = None
unit_index = None
color_palette = ['Pastel1','Paired','Pastel2', 'Set3', 'autumn','Accent','RdGy','Blues_r']
x_names = 'update_times'  # x轴名称
reward_x_names = 'episodes'
save_name = ''
estimator = getattr(np, estimator)  # choose what to show on main curve: mean? max? min?
j = 0
for i,condition_name in enumerate(condition_names):
    if 'alpha' in condition_name: continue

    if (i -  (3 * 8) - 3) == 0:
        plt.figure()
        save_name = condition_name[7:]
    # i = 0 8 16 24
    if (i -  (3 * 8) - 3) > 0 and (i -  (3 * 8) - 3)  % 8 == 0:
        condition1_index = condition1 + str(i)
        condition2_index = condition2 + str(i)
        unit_index = unit + str(i)
        plot_data(datas=datas,data_name=data_name +condition_name ,xaxis_name=x_names, yaxis_name=condition_name,
                  condition1=condition1_index,condition2=condition2_index,unit=unit_index,color_index=color_palette[j],
                  smooth=smooth,estimator=estimator,linewidth=linewidth,rank=rank,performance=performance)
        j +=  1

manager = plt.get_current_fig_manager()
try:
    manager.resize(*manager.window.maxsize())
except:
    manager.window.showMaximized()
fig = plt.gcf()
fig.set_size_inches((16,9), forward=False)

select_str = ''
exclued_str = ''

fig.savefig(test_logdir[0] + '\\'+ data_name + save_name + '.png',
            bbox_inches='tight',
            dpi=300)

