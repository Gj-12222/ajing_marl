import xlwt
from experiments.train import parse_args
import matplotlib.pyplot as plt
import numpy
import pickle
import xlsxwriter



##创建一个excel文件并导入数据
if __name__ == '__main__':
    # xlwt 库将数据导入Excel并设置默认字符编码为ascii
    workbook = xlwt.Workbook(encoding='ascii')
    #添加一个表 参数为表名
    worksheet = workbook.add_sheet('demo')
    #提取数据
    arglist = parse_args()
    lujing = 'D:/software/PyCharm/pyCharmProject/MADDPG'
    # duikanghuanjing  = '/learning_curves/'
    """MADDPG-VS-MASAC/MADDPG/SAC/DDPG"""
    # maddpg-vs-masac: /训练模型/对抗环境-uavs_红方4v蓝方8/红方_MADDPG_vs蓝方_MASAC-MADDPG-SAC-DDPG/4-红MADDPG-VS-蓝-MASAC-20210927/训练成功
    # maddpg-vs-sac: /训练模型/对抗环境-uavs_红方4v蓝方8/红方_MADDPG_vs蓝方_MASAC-MADDPG-SAC-DDPG/3-红MADDPG-VS-蓝-SAC-20210925/
    # maddpg-vs-maddpg: /训练模型/对抗环境-uavs_红方4v蓝方8/红方_MADDPG_vs蓝方_MASAC-MADDPG-SAC-DDPG/1-红-MADDPG-VS-蓝MADDPG/3-完成/
    # maddpg-vs-ddpg: /训练模型/对抗环境-uavs_红方4v蓝方8/红方_MADDPG_vs蓝方_MASAC-MADDPG-SAC-DDPG/2-红MADDPG-VS-蓝-DDPG/
    """MASAC-VS-MASAC/MADDPG/SAC/DDPG"""
    # masac-vc-masac : /训练模型/对抗环境-uavs_红方4v蓝方8/红方_MASAC_vs蓝方_MASAC-MADDPG-SAC-DDPG/红MASAC-VS-蓝-MASAC/2-完成训练
    # masac-vc-masac : /训练模型/对抗环境-uavs_红方4v蓝方8/红方_MASAC_vs蓝方_MASAC-MADDPG-SAC-DDPG/红MASAC-VS-蓝-MASAC/2-MASAC-VS-MASAC-20210722
    # masac-vs-sac: \训练模型\对抗环境-uavs_红方4v蓝方8\红方_MASAC_vs蓝方_MASAC-MADDPG-SAC-DDPG\1-MASAC-VS-SAC-20210803\训练成功
    # masac-vs-maddpg:\训练模型\对抗环境-uavs_红方4v蓝方8\红方_MASAC_vs蓝方_MASAC-MADDPG-SAC-DDPG\3-MASAC-VS-MADDPG-20210727
    # masac-vs-ddpg: \训练模型\对抗环境-uavs_红方4v蓝方8\红方_MASAC_vs蓝方_MASAC-MADDPG-SAC-DDPG\4-MASAC-VS-DDPG-20210807
    """SAC-VS-MASAC/MADDPG/SAC/DDPG"""
    # sac-vs-masac: /训练模型/对抗环境-uavs_红方4v蓝方8/红方_SAC_vs蓝方_MASAC-MADDPG-SAC-DDPG/SAC-VS-MASAC-20210722/训练完成/
    # sac-vs-sac: /训练模型/对抗环境-uavs_红方4v蓝方8/红方_SAC_vs蓝方_MASAC-MADDPG-SAC-DDPG/2-SAC-VS-SAC-20210909/训练完成/
    # sac-vs-maddpg: /训练模型/对抗环境-uavs_红方4v蓝方8/红方_SAC_vs蓝方_MASAC-MADDPG-SAC-DDPG/3-SAC-VS-MADDPG-20210922/SAC-VS-MADDPG-20210922/
    # sac-vs-ddpg: /训练模型/对抗环境-uavs_红方4v蓝方8/红方_SAC_vs蓝方_MASAC-MADDPG-SAC-DDPG/4-SAC-VS-DDPG-20211003/
    """DDPG-VS-MASAC/MADDPG/SAC/DDPG"""
    # ddpg-vs-masac: /训练模型/对抗环境-uavs_红方4v蓝方8/红方_DDPG_vs蓝方_MASAC-MADDPG-SAC-DDPG/1-DDPG-VS-MASAC/20210911/
    # ddpg-vs-sac: /训练模型/对抗环境-uavs_红方4v蓝方8/红方_DDPG_vs蓝方_MASAC-MADDPG-SAC-DDPG/3-DDPG-VS-SAC/DDPG-VS-SAC-2021-10-9/
    # ddpg-vs-maddpg: /训练模型/对抗环境-uavs_红方4v蓝方8/红方_DDPG_vs蓝方_MASAC-MADDPG-SAC-DDPG/2-DDPG-VS-MADDPG/DDPG-VS-MADDPG-20210912/
    # ddpg-vs-ddpg: /训练模型/对抗环境-uavs_红方4v蓝方8/红方_DDPG_vs蓝方_MASAC-MADDPG-SAC-DDPG/4-DDPG-VS-DDPG-20210929/
    duikanghuanjing = '/训练模型/对抗环境-uavs_红方4v蓝方8/红方_MASAC_vs蓝方_MASAC-MADDPG-SAC-DDPG/2-MASAC-VS-MASAC/2-MASAC-VS-MASAC-20210722/'

    rewards = arglist.exp_name + '_rewards.pkl'
    agloss = arglist.exp_name + '_agloss.pkl'
    agreward = arglist.exp_name + '_agrewards.pkl'
    everagrew = arglist.exp_name + '_everyep_agrew.pkl'
    everallrew = arglist.exp_name + '_everyep_allrew.pkl'
    good_death = arglist.exp_name + '_good_death.pkl'
    adv_death = arglist.exp_name + '_adv_death.pkl'
    file = [ rewards, agloss,agreward,everagrew,everallrew,good_death, adv_death ]
    datas = []
    datass = []
    save_rate_sumreward = 0
    episode_agloss = 1
    save_rate_agreward = 2
    episode_agreward = 3
    episode_sumreward = 4
    episode_good_death = 5
    episode_adv_death = 6
    """② episode_agloss """
    excel = xlsxwriter.Workbook(r'D:\software\PyCharm\pyCharmProject\MADDPG\训练结果图\4V8\新-MASAC\20210817\MASAC_loss\1_2-MASAC-MASAC.xlsx')
    """一、MASAC-VS-MASAC"""
    sheet = excel.add_worksheet('MASAC_VS_MASAC')
    bold = excel.add_format({'bold': True})  # 设置excel样
    # 向excel写数据    # sheet.write(3, 0, 32)
    rew_file_name = lujing + duikanghuanjing + file[episode_agloss]  # save_rate_sumreward
    data = pickle.load(open(rew_file_name, 'rb+'))  # [ episode,N, loss_num ]
    datas = data[-1]  # 这是所有agent的每次更新的loss，按照[[agent1_update1_loss],[agent2_update1_loss],...,[agentn_update1_loss],...,[agent1_updatem_loss],[agent2_updatem_loss],...,[agentn_updatem_loss]]
    len_data = len(data)
    i = 0
    for datas_in in datas:  # datas的list里有1个agent的loss
        if datas_in==None: continue
        for j,datas_in_e in enumerate(datas_in):  # 对每个agent都进行loss提取  进入  [1,2,3,4,5,6,7,8]
            if datas_in_e == [None]:
                sheet.write(int(i/12), j+(i%12)*8, 0)  # 有log_alpha_loss =None
            else:       # j是1~8， 1~96 = j+11*8
                sheet.write(int(i/12), j+(i%12)*8, datas_in_e)
        i += 1
    excel.close()  # 关闭excel
