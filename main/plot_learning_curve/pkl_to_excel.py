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
    duikanghuanjing = '/训练模型/对抗环境-uavs_红方4v蓝方8/红方_MADDPG_vs蓝方_MASAC-MADDPG-SAC-DDPG/2-红MADDPG-VS-蓝-DDPG/'

    rewards =  arglist.exp_name + '_rewards.pkl'
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
    # excel = xlsxwriter.Workbook(r'D:\software\PyCharm\pyCharmProject\MADDPG\testPaper-MADDPG\xlswrite_test_aa.xlsx')
    # sheet = excel.add_worksheet('episode_agloss')
    # bold = excel.add_format({'bold': True})  # 设置excel样
    # # 向excel写数据    # sheet.write(3, 0, 32)
    # rew_file_name = lujing + duikanghuanjing + file[episode_agloss]  # save_rate_sumreward
    # data = pickle.load(open(rew_file_name, 'rb+'))  # [ episode,N, loss_num ]
    # for m in range(len(data[0])):  # 获取列数
    #     for k in range(len(data)):  # 获取行数
    #         sheet.write(k, m, data[k][m])
    # excel.close()  # 关闭excel
    """④ episode_agreward 有用 """
    excel = xlsxwriter.Workbook(r'D:\software\PyCharm\pyCharmProject\MADDPG\训练结果图\4V8\新-MASAC\20210817\16-4v8-红MADDPG-VS-蓝-DDPG.xlsx')
    bold = excel.add_format({'bold': True})  # 设置excel样
    sheet = excel.add_worksheet('episode_agreward')
    # 向excel写数据    # sheet.write(3, 0, 32)
    rew_file_name = lujing + duikanghuanjing + file[episode_agreward]  # save_rate_sumreward
    data = pickle.load(open(rew_file_name, 'rb+'))  # [ episode * step, N, episode ] = [12, 10000]
    # datas = data[-1]  # 随便取的，好像大家都一样
    for m in range(len(data)):  # 获取行数
        for k in range(len(data[0])):  # 获取列数
            sheet.write(k, m, data[m][k])
    # excel.close()  # 关闭excel

    """⑤ episode_sumreward 有用 """
    # excel = xlsxwriter.Workbook(r'D:\software\PyCharm\pyCharmProject\MADDPG\testPaper-MADDPG\1-4v8-红-MADDPG-VS-蓝MADDPG.xlsx')
    sheet = excel.add_worksheet('episode_sumreward')
    bold = excel.add_format({'bold': True})  # 设置excel样
    # 向excel写数据    # sheet.write(3, 0, 32)
    rew_file_name = lujing + duikanghuanjing + file[episode_sumreward]  # save_rate_sumreward
    data = pickle.load(open(rew_file_name, 'rb+'))  # [ episode * step, episode ] = [ 10000,1]
    # datas = data[-1]  # 取最后
    for m in range(len(data)):  # 获取行数
        sheet.write(m, 0, data[m])
    # excel.close()  # 关闭excel

    """⑥ episode_good_death 有用 """
    # excel = xlsxwriter.Workbook(r'D:\software\PyCharm\pyCharmProject\MADDPG\testPaper-MADDPG\1-4v8-红-MADDPG-VS-蓝MADDPG.xlsx')
    bold = excel.add_format({'bold': True})  # 设置excel样
    sheet = excel.add_worksheet('episode_good_death')
    # 向excel写数据    # sheet.write(3, 0, 32)
    rew_file_name = lujing + duikanghuanjing + file[episode_good_death]  # save_rate_sumreward
    data = pickle.load(open(rew_file_name, 'rb+'))  # [ episode/save_rate,1 ]
    for m in range(len(data)):  # 获取行数
        sheet.write(m, 0, data[m])
    # excel.close()  # 关闭excel

    """⑦ episode_adv_death 有用 """
    # excel = xlsxwriter.Workbook(r'D:\software\PyCharm\pyCharmProject\MADDPG\testPaper-MADDPG\1-4v8-红-MADDPG-VS-蓝MADDPG.xlsx')
    bold = excel.add_format({'bold': True})  # 设置excel样
    sheet = excel.add_worksheet('episode_adv_death')
    # 向excel写数据    # sheet.write(3, 0, 32)
    rew_file_name = lujing + duikanghuanjing + file[episode_adv_death]  # save_rate_sumreward
    data = pickle.load(open(rew_file_name, 'rb+'))  # [ episode/save_rate,1 ]
    for m in range(len(data)):  # 获取行数
            sheet.write(m, 0, data[m])
    excel.close()  # 关闭excel


    """                                       aa                          """

    # 创建xlsx后缀名的excel
    # excel = xlsxwriter.Workbook(r'D:\software\PyCharm\pyCharmProject\MADDPG\testPaper-MADDPG\xlswrite_test_aa.xlsx')
    # 添加sheet页名称
    """
    sheet = excel.add_worksheet('设备名称')
    # 设置行宽，行高
    sheet.set_column('A:F', 20)
    # 设置excel样
    bold = excel.add_format({'bold': True})
    # 向excel写数据
    # sheet.write(3, 0, 32)
    for i in range(len(aa)):
        rew_file_name = lujing + duikanghuanjing + file[aa[i]]
        data = pickle.load(open(rew_file_name, 'rb+'))  # [ 600000 10 6000 ]
        for m in range(len(data[0])):  # 获取列数
            for  k in range(len(data)):  # 获取行数
                sheet.write(k, m, data[k][m])
    # 关闭excel
    excel.close()
    """
    # 21212
    # """                                      bb                             """
    # 创建xlsx后缀名的excel

    # excel = xlsxwriter.Workbook(r'D:\software\PyCharm\pyCharmProject\MADDPG\testPaper-MADDPG\xlswrite_test_bb.xlsx')
    """
    # 添加sheet页名称
    sheet = excel.add_worksheet('设备名称')
    # 设置行宽，行高
    sheet.set_column('A:F', 20)
    # 设置excel样
    bold = excel.add_format({'bold': True})
    # 向excel写数据
    # sheet.write(3, 0, 32)
    for i in range(len(bb)):
        rew_file_name = lujing + duikanghuanjing + file[bb[i]]
        data = pickle.load(open(rew_file_name, 'rb+'))  # [ 600000 10 6000 ]
        for n in range(len(data[0])):  # 获取列数
            for k in range(len(data)):  # 获取行数
                sheet.write(k, n+i, data[k][n])  # 先是采样reward，最后是每回合reward
    # 关闭excel
    excel.close()
    """
    #  21212121
    """                                      cc                            """
    # 创建xlsx后缀名的excel
    # excel = xlsxwriter.Workbook(r'D:\software\PyCharm\pyCharmProject\MADDPG\testPaper-MADDPG\xlswrite_test_cc.xlsx')
    # # 添加sheet页名称
    # sheet = excel.add_worksheet('设备名称')
    # # 设置行宽，行高
    # sheet.set_column('A:F', 20)
    # # 设置excel样
    # bold = excel.add_format({'bold': True})
    # # 向excel写数据
    # # sheet.write(3, 0, 32)
    # for i in range(len(cc)):
    #     rew_file_name = lujing + duikanghuanjing + file[cc[i]]
    #     data = pickle.load(open(rew_file_name, 'rb+'))  #[ 600000 10 6000 ]
    #     for k in range(len(data)):
    #         sheet.write(k, i, data[k])
    # # 关闭excel
    # excel.close()
    """
    dd = [0]
    for i in aa:
        rew_file_name = lujing + duikanghuanjing + file[i]
        data = pickle.load(open(rew_file_name, 'rb+'))  # [600000 10 6000 ]
        for j in dd:  #
            datas = data[j] # [10 6000]
            for k in range(len(datas)):
                datasd = datas[k]  # [6000]
                sheet.write(k, 0, datasd)  # m
    """

