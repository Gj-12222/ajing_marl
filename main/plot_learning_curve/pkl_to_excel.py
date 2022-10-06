import xlwt
from config.Config import Config
import matplotlib.pyplot as plt
import numpy
import pickle
import xlsxwriter
import os

r'''Only support to export the data(.pkl) of default env(uav_env) of library '''
# need you to custom 'sheet_name'
if __name__ == '__main__':
    arglist = Config()

    sheet_name = ''  # custom your name from data type
    data_file = os.path.join(os.getcwd(), arglist.data_file_dir)
    create_xslx_dir = os.path.join(os.getcwd(), arglist.create_xslx_dir)

    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet('demo')

    rewards =  arglist.save_name + '_rewards.pkl'
    agloss = arglist.save_name + '_agloss.pkl'
    agreward = arglist.save_name + '_agrewards.pkl'
    everagrew = arglist.save_name + '_everyep_agrew.pkl'
    everallrew = arglist.save_name + '_everyep_allrew.pkl'
    good_death = arglist.save_name + '_good_death.pkl'
    adv_death = arglist.save_name + '_adv_death.pkl'
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
    excel = xlsxwriter.Workbook(create_xslx_dir)
    sheet = excel.add_worksheet(sheet_name)
    bold = excel.add_format({'bold': True})

    data = pickle.load(open(data_file, 'rb+'))  # [ episode,N, loss_num ]
    for m in range(len(data[0])):  # 获取列数
        for k in range(len(data)):  # 获取行数
            sheet.write(k, m, data[k][m])
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

