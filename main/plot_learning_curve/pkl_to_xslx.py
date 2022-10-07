import xlwt
from config.config import Config
import matplotlib.pyplot as plt
import numpy
import pickle
import xlsxwriter
import os

# custom read datas from .pkl to .xslx
def custom_read_data(datas):
    pass

# create a excel
if __name__ == '__main__':
    # get args
    arglist = Config()
    # setup xlsx
    workbook = xlwt.Workbook(encoding='ascii')
    #add sheet , the name of sheet is 'demo'
    worksheet = workbook.add_sheet('demo')
    # input datas
    data_file = os.path.join(os.getcwd(),arglist.data_file_dir)
    # create excle
    create_data_file  = os.path.join(os.getcwd(),arglist.create_xslx_dir)

    excel = xlsxwriter.Workbook(create_data_file)
    sheet = excel.add_worksheet('sheet1')
    bold = excel.add_format({'bold': True})
    # The data inputs excel

    data = pickle.load(open(data_file, 'rb+'))
    custom_read_data(data)  # importance function

    excel.close()

