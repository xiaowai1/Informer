# -*- coding = utf-8 -*-
# @Time : 2024/5/17 22:19
# @Author : ChiXiaoWai
# @File : date_analysis.py
# @Project : code
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
matplotlib.use('TkAgg')


data = pd.read_csv("../data_set/gc11.csv", index_col=['date'], parse_dates=['date'])
data.index = pd.to_datetime(data.index, unit='us')
# 可视化原始数据
data.plot()
plt.title('original data')
plt.show()