# -*- coding = utf-8 -*-
# @Time : 2024/5/18 12:24
# @Author : ChiXiaoWai
# @File : handle.py
# @Project : code
# @Description : 异常处理
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import zscore
plt.switch_backend('agg')
matplotlib.use('TkAgg')

# 加载数据
file_path = '../data_set/ali20_c.csv'  # 替换为您的文件路径
data = pd.read_csv(file_path)

# 计算 'OT' 列的 Z 分数
data['avgcpu_ZScore'] = zscore(data['avgcpu'])

# 设置滚动均值的默认窗口大小
window_size = 5

# 计算滚动均值
rolling_mean = data['avgcpu'].rolling(window=window_size, center=True).mean()

# 用滚动均值替换异常值 (Z 分数 > 2)
data['avgcpu_RollingMean'] = data.apply(lambda row: rolling_mean[row.name] if abs(row['avgcpu_ZScore']) > 2 else row['avgcpu'], axis=1)

# 创建包含两个子图的图形
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# 在第一个子图中绘制原始数据
axes[0].plot(data['avgcpu'], label='Original Data', color='blue', alpha=0.7)
axes[0].set_title('原始 "avgcpu" 数据')
axes[0].set_xlabel('索引')
axes[0].set_ylabel('avgcpu 值')
axes[0].grid(True)

# 在第二个子图中绘制用滚动均值替换异常值后的数据
axes[1].plot(data['avgcpu_RollingMean'], label='Data After Rolling Mean Replacement', color='green', alpha=0.7)
axes[1].set_title('用滚动均值替换异常值后的数据')
axes[1].set_xlabel('索引')
axes[1].set_ylabel('avgcpu 值')
axes[1].grid(True)

# 调整布局并显示图表
plt.tight_layout()
plt.show()


