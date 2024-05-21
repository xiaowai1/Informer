# -*- coding = utf-8 -*-
# @Time : 2024/5/20 15:07
# @Author : ChiXiaoWai
# @File : data_cleaning.py
# @Project : code
import pandas as pd

# 加载数据
data_path = r'D:\paper\dataset\c_1.csv'  # 替换为你的 CSV 文件路径
df = pd.read_csv(data_path)

# 设置起始时间为2018年1月1日
start_date = pd.Timestamp('2018-01-01')

# 将时间戳转换为日期时间格式
df['date'] = start_date + pd.to_timedelta(df['date'], unit='s')
df.set_index('date', inplace=True)

# 计算时间间隔（秒数）
df['time_diff'] = df.index.to_series().diff().dt.total_seconds()

# 描述时间间隔的统计信息
print(df['time_diff'].describe())

# # 选择一个合适的时间间隔，比如使用中位数或众数
# mode_interval = df['time_diff'].mode()[0]  # 使用众数作为时间间隔
# print(f'Mode time interval: {mode_interval} seconds')

# 选择一个合适的时间间隔，比如使用中位数或众数
median_interval = df['time_diff'].median()
print(f'Median time interval: {median_interval} seconds')

# 将时间间隔转换为 Pandas 频率字符串
def seconds_to_freq(seconds):
    if seconds < 60:
        return f'{int(seconds)}S'
    elif seconds < 3600:
        return f'{int(seconds / 60)}T'
    elif seconds < 86400:
        return f'{int(seconds / 3600)}H'
    else:
        return f'{int(seconds / 86400)}D'

# 获取合适的频率字符串
resample_interval = seconds_to_freq(median_interval)
print(f'Selected resample interval: {resample_interval}')

# 如果有缺失值，可以选择填充或删除
df.dropna(inplace=True)  # 向前填充缺失值

# 确保所有列都是数值类型
df = df.apply(pd.to_numeric, errors='coerce')
# 重新采样数据，使用线性插值填充缺失值
df_resampled = df.resample(resample_interval).mean().interpolate(method='linear')

# 打印重新采样后的数据
print(df_resampled.head())

# 将处理后的数据保存回 CSV 文件（可选）
df_resampled.to_csv('cleaned_data.csv')