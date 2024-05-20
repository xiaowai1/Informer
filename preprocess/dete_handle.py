# -*- coding = utf-8 -*-
# @Time : 2024/5/18 12:31
# @Author : ChiXiaoWai
# @File : dete_handle.py
# @Project : code
# @Description : 异常检测和异常处理
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
plt.switch_backend('agg')
matplotlib.use('TkAgg')

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 加载数据
data_path = '../data_set/ali20_c.csv'  # 替换为您的文件路径
new_data = pd.read_csv(data_path)
new_data['avgcpu'] = np.minimum(new_data['avgcpu'], 1)
new_data['avgmem'] = np.minimum(new_data['avgmem'], 1)
new_data['avgcpu'] = np.abs(new_data['avgcpu'])
new_data['avgmem'] = np.abs(new_data['avgmem'])
# 准备数据
data_tensor = torch.tensor(new_data[['avgcpu', 'avgmem']].values.astype(np.float32))
dataset = TensorDataset(data_tensor, data_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 初始化模型、损失函数和优化器
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    for data in dataloader:
        inputs, _ = data
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 评估模型
model.eval()
with torch.no_grad():
    predictions = model(data_tensor)
    mse = nn.functional.mse_loss(predictions, data_tensor, reduction='none')
    mse = mse.mean(dim=1).numpy()

new_data['Reconstruction_Error'] = mse


# 定义重建误差的阈值为离群点
threshold = np.percentile(new_data['Reconstruction_Error'], 95)
new_data['Outlier_Autoencoder'] = new_data['Reconstruction_Error'] > threshold

# 计算 avgcpu 和 avgmem 列的 Z 分数
new_data['avgcpu_ZScore'] = zscore(new_data['avgcpu'])
new_data['avgmem_ZScore'] = zscore(new_data['avgmem'])

# 设置滚动均值的默认窗口大小
window_size = 5

# 计算滚动均值
rolling_mean_cpu = new_data['avgcpu'].rolling(window=window_size, center=True).mean()
rolling_mean_mem = new_data['avgmem'].rolling(window=window_size, center=True).mean()

# 用滚动均值替换 avgcpu 和 avgmem 列的异常值
new_data['avgcpu_RollingMean'] = new_data.apply(lambda row: rolling_mean_cpu[row.name] if row['Outlier_Autoencoder'] else row['avgcpu'], axis=1)
new_data['avgmem_RollingMean'] = new_data.apply(lambda row: rolling_mean_mem[row.name] if row['Outlier_Autoencoder'] else row['avgmem'], axis=1)

# 将处理后的数据保存到 CSV 文件，只保留日期、处理后的 avgmem 和 avgcpu 列，并将列名改为 avgmem 和 avgcpu
processed_data_path = '../preprocess_data/processed_ali20_c.csv'
new_data[['date', 'avgmem_RollingMean', 'avgcpu_RollingMean']].rename(columns={'avgmem_RollingMean': 'avgmem', 'avgcpu_RollingMean': 'avgcpu'}).to_csv(processed_data_path, index=False)
print("处理后的数据已保存到:", processed_data_path)


# 使用seaborn样式
sns.set(style="whitegrid")

# 创建包含两个子图的图形
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

# 在第一个子图中绘制原始 avgcpu 数据
axes[0, 0].plot(new_data['avgcpu'], label='Original avgcpu', color='blue', alpha=0.7)
axes[0, 0].set_title('原始 "avgcpu" 数据')
axes[0, 0].set_xlabel('索引')
axes[0, 0].set_ylabel('值')
axes[0, 0].grid(True)
axes[0, 0].legend()

# 在第二个子图中绘制用滚动均值替换异常值后的 avgcpu 数据
axes[0, 1].plot(new_data['avgcpu_RollingMean'], label='avgcpu After Rolling Mean Replacement', color='blue', alpha=0.7)
axes[0, 1].set_title('用滚动均值替换异常值后的 "avgcpu" 数据')
axes[0, 1].set_xlabel('索引')
axes[0, 1].set_ylabel('值')
axes[0, 1].grid(True)
axes[0, 1].legend()

# 在第三个子图中绘制原始 avgmem 数据
axes[1, 0].plot(new_data['avgmem'], label='Original avgmem', color='green', alpha=0.7)
axes[1, 0].set_title('原始 "avgmem" 数据')
axes[1, 0].set_xlabel('索引')
axes[1, 0].set_ylabel('值')
axes[1, 0].grid(True)
axes[1, 0].legend()

# 在第四个子图中绘制用滚动均值替换异常值后的 avgmem 数据
axes[1, 1].plot(new_data['avgmem_RollingMean'], label='avgmem After Rolling Mean Replacement', color='green', alpha=0.7)
axes[1, 1].set_title('用滚动均值替换异常值后的 "avgmem" 数据')
axes[1, 1].set_xlabel('索引')
axes[1, 1].set_ylabel('值')
axes[1, 1].grid(True)
axes[1, 1].legend()

# 调整布局并显示图表
plt.tight_layout()
plt.show()