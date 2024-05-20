# -*- coding = utf-8 -*-
# @Time : 2024/5/18 12:08
# @Author : ChiXiaoWai
# @File : detection.py
# @Project : code
# @Description : 异常检测
import torch
import matplotlib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.switch_backend('agg')
matplotlib.use('TkAgg')

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 加载数据
data_path = '../data_set/ali20_c.csv'  # 替换为您的文件路径
new_data = pd.read_csv(data_path)

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
# 检查数据是否超出 [0, 1] 范围
out_of_range = (new_data['avgcpu'] < 0) | (new_data['avgcpu'] > 1) | (new_data['avgmem'] < 0) | (new_data['avgmem'] > 1)
new_data['Outlier_OutOfRange'] = out_of_range
new_data['Outlier_Final'] = new_data['Outlier_Autoencoder'] | new_data['Outlier_OutOfRange']

# 使用seaborn样式
sns.set(style="whitegrid")

# 创建折线图，并在其中标注离群点
plt.figure(figsize=(12, 6))

# 绘制avgcpu值的折线图
plt.plot(new_data['avgcpu'], label='avgcpu', color='blue')

# 标注avgcpu离群点
outliers = new_data[new_data['Outlier_Final']]
plt.scatter(outliers.index, outliers['avgcpu'], color='red', label='Outliers in avgcpu (Autoencoder)')

# 绘制avgmem值的折线图
plt.plot(new_data['avgmem'], label='avgmem', color='green')

# 标注avgmem离群点
plt.scatter(outliers.index, outliers['avgmem'], color='orange', label='Outliers in avgmem (Autoencoder)')

# 设置标题和标签
plt.title('avgcpu and avgmem with Autoencoder Outliers', fontsize=15, fontweight='bold')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()

# 显示图表
plt.show()