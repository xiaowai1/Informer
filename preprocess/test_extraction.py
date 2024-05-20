# -*- coding = utf-8 -*-
# @Time : 2024/5/18 18:57
# @Author : ChiXiaoWai
# @File : test_extraction.py
# @Project : code
# @Description : 测试集提取

import pandas as pd

# 读取原始 CSV 文件
file_path = '../preprocess_data/processed_ali20_c.csv'
data = pd.read_csv(file_path)

# 计算后 30% 数据的起始索引
start_index = int(len(data) * 0.8)

# 截取后 30% 的数据
last_30_percent_data = data[start_index:]

# 保存截取的数据到新的 CSV 文件
output_file_path = '../preprocess_data/ali20_c_test.csv'
last_30_percent_data.to_csv(output_file_path, index=False)

print(f"后 30% 的数据已保存到: {output_file_path}")
