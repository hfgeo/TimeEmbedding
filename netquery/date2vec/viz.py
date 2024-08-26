# 安装 torchviz


import torch
import torchviz
from Date2VecModel import Date2Vec  # 假设您的模型定义在名为 Date2Vec.py 的文件中

# 创建模型
model = Date2Vec()

# 创建一个虚拟输入
dummy_input = torch.randn(1, 6)  # 输入维度是 (1, 6)

# 可视化模型结构
dot = torchviz.make_dot(model(dummy_input), params=dict(model.named_parameters()))

dot.render("model",format='svg')

