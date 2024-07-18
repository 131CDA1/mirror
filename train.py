import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


# 定义一个简单的神经网络
class FacialKeypointsClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FacialKeypointsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 1200)  # 第一个全连接层
        self.fc2 = nn.Linear(1200, 600) # 第二个全连接层
        self.fc3 = nn.Linear(600, num_classes) # 输出层

    def forward(self, x):
        # 将三维关键点数据转换为一维数据
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))  # 通过第一个全连接层并应用ReLU激活函数
        x = F.relu(self.fc2(x))  # 通过第二个全连接层并应用ReLU激活函数
        x = self.fc3(x)          # 通过输出层得到分类结果
        return x

class FacialKeypointsDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        keypoints = np.array(item['keypoints']).astype(np.float32)  # 将关键点转换为numpy数组
        label = item['label']  # 假设标签已经是适合模型输入的格式
        if self.transform:
            keypoints = self.transform(keypoints)
        return keypoints, label
if __name__ == '__main__':
    # 定义一些超参数
    input_size = 3 * 468  # 假设每个关键点有3个坐标值
    num_classes = 4       # 分类的数量

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模型
    model = FacialKeypointsClassifier(input_size, num_classes).to(DEVICE)

    # 显示模型结构
    # print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = FacialKeypointsDataset(json_file='face_data.json')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 训练模型

    model.train()  # 设置模型为训练模式
    num_epochs = 500
    for epoch in range(num_epochs):
        # print(f'Starting Epoch {epoch + 1}/{num_epochs}')
        for i, (inputs, *_, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # 清除之前的梯度
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            print(loss.item())
            if (i+1) % 10 == 0:
                print(f'  Batch {i + 1}/{len(train_loader)}, Loss: {loss.item()}')
        # print(f'Finished Epoch {epoch + 1}/{num_epochs}')
    torch.save(model.state_dict(), 'facial_keypoints_classifier.pth')

from torchviz import make_dot
from graphviz import Source

# 创建一个假的输入张量
fake_input = torch.randn(1, 3 * 468)

# 使用模型的 forward 方法来获取输出，并生成计算图
dot_data = make_dot(model(fake_input), params=dict(model.named_parameters()))

# 将 make_dot 生成的计算图转换为 Graphviz 源码字符串
dot_source = dot_data.source

# 直接在字符串中设置横向布局的属性
dot_source = dot_source.replace('digraph G {', 'digraph G {\n  rankdir=LR;\n')

# 使用 graphviz 的 Source 类来创建图
graph = Source(dot_source, format='png')

# 将计算图渲染为图像
graph.render('facial_keypoints_classifier_horizontal', cleanup=True)

# 显示图像
graph.view()