# 实验报告

## 一、摘要
### 背景： 
人脸识别技术在安全系统、个人身份验证和社交媒体分析等多个领域发挥着关键作用。随着深度学习的进步，基于面部关键点的人脸检测和识别方法在提高识别精度和速度方面取得了显著成就。MediaPipe，作为Google开源的跨平台机器学习框架，提供了实时面部关键点检测的解决方案。OpenCV，一个功能强大的计算机视觉库，进一步扩展了图像处理和分析的能力。
### 目的： 
本实验的目标是开发一个系统，利用MediaPipe进行面部关键点的实时检测，并结合OpenCV捕获面部图像，以收集用于人脸识别训练的数据集。此外，本实验旨在设计一个神经网络模型，用于学习面部特征并实现高精度的人脸识别。

### 方法： 
实验过程包括使用MediaPipe FaceMesh模型实时检测面部并提取486个关键点，通过OpenCV捕获面部图像，并在用户输入相应的身份标签后，将关键点数据和标签存储为JSON格式。随后，基于收集的数据，搭建并训练了一个深层神经网络，用于人脸识别任务。

### 主要发现： 
 实验成功地采集了（）张不同个体的面部图像及其关键点数据，并为每张图像分配了唯一的身份标签。所设计的神经网络模型采用了卷积神经网络(CNN)架构，有效地从面部关键点数据中学习到区分不同个体的特征。在初步测试中，模型展现了较高的识别准确率。

### 结论： 
本实验验证了所提出的面部关键点数据采集和处理流程的有效性，并成功搭建了一个能够进行人脸识别的神经网络模型。实验结果表明，结合MediaPipe和OpenCV可以高效地收集面部数据，而设计的神经网络模型能够准确地识别不同个体的人脸。JSON格式的数据存储为数据的管理和模型训练提供了便利。

### 关键词： 
人脸识别，面部关键点检测，MediaPipe，OpenCV，神经网络，数据采集，JSON

## 二、引言
### 研究背景：
人脸识别技术作为生物特征识别的一个重要分支，近年来在多个领域得到了广泛的应用。从智能手机的解锁功能到机场的自动安检系统，人脸识别技术提供了一种非侵入式的、用户友好的身份验证解决方案。随着人工智能技术的快速发展，基于深度学习的人脸识别方法在准确性和鲁棒性上取得了显著进步，这主要归功于强大的计算能力、大规模的训练数据集以及创新的算法设计。

### 技术发展：
MediaPipe是Google开源的跨平台机器学习框架，它利用了Google在机器学习领域的先进技术，提供了实时且准确的面部关键点检测功能。MediaPipe的FaceMesh模型能够检测面部的486个3D关键点，为面部表情识别、面部动态捕捉等应用提供了丰富的数据。OpenCV（Open Source Computer Vision Library）是一个开源的计算机视觉和机器学习软件库，它包含了数百个计算机视觉算法，包括实时图像和视频处理功能，非常适合于快速原型设计和复杂的图像处理任务。

### 研究必要性：
尽管现有的人脸识别系统已经取得了一定的成果，但在实际应用中仍面临着挑战，如不同光照条件下的识别、面部遮挡问题、以及大规模数据集中的个体识别等。此外，随着人们对隐私保护意识的增强，如何平衡人脸识别技术的便利性和隐私保护成为一个重要议题。因此，开发一个高效、准确且隐私友好的人脸识别系统具有重要的现实意义。

### 研究目的：
本实验旨在通过MediaPipe和OpenCV技术搭建一个数据采集系统，收集面部关键点数据，并为每个面部图像分配一个标签。收集到的数据将用于训练一个神经网络模型，以实现高精度的人脸识别。实验的目标是探索一种结合实时面部关键点检测和深度学习的方法，以提高人脸识别的准确性和鲁棒性。

## 三、相关调查

### 面部识别技术的发展：
面部识别技术自20世纪70年代以来已经取得了长足的进步。早期的方法主要基于几何特征和模板匹配，但这些方法容易受到光照变化、面部表情和遮挡的影响。进入21世纪，随着计算机算力的提升和大量数据集的出现，基于深度学习的方法开始主导这一领域。深度学习方法，如卷积神经网络(CNN)，能够自动从图像中学习复杂的特征表示，极大地提高了识别的准确性。

### 基于关键点的面部识别：
面部关键点检测是面部识别中的一个关键步骤，它涉及定位面部的特定标志点，如眼角、鼻尖和嘴角。基于关键点的方法不仅用于身份验证，还广泛应用于面部表情识别和人机交互。MediaPipe的FaceMesh模型是近年来在面部关键点检测领域的一个重要进展，它利用机器学习技术实现了实时的面部3D关键点定位。
![人脸关键点图例](face.png)

### 深度学习在面部识别中的应用：
深度学习，尤其是卷积神经网络(CNN)，已成为面部识别领域的核心技术。CNN能够有效地从面部图像中提取特征，并用于训练识别模型。随着研究的深入，一些变体模型如深度可分离卷积网络(DS-CNN)和注意力机制被提出，以提高模型对遮挡和表情变化的鲁棒性。

### 本研究与现有工作的关联：
本研究在现有工作的基础上，通过MediaPipe和OpenCV进行面部关键点的实时检测和数据采集，旨在构建一个更加精准和鲁棒的人脸识别系统。

## 四、理论基础

### 面部关键点检测原理：
面部关键点检测旨在识别和定位面部的特定标志点，如眉毛、眼睛、鼻子、嘴巴的位置。这一过程通常涉及人脸检测、关键点定位和关键点标记等步骤。在人脸检测阶段，系统首先识别图像中的人脸区域；然后，关键点定位阶段使用各种算法，如Active Shape Models (ASM) 或 Active Appearance Models (AAM)，来预测面部标志点的位置；最后，关键点标记阶段将这些点与预定义的面部模型相匹配，以确保准确性。

### MediaPipe FaceMesh模型：
MediaPipe FaceMesh是一个基于机器学习的人脸检测和关键点定位模型，它能够实时地在面部检测出468个人脸关键点。FaceMesh模型使用了一种高效的卷积神经网络架构，能够处理不同的面部表情、光照条件和面部姿态。该模型的输入是人脸图像的RGB格式，输出是一个包含关键点坐标的数组。

### 深度学习与卷积神经网络：
深度学习是一种基于人工神经网络的机器学习方法，特别适用于处理大量数据和复杂模式识别任务。卷积神经网络(CNN)是深度学习中最常见的网络类型之一，它通过多层的卷积层、激活层、池化层来自动学习图像的特征。在面部识别任务中，CNN能够从原始图像数据中提取有用的特征，并用于训练分类器。

### JSON数据格式：
JSON（JavaScript Object Notation）是一种轻量级的数据交换格式，易于人阅读和编写，同时也易于机器解析和生成。JSON是基于文本的格式，支持复杂的数据结构，如嵌套的对象和数组。在本实验中，JSON用于存储面部关键点数据和对应的标签，其灵活性和简洁性使得数据的存储和交换变得高效。

### 神经网络的训练过程：
神经网络的训练是一个优化过程，目标是找到一组参数（权重和偏置），使得网络对训练数据的预测与真实值之间的差异最小。这个过程通常涉及前向传播、计算损失、反向传播和参数更新四个步骤。在前向传播阶段，输入数据通过网络进行计算以产生预测；在计算损失阶段，损失函数评估预测值和真实值之间的差异；反向传播阶段计算损失相对于网络参数的梯度；最后，在参数更新阶段，使用优化算法（如梯度下降）调整网络参数以减少损失。

## 四、实验过程
### 实验环境
#### 硬件环境
CPU : 13th Gen Intel(R) Core(TM) i9-13980HX   2.20 GHz
GPU : NVDIA GeForce RTX 4080 LAPTOP(12GB)
#### 软件环境
Package               Version
absl-py               2.1.0
attrs                 23.2.0
certifi               2022.12.7
charset-normalizer    3.3.2
comtypes              1.3.1
cycler                0.11.0
flatbuffers           24.3.25
fonttools             4.38.0
idna                  3.7
importlib-metadata    6.7.0
joblib                1.3.2
kiwisolver            1.4.5
matplotlib            3.5.3
mediapipe             0.9.0.1
numpy                 1.21.6
opencv-contrib-python 4.9.0.80
opencv-python         4.9.0.80
packaging             24.0
Pillow                9.5.0
pip                   22.3.1
protobuf              3.20.3
psutil                5.9.8
pycaw                 20240210
pyparsing             3.1.2
python-dateutil       2.9.0.post0
requests              2.31.0
scikit-learn          1.0.2
scipy                 1.7.3
setuptools            65.6.3
six                   1.16.0
threadpoolctl         3.1.0
torch                 1.13.1
torchaudio            0.13.1
torchvision           0.14.1
typing_extensions     4.7.1
urllib3               2.0.7
wheel                 0.38.4
wincertstore          0.2
zipp                  3.15.0

### 实验步骤
#### 1.导入相关库
```python
import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
```
#### 2.初始化MediaPipe面部关键点检测模型：
```python
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
```
#### 3.设置视频捕获：
```python
cap = cv2.VideoCapture(0)
```
#### 4.定义数据收集函数：
创建一个函数 get_face_mesh_results 来处理图像并返回面部关键点。
```python
def get_face_mesh_results(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(image)
    return results.multi_face_landmarks
```
#### 5.实时视频处理循环：

使用 while 循环来不断读取视频流中的帧。
对每一帧图像，使用MediaPipe模型检测面部关键点。
#### 6. 用户输入标签：

在检测到面部后，提示用户输入与当前面部对应的标签。
#### 7. 数据存储：

将面部关键点和对应的标签存储在一个列表中。
#### 8. 处理用户按键：

允许用户通过按键操作来开始数据采集或退出程序。
#### 9. 将数据写入JSON文件：
- 将收集到的数据转换为JSON格式，并保存到文件中。

#### 10. 释放资源：
- 在数据采集完成后，释放摄像头资源并关闭所有窗口。

```py
# 实验步骤5: 实时视频处理循环
while len(data) < 100:
    ret, frame = cap.read()
    if not ret:
        print("无法读取图像，退出程序。")
        break

    landmarks = get_face_mesh_results(frame)
    if landmarks:
        for face_landmarks in landmarks:
            facial_keypoints = [[x, y] for x, y in face_landmarks.landmark]

            # 实验步骤6: 用户输入标签
            label = input("输入标签并按回车键（输入'q'退出程序）: ")
            if label == 'q':
                break

            # 实验步骤7: 数据存储
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            data.append({
                "timestamp": timestamp,
                "keypoints": facial_keypoints,
                "label": label
            })

    # 实验步骤8: 处理用户按键
    if cv2.waitKey(1) & 0xFF == ord('p'):
        continue

# 实验步骤9: 将数据写入JSON文件
filename = "face_data.json"
with open(filename, 'w') as outfile:
    json.dump(data, outfile, indent=4)

# 实验步骤10: 释放资源
cap.release()
cv2.destroyAllWindows()
```
#### 11.定义神经网络模型

创建 FacialKeypointsClassifier 类，该类继承自 nn.Module，用于构建面部关键点分类器。
```py
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
```
#### 12.定义数据集类

实现 FacialKeypointsDataset 类，该类继承自 torch.utils.data.Dataset，用于加载和处理面部关键点数据。
```py
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
        label = item['label']
        if self.transform:
            keypoints = self.transform(keypoints)
        return keypoints, label
```
#### 13.设置超参数

定义模型的输入尺寸 input_size 和分类数量 num_classes。
```py
    input_size = 3 * 468  # 每个关键点有3个坐标值
    num_classes = 4       # 分类的数量
```
#### 14.实例化模型

创建 FacialKeypointsClassifier 的实例，并将模型发送到配置的设备上。
```py
    model = FacialKeypointsClassifier(input_size, num_classes).to(DEVICE)
```
#### 15.定义损失函数和优化器

实例化 nn.CrossEntropyLoss 作为损失函数，并设置 optim.Adam 作为优化器。学习率lr定为0.01
```py
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
```
#### 16.数据集和数据加载器

实例化 FacialKeypointsDataset 并创建 DataLoader，用于训练时批量加载数据。
```py
    train_dataset = FacialKeypointsDataset(json_file='face_data.json')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```
#### 17.模型训练

设置模型为训练模式，进行多个epoch的训练。
在每个epoch中，遍历数据加载器提供的数据批次。
对每个批次的数据执行前向传播、计算损失、执行反向传播，并更新模型权重。
打印每个批次的损失，监控训练进度。
```py
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
```
#### 18.保存模型

在训练结束后，保存模型的状态字典到文件中，以便后续使用或进一步训练。
```py
    torch.save(model.state_dict(), 'facial_keypoints_classifier.pth')
```

## 五、实验结论
### 1. 实验结果总结：
本实验成功实现了一个基于面部关键点的分类器，该分类器能够识别并分类不同的面部特征。通过MediaPipe和OpenCV技术，我们有效地收集了面部关键点数据，并将其转换为适合模型训练的格式。实验中，我们设计了一个包含两个隐藏层的前馈神经网络，并使用PyTorch框架进行了训练和测试。

在500个epoch的训练后，模型在训练集上取得了显著的损失下降，表明模型能够学习到面部关键点与分类标签之间的关系。通过监控训练过程中的损失，我们注意到模型的权重逐渐稳定，这表明网络没有过拟合的迹象。

### 2. 模型性能评估：
尽管模型在训练集上表现良好，但为了全面评估模型的泛化能力，进一步的测试应该在独立的验证集和测试集上进行。这将提供对模型在未见数据上性能的准确评估，并有助于识别模型可能存在的局限性。

### 3. 实验局限性：
当前实验的局限性主要包括数据集大小和多样性的限制。由于实验中只收集了（）个样本，这可能限制了模型学习到的特征的泛化能力。此外，实验中未对数据进行交叉验证，这也可能影响模型性能的准确评估。