# -*- coding: utf-8 -*-
# 指定文件编码为 UTF-8，保证中文注释在不同环境下不会出现编码错误。

import torch                                  # 导入 PyTorch 主库（用于张量计算、自动微分、模型保存等）
import torch.nn as nn                         # 导入神经网络模块，包含常用的层（Conv, Linear, ...）
import torch.utils.data as Data               # 导入数据工具，包含 DataLoader 等
import torchvision                            # PyTorch 的视觉工具包，内含常用数据集和工具函数
import torchvision.transforms as transforms   # 图像预处理/增强工具（将 PIL 图像或 ndarray 转为 Tensor 等）
import matplotlib.pyplot as plt               # 绘图库，用来可视化图像与结果
import numpy as np                            # 数值处理库，主要用于数组和数学运算（这里用得少）
import cv2                                    # OpenCV，用于更复杂的图像处理（本脚本导入但未深度使用）
import os                                     # 操作系统接口，用于检测文件是否存在等
from tqdm import tqdm                         # 进度条工具，能在训练循环中显示进度（更友好）

# =====================================================
# 1. 设备检查（GPU/CPU）
# =====================================================
# 先检查当前环境是否支持 GPU（CUDA），如果支持优先使用 GPU 以加速训练，否则使用 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.is_available() 返回 True/False；torch.device 用于标记后续操作的运行设备。
print(f"当前训练设备: {device}")  # 打印当前选择的设备，便于调试与日志记录。

if device.type == "cuda":
    # 如果检测到 CUDA 可用，打印 GPU 型号和显存大小，帮助了解硬件资源。
    print("GPU 型号:", torch.cuda.get_device_name(0))
    # get_device_properties(0).total_memory 返回字节数，这里换算成 GB 并四舍五入两位小数显示。
    print("显存总量:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")


# =====================================================
# 2. 超参数
# =====================================================
# 将训练时常用的参数定义在顶部，便于后续修改与试验。
EPOCHS = 8         # 训练轮数。每个 epoch 表示把训练集完整送入模型一次。
BATCH_SIZE = 64    # 批量大小。一次送入网络训练的样本数，会影响显存占用和训练稳定性。
LR = 0.001         # 学习率（learning rate），决定参数更新步幅大小，过大或过小都不合适。


# =====================================================
# 3. 加载 MNIST 数据集
# =====================================================
# 使用 torchvision 自带的 MNIST 数据集。MNIST 是 28x28 灰度手写数字数据集，共 10 类（0-9）。
transform = transforms.ToTensor()
# transforms.ToTensor() 会把 PIL Image 或 numpy.ndarray 转成形状为 (C, H, W) 的 float Tensor，
# 并把像素值从 [0,255] 归一化到 [0.0, 1.0]（除以 255）。

train_data = torchvision.datasets.MNIST(
    root='./data/',      # 数据保存的根目录（如果不存在，会自动创建）
    train=True,          # 指明是训练集
    transform=transform, # 应用到数据上的预处理/变换
    download=True        # 如果本地没有，会自动从网上下载数据集
)

test_data = torchvision.datasets.MNIST(
    root='./data/',      # 同上，测试集也存放在 data/ 下
    train=False,         # 指明是测试集（非训练）
    transform=transform  # 同样做 ToTensor 预处理
)

# DataLoader 用于按批次取出数据，并可打乱顺序（shuffle）
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 训练时通常要 shuffle=True，避免模型看到数据的固定顺序导致偏差或过拟合。
test_loader = Data.DataLoader(dataset=test_data, batch_size=1000, shuffle=False)
# 测试时不需要打乱，batch_size 可以设大一些以加速评估（但受显存/内存限制）


# =====================================================
# 4. 定义 CNN 模型
# =====================================================
# 下面定义一个非常简单的卷积神经网络（CNN）。适合 MNIST 这种小图像任务。
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 初始化父类，必须调用
        # 第一层卷积：输入通道 1（灰度图），输出通道 16，卷积核大小 5，padding=2 保持尺寸不变
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),  # 卷积层：输出 16 个特征图
            nn.ReLU(),                                   # 非线性激活函数 ReLU
            nn.MaxPool2d(2)                              # 池化层：2x2 最大池化，图像尺寸变为原来的一半
        )

        # 第二层卷积：输入 16 通道，输出 32 通道，kernel_size=5，padding=2 保持尺寸，之后再池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2), # 卷积后输出 32 个特征图
            nn.ReLU(),
            nn.MaxPool2d(2)                              # 再次池化，尺寸再次减半
        )

        # 全连接层（分类层）
        # 经过两次池化后，28x28 -> first pool -> 14x14 -> second pool -> 7x7
        # 特征图数量为 32，所以展平后特征数量是 32 * 7 * 7
        self.fc = nn.Linear(32 * 7 * 7, 10)  # 输出 10 个神经元，对应 10 个数字类别（0-9）

    def forward(self, x):
        # 定义前向传播过程（即数据如何通过网络）
        x = self.conv1(x)                      # 输入经过 conv1（卷积->ReLU->池化）
        x = self.conv2(x)                      # 再经过 conv2（卷积->ReLU->池化）
        x = x.view(x.size(0), -1)              # 把多维张量展平为 (batch_size, features) 以送入全连接层
        out = self.fc(x)                       # 全连接层得到原始 logits（未归一化的类别分数）
        return out                             # 返回 logits（后面会和 CrossEntropyLoss 一起使用）


model = CNN().to(device)  # 实例化模型并把它移动到选定的设备（GPU 或 CPU）
print(model)              # 打印模型结构，查看各层、参数数量和输出形状信息


# =====================================================
# 5. 定义训练与测试函数
# =====================================================
loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类（内部会做 softmax）
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# Adam 优化器，自动调整每个参数的学习率，是常用且表现稳定的优化器之一。

def train(model, device, loader, optimizer):
    """
    训练函数：对一个 epoch 的训练集做一次完整迭代并返回平均损失
    参数:
        model: 网络模型
        device: 计算设备（'cuda' 或 'cpu'）
        loader: DataLoader（训练集）
        optimizer: 优化器（例如 Adam）
    返回:
        每个 batch 平均 loss（用于监控训练过程）
    """
    model.train()             # 设置模型为训练模式，会启用 dropout、batchnorm 的训练行为（如果有的话）
    total_loss = 0            # 记录总损失以便计算平均损失

    # 使用 tqdm 包装 loader 可以显示进度条（更友好）
    for x, y in tqdm(loader, desc="训练中", leave=False):
        # x 是一个 batch 的图片张量，形状 [batch_size, 1, 28, 28]
        # y 是对应的标签张量，形状 [batch_size]
        x, y = x.to(device), y.to(device)   # 把数据移动到指定设备（GPU/CPU）

        output = model(x)                   # 前向传播，得到 logits（未经过 softmax 的分数）
        loss = loss_func(output, y)         # 计算当前 batch 的损失

        optimizer.zero_grad()               # 清空上一轮的梯度（非常重要），否则梯度会累加
        loss.backward()                     # 反向传播，计算当前损失对参数的梯度
        optimizer.step()                    # 优化器更新参数（基于计算得到的梯度）

        total_loss += loss.item()           # loss.item() 将 tensor 转为 Python 数值，累加到 total_loss

    return total_loss / len(loader)         # 返回平均损失（总损失 / batch 数）


def test(model, device, loader):
    """
    测试/验证函数：计算在给定数据集上的准确率
    参数:
        model: 网络模型
        device: 计算设备
        loader: DataLoader（测试集或验证集）
    返回:
        准确率（百分比）
    """
    model.eval()              # 设置模型为评估模式，停用 dropout、用训练好的均值方差（如果有 batchnorm）
    correct = 0               # 用于累计正确预测的样本数
    total = 0                 # 累计样本总数

    # 在评估阶段不需要计算梯度，可以节省显存和计算（torch.no_grad）
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)  # 把数据移到同一设备
            output = model(x)                  # 前向传播
            pred = output.argmax(dim=1)        # 取 logits 最大值对应的索引作为预测类别
            correct += (pred == y).sum().item()# 累计预测正确的数量（布尔数组求和）
            total += y.size(0)                 # 累计图片数量

    acc = 100. * correct / total               # 计算准确率并转为百分比
    return acc


# =====================================================
# 6. 是否已经有模型？若无就训练
# =====================================================
model_path = "mnist_cnn.pth"  # 模型文件名（常用 .pth 或 .pt 扩展名）

if os.path.exists(model_path):
    # 如果模型文件存在，就加载它，而不是重新训练（节省时间）
    print("\n发现已保存的模型，正在加载...")
    # map_location=device：确保模型参数加载到当前选定设备（如 CPU 或 GPU）
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    # 如果找不到已保存模型，就开始训练
    print("\n未找到模型，开始训练...\n")
    for epoch in range(1, EPOCHS + 1):
        # 每个 epoch 调用 train 函数训练一个完整的训练集
        train_loss = train(model, device, train_loader, optimizer)
        # 每个 epoch 后用测试集评估当前模型的准确率
        test_acc = test(model, device, test_loader)
        # 打印本轮训练结果，便于观察训练过程（损失是否下降，准确率是否上升）
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # 训练完成后把模型参数保存到文件，方便下次直接加载使用
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存为 {model_path}")


# =====================================================
# 7. 使用模型进行预测（前 64 个样本）
# =====================================================
model.eval()  # 进入评估模式以保证预测稳定（比如如果有 dropout）

# 从 test_data 中取前 64 张图像作为示例：
# test_data.data 的形状通常为 [N, 28, 28]（整数 0-255），需要转成 [N, 1, 28, 28] 并归一化到 [0,1]
samples = test_data.data[:64].unsqueeze(1).float() / 255.0
# .unsqueeze(1) 在第 1 个维度插入一个通道维（灰度图的通道数为 1）
# .float() 将数据类型变为浮点数；/255.0 进行归一化

labels = test_data.targets[:64]  # 对应的真实标签（0-9）

samples = samples.to(device)     # 把样本移动到 device（否则模型和数据处于不同设备会报错）
with torch.no_grad():            # 预测阶段不需要梯度
    output = model(samples)      # 前向传播得到 logits
pred_y = output.argmax(dim=1).cpu().numpy()  # 取每行最大值索引作为预测类别，转成 numpy 方便打印
labels = labels.numpy()                     # 把真实标签也转为 numpy（之前是 tensor）

print("\n预测结果:", pred_y)  # 打印模型预测的 64 个数字
print("真实标签:", labels)   # 打印对应的真实标签


# =====================================================
# 8. 可视化
# =====================================================
# torchvision.utils.make_grid 可以把多个图片拼接成一张大图，便于一次显示多张图像
img = torchvision.utils.make_grid(samples.cpu(), nrow=8).numpy().transpose(1, 2, 0)
# 说明：
# - samples.cpu()：把张量转到 CPU（matplotlib 只能读 CPU 上的 numpy 数组）
# - make_grid(..., nrow=8)：按每行 8 张图拼接（这里 64 张 -> 8 行 x 8 列）
# - 结果是 shape (C, H, W)，用 numpy() 转为 ndarray 后用 transpose 调整为 (H, W, C)
# - MNIST 是单通道，转成 (H, W, C) 时 C=1，但 matplotlib 在显示时会期望 (H, W) 或 (H, W, 3)。
#   这里我们仍然把它作为灰度图用 cmap='gray' 来绘制。

plt.figure(figsize=(10, 6))        # 新建一个画布并设定大小（英寸）
plt.imshow(img, cmap='gray')       # 显示图像，cmap='gray' 指定为灰度色图
plt.title("MNIST Predictions")     # 设置标题
plt.axis("off")                    # 关闭坐标轴显示，使图片更整洁

# 在拼接后的图上写入预测数字（绿色=正确，红色=错误）
# 注意：每张 MNIST 小图的大小是 28x28，make_grid 使用保持这个尺寸进行拼接
for i in range(64):
    # 计算第 i 张图片在大图中的文本位置：
    # (i % 8) 代表列索引（0..7），(i // 8) 代表行索引
    # 乘以 28 转换为像素坐标，再加上偏移放置数字
    plt.text((i % 8) * 28 + 10, (i // 8) * 28 + 2,
             str(pred_y[i]),  # 要写的文本（预测数字）
             color="green" if pred_y[i] == labels[i] else "red",  # 对比真实标签，正确为绿，错误为红
             fontsize=12, fontweight="bold")  # 设置字体大小和加粗

plt.show()  # 显示图像窗口（在 Jupyter 中会 inline 显示，在脚本运行时会弹出窗口）
