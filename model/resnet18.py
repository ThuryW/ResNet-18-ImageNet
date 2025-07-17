import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# --- 1. ResNet18 模型定义 ---
# PyTorch 已经内置了 ResNet 系列模型，我们可以直接使用，这大大简化了代码。
# 为了保持完整性，这里也展示了如何从头定义一个简化的残差块，但实际使用时通常直接用torchvision.models

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18(num_classes=1000, pretrained=False):
    """
    ResNet-18 model definition.
    Args:
        num_classes (int): Number of output classes (1000 for ImageNet).
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
    """
    if pretrained:
        # 推荐直接使用 torchvision 内置的预训练模型
        print("Loading pre-trained ResNet18 model.")
        model = models.resnet18(pretrained=True)
        # 如果需要适应不同的类别数，需要修改最后一层
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        print("Initializing ResNet18 model from scratch.")
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# --- 2. 数据加载器 ---
def get_imagenet_dataloaders(data_dir, batch_size=64, num_workers=8):
    """
    Creates data loaders for ImageNet.
    Args:
        data_dir (str): Path to the ImageNet dataset root directory.
        batch_size (int): Batch size for training and validation.
        num_workers (int): Number of worker processes for data loading.
    Returns:
        train_loader, val_loader
    """
    # ImageNet 标准的预处理
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224), # 随机裁剪到224x224
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),       # 先缩放到256
        transforms.CenterCrop(224),   # 然后中心裁剪到224
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    val_dataset = datasets.ImageFolder(val_dir, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Found {len(train_dataset)} training images.")
    print(f"Found {len(val_dataset)} validation images.")

    return train_loader, val_loader

# --- 3. 训练和评估函数 ---
def train_model(model, train_loader, criterion, optimizer, device, epoch):
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 使用 tqdm 显示进度条
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # 清空梯度

        outputs = model(inputs)
        loss = criterion(outputs, labels) # 计算损失

        loss.backward() # 反向传播
        optimizer.step() # 更新权重

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        progress_bar.set_postfix(loss=loss.item(), acc=100. * correct_predictions / total_samples)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def evaluate_model(model, val_loader, criterion, device):
    model.eval() # 设置模型为评估模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # 禁用梯度计算
        progress_bar = tqdm(val_loader, desc="Validation")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=loss.item(), acc=100. * correct_predictions / total_samples)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    print(f"Validation Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

# --- 主程序入口 ---
if __name__ == '__main__':
    # 配置参数
    data_directory = '/path/to/your/imagenet' # !! 更改为你的 ImageNet 路径
    num_classes = 1000 # ImageNet 的类别数是 1000
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    num_workers = 8 # 根据你的 CPU 核数和内存调整

    # 检查 GPU 是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 获取数据加载器
    train_loader, val_loader = get_imagenet_dataloaders(data_directory, batch_size, num_workers)

    # 初始化模型
    # model = resnet18(num_classes=num_classes, pretrained=False) # 从头训练
    model = resnet18(num_classes=num_classes, pretrained=True) # 使用预训练模型（推荐，并进行微调）
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 可以使用学习率调度器来进一步优化训练过程，例如：
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # scheduler.step() # 如果使用了学习率调度器，在每个 epoch 结束后调用

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = 'resnet18_imagenet_best.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with validation accuracy: {best_val_acc:.4f}")

    print("Training finished!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # --- 如何加载和使用保存的模型 ---
    # saved_model = resnet18(num_classes=num_classes)
    # saved_model.load_state_dict(torch.load('resnet18_imagenet_best.pth'))
    # saved_model = saved_model.to(device)
    # saved_model.eval()
    # print("Model loaded successfully for inference.")