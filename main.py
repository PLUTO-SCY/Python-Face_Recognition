# 导入需要的包
import os
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import wx
from PIL import Image

# 设置超参数
BATCH_SIZE = 128  # 如果是笔记本电脑跑或者显卡显存较小，可以减小此值
LR = 0.001  # 学习率
MM = 0.95  # 随机梯度下降法中momentum参数  通常，在初始学习稳定之前，取0.5，之后取0.9或更大。
EPOCH = 0  # 训练轮数
Num_Workers = 4

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

# 图像变换
data_transforms = {
    "train":
        transforms.Compose([
            transforms.Resize((227, 227)),  # 调整图片大小为给定大小
            transforms.RandomHorizontalFlip(),  # 依概率p水平翻转：transforms.RandomHorizontalFlip(p=0.5)，p默认值为0.5
            transforms.ToTensor(),  # 将图片转换为tensor，并且归一化至[0-1] 归一化至[0-1]是直接除以255
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    "test":
        transforms.Compose([
            transforms.Resize((227, 227)),  # 调整图片大小为给定大小
            transforms.RandomHorizontalFlip(),  # 依概率p水平翻转：transforms.RandomHorizontalFlip(p=0.5)，p默认值为0.5
            transforms.ToTensor(),  # 将图片转换为tensor，并且归一化至[0-1] 归一化至[0-1]是直接除以255
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
}
# 通用的数据加载器
train_dataset = datasets.ImageFolder(root='./data/train', transform=data_transforms["train"])  # 加载训练集
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,  # 一次训练所抓取的数据样本数量
                              shuffle=True,  # 每次对数据进行重新排序
                              num_workers=Num_Workers)  # 有几个进程来处理data loading

val_dataset = datasets.ImageFolder(root='./data/val', transform=data_transforms["test"])  # 加载验证集
val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,  # 在验证集中不需要每次对数据进行重新排序
                            num_workers=Num_Workers)  # 有几个进程来处理data loading

test_dataset = datasets.ImageFolder(root='./data/test', transform=data_transforms["test"])  # 加载测试集
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,  # 一次训练所抓取的数据样本数量；
                             shuffle=True,  # 在测试集中不需要每次对数据进行重新排序
                             num_workers=Num_Workers)  # 有几个进程来处理data loading

"""
返回的dataset都有以下三种属性：
self.classes：用一个 list 保存类别名称
self.class_to_idx：类别对应的索引，与不做任何转换返回的 target 对应
self.imgs：保存(img-path, class) tuple的 list

pytorch 的数据加载到模型的操作顺序是这样的：
① 创建一个 Dataset 对象
② 创建一个 DataLoader 对象
③ 循环这个 DataLoader 对象，将img, label加载到模型中进行训练
"""


# 定义网络结构，简单的网络结构可以通过nn.Sequential来实现，复杂的
# 网络结构需要通过继承nn.Module来自定义网络类来实现，在此使用自定义
# 类的方法给出一个简单的卷积神经网络，包括两个卷积层和两个全连接层
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, 4)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, 2)

    def forward(self, x):  # in:277*277*3
        x = self.conv1(x)  # in:227*227*3 out:96*55*55
        x = self.relu(x)
        x = self.max_pool(x)  # in:96*55*55  out:27*27*96
        x = self.conv2(x)  # in:27*27*96 out:13x13x256
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)  # in:27*27*96 out:13x13x384
        x = self.relu(x)
        x = self.conv4(x)  # in:13x13x384 out:13x13x384
        x = self.relu(x)
        x = self.conv5(x)  # in:13x13x384 out:6x6x256
        x = self.relu(x)
        x = self.max_pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


model = Net().to(device)

# 定义损失函数，分类问题采用交叉熵损失函数
loss_func = nn.CrossEntropyLoss()

# 定义优化方法，此处使用随机梯度下降法
optimizer_ft = optim.SGD(model.parameters(), lr=LR, momentum=MM)
# 定义优化方法，此处采取adam优化算法
# optimizer_ft = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


# 定义每5个epoch，学习率变为之前的0.1
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)


# 训练神经网络
def train_model(model, criterion, optimizer, scheduler):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    scheduler.step()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    print('train Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))
    return model


def val_model(model, criterion):  # 这是利用验证集对每一轮训练完成的模型进行检验
    model.eval()  # 不启用 Batch Normalization 和 Dropout。
    running_loss = 0.0
    running_corrects = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_dataset)
    epoch_acc = running_corrects.double() / len(val_dataset)

    print('val Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))
    return epoch_acc


def test_model(model, criterion):  # 这是利用测试集进行最终的对模型效果的测试
    model.eval()  # 不启用 Batch Normalization 和 Dropout。
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_dataset)
    epoch_acc = running_corrects.double() / len(test_dataset)

    print('***  test Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))
    return epoch_acc


def pre_model(model, criterion):  # 这是利用训练好的模型对单张图片进行预测
    model.eval()  # 不启用 Batch Normalization 和 Dropout。
    fig = plt.figure()
    with torch.no_grad():
        for inputs, labels in pre_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            PreLable = labels.data.item()
            if preds == labels.data:
                if PreLable == 1:
                    print("man\n")
                    return "man"
                    sys.exit()
                else:
                    print("woman\n")
                    return "woman"
                    sys.exit()
            else:
                if PreLable == 0:
                    print("man\n")
                    return "man"
                    sys.exit()
                else:
                    print("woman\n")
                    return "woman"
                    sys.exit()


class Myframe(wx.Frame):
    def __init__(self, filename, stt):
        wx.Frame.__init__(self, None, -1, u'Predict', size=(640, 740))
        self.filename = filename
        self.Bind(wx.EVT_SIZE, self.change)
        self.p = wx.Panel(self, -1)
        self.SetBackgroundColour('white')

        text = wx.StaticText(self, -1, stt, (230, 50))
        font = wx.Font(20, wx.ROMAN, wx.NORMAL, wx.BOLD)
        text.SetFont(font)

    def start(self):
        self.p.DestroyChildren()  # 抹掉原先显示的图片
        self.width, self.height = self.GetSize()
        self.height += 100
        image = Image.open(self.filename)
        self.x, self.y = image.size
        self.x = self.width / 2 - self.x / 2
        self.y = self.height / 2 - self.y / 2
        self.pic = wx.Image(self.filename, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        # 通过计算获得图片的存放位置
        self.button = wx.BitmapButton(self.p, -1, self.pic, pos=(int(self.x), int(self.y)))
        self.p.Fit()

    def change(self, size):  # 如果检测到框架大小的变化，及时改变图片的位置
        if self.filename != "":
            self.start()
        else:
            pass


# 训练和测试
if __name__ == "__main__":
    since = time.time()
    best_acc = 0.0

    model.load_state_dict(torch.load('model-0810-7.pkl'))

    for epoch in range(EPOCH):
        print('\nEpoch {}/{}'.format(epoch, EPOCH - 1))
        print('-' * 10)

        model = train_model(model, loss_func, optimizer_ft, exp_lr_scheduler)
        epoch_acc = val_model(model, loss_func)
        if epoch_acc > best_acc:
            torch.save(model.state_dict(), 'model-0810-9.pkl.pkl')
        best_acc = epoch_acc if epoch_acc > best_acc else best_acc
    if EPOCH != 0:
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best Val Acc: {:4f}'.format(best_acc))

        print("\n\n\n******test begin: ******")
        test_acc = test_model(model, loss_func)
        print('Test Acc: {:4f}'.format(test_acc))

# 以下均为单张图片的预测程序
    if EPOCH == 0:
        mkdirName4 = 'H:\\人脸识别\\torch1.9.0+anaconda环境版本\\data\\pre\\female'
        shutil.rmtree(mkdirName4)
        os.mkdir(mkdirName4)
        file_dir = "./Photo_To_Predict"
        for root, dirs, files in os.walk(file_dir, topdown=False):
            #print(root)  # 当前目录路径
            # print(dirs)  # 当前目录下所有子目录
            # print(files[0])  # 当前路径下所有非目录子文件
            photoName = files[0]
            filename2 = root + '/' + files[0]
            shutil.move(filename2, 'H:\\人脸识别\\torch1.9.0+anaconda环境版本\\data\\pre\\female')

        pre_dataset = datasets.ImageFolder(root='./data/pre', transform=data_transforms["test"])  # 加载预测集
        pre_dataloader = DataLoader(dataset=pre_dataset,
                                    batch_size=1,
                                    shuffle=False,  # 在预测集中不需要每次对数据进行重新排序
                                    num_workers=1)  # 有几个进程来处理data loading

        stt = pre_model(model, loss_func)

        app = wx.App()
        filename3 = './data/pre/female/'+photoName
        frame = Myframe(filename3, "Predict: " + stt)
        frame.start()
        frame.Center()
        frame.Show()
        app.MainLoop()

# torch.save(model.state_dict(),'model.pkl')
