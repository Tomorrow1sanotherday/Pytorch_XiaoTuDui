import torch
from torch.utils.tensorboard import SummaryWriter

from model import *
#准备数据集
import torchvision
from torch import nn, optim
import time

train_data = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())


test_data = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())

#数据长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为{}".format(train_data_size))
print("测试数据集的长度为{}".format(test_data_size))

#利用dataloader加载数据集
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)





#创建模型
tudui = Tudui()
tudui = tudui.cuda()

#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

#定义优化器
learning_rate = 0.01
optimizer = optim.SGD(tudui.parameters(), lr=learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 10


#添加tensorboard
writer = SummaryWriter("logs")

start_time = time.time()
for i in range(epoch):
    print("第{}轮训练开始".format(i))

    #训练步骤开始
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = tudui.forward(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数{}，loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #测试步骤
    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            outputs = tudui.forward(images)
            loss = loss_fn(outputs, labels)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == labels).sum().item()
            total_accuracy += accuracy
    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()









