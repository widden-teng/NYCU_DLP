import torch
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from torch .utils.data import TensorDataset,DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from dataloader import read_bci_data


#基本設定與超參數設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_funciton = nn.CrossEntropyLoss()
Batch_size = 128 #128
Learning_rate = 5e-3 #0.001
Epochs = 300

# load data (輸入的資料型態須為tensor)
train_data, train_label, test_data, test_label=read_bci_data()

trian_dataset=TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
trian_loader = DataLoader(trian_dataset, batch_size = Batch_size, shuffle=True)
total_train_size = len(trian_dataset)
train_evaluate = DataLoader(trian_dataset, batch_size = total_train_size)

test_dataset = TensorDataset (torch.Tensor(test_data), torch.Tensor(test_label))
total_test_size = len(test_dataset)
test_loader=DataLoader(test_dataset, batch_size=total_test_size)


class EEGNet(nn.Module):
    def __init__(self, activation=nn.ELU):
        # super() 是一個內建函數，用於調用父類的方法
        # 在子類中重寫一個父類的方法，需要使用 super() 函數調用父類的同名方法，以保留父類的行為
        # 這邊的self為代表nn.Module
        super(EEGNet, self).__init__()
        
        self.first_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51),stride=(1,1),padding=(0,25),bias =True),
            nn.BatchNorm2d(16,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
        )

        self.second_layer = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=(2,1),stride=(1,1),groups=16,bias=True),
            nn.BatchNorm2d(32,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1,4),stride=(1,4),padding=0),
            nn.Dropout(p=0.25)
        )

        self.third_layer = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=(1,15),stride=(1,1),padding=(0,7), bias=True),
            nn.BatchNorm2d(32,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1,8),stride=(1,8),padding=0),
            nn.Dropout(p=0.25)
        )

        self.classify=nn.Sequential(
            nn.Flatten(),
            nn.Linear(736, 2, bias=True),
        )
    def forward(self, x):
        output = self.first_layer(x)
        output = self.second_layer(output)
        output = self.third_layer(output)
        output = self.classify(output)
        return output

class DeepConvNet(nn.Module):
    def __init__(self, activation=nn.ELU):
        super(DeepConvNet, self).__init__()
        self.all_layer = nn.Sequential(
            nn.Conv2d(1, 25 , kernel_size = (1,5), stride = (1,1)),
            nn.Conv2d(25, 25, kernel_size = (2,1), stride = (1,1), bias=True),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine = True, track_running_stats = True), 
            activation(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(25, 50, kernel_size = (1,5), stride = (1,1), bias=True),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine = True, track_running_stats = True), 
            activation(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(50, 100, kernel_size = (1,5), stride = (1,1), bias=True),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine = True, track_running_stats = True), 
            activation(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(100, 200, kernel_size = (1,5), stride = (1,1), bias=True),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine = True, track_running_stats = True), 
            activation(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(8600,2),
        )
        
    def forward(self, input):
        out = self.all_layer(input)
        return out


def training(model, epochs = Epochs, lr = Learning_rate):
    model=model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    n_total_steps = len(trian_loader)
    error_list = []
    train_acc = []
    test_acc = []

    tqdm_epochs = tqdm(range(epochs), desc='Epoch %d' % (1))
    for epoch in tqdm_epochs:
        for step,(train, label) in enumerate(trian_loader):
            train = train.to(device)
            # long() 將標籤張量轉換為整數類別, PyTorch 的損失函數通常要求真實標籤為長整型
            label = label.to(device).long()
            outputs = model(train)

            # optimize parameter
            loss = loss_funciton(outputs, label)
            # optimizer.zero_grad() 会清空当前计算图中所有张量的梯度信息,包括 .grad 属性
            optimizer.zero_grad()
            # loss.backward() 方法進行反向傳播，計算出每個模型參數的梯度。這些梯度會被存儲在每個模型參數的 .grad 屬性中
            loss.backward()
            optimizer.step()
            # print(f' epoch {epoch+1}, iteration {step+1}/{n_total_steps}, loss = {loss.item():.4f}')

        tqdm_epochs.set_description(desc='Epoch %d' % (epoch+1))
        # 由于 Tensor 对象不能直接转换为 Python 数字，需要使用 item() 方法将其转换为标量值
        error_list.append(loss.item())
        train_acc.append(train_accuracy(model))
        test_acc.append(test_accuracy(model))
    return error_list, train_acc, test_acc,max(test_acc),np.argmax(test_acc)

def train_accuracy(model):
    with torch.no_grad():
        for data,target in train_evaluate:
            data,target =data.to(device),target.to(device)
            outputs=model(data)
            target=torch.reshape(target,(total_train_size,1))
            pred =outputs.argmax (dim=1,keepdim=True)
    return len(pred[(pred==target)])/total_train_size * 100  


def test_accuracy(model):
    with torch.no_grad():
        for data,target in test_loader:
            data,target =data.to(device),target.to(device)
            outputs=model(data)
            target=torch.reshape(target,(total_test_size,1))
            pred =outputs.argmax (dim=1,keepdim=True)
    return len(pred[(pred==target)])/total_test_size * 100



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="select the model")
    parser.add_argument('--model', default = "EEGNet")
    parser.add_argument('--batch_size', type = int, default = 256)
    parser.add_argument('--lr', type = float, default = 5e-3)
    parser.add_argument('--epochs', type = int, default = 1000)
    args = parser.parse_args()
    Batch_size = args.batch_size
    Learning_rate = args.lr
    Epochs = args.epochs

    if (args.model == "EEGNet"):
        EEGELU = EEGNet(nn.ELU).to(device)
        EEGLeakyrelu = EEGNet(nn.LeakyReLU).to(device)
        EEGRELU = EEGNet(nn.ReLU).to(device)
        
        errl1, train_accl1, test_acc1, maxacc1, index1 = training(EEGRELU, epochs = Epochs)
        errl2, train_accl2, test_acc2, maxacc2, index2 = training(EEGLeakyrelu, epochs = Epochs)
        errl3, train_accl3, test_acc3, maxacc3,index3 = training(EEGELU, epochs = Epochs)
        max_dic = {'EEGRELU': maxacc1, 'EEGLeakyrelu': maxacc2, 'EEGELU': maxacc3}
    elif (args.model == "DeepConvNet"):
        DeepConvNet_ELU = DeepConvNet(nn.ELU).to(device)
        DeepConvNet_relu = DeepConvNet(nn.LeakyReLU).to(device)
        DeepConvNet_RELU = DeepConvNet(nn.ReLU).to(device)
        
        errl1, train_accl1, test_acc1, maxacc1, index1 = training(DeepConvNet_ELU, epochs = Epochs)
        errl2, train_accl2, test_acc2, maxacc2, index2 = training(DeepConvNet_relu, epochs = Epochs)
        errl3, train_accl3, test_acc3, maxacc3,index3 = training(DeepConvNet_RELU, epochs = Epochs)
        max_dic = {'DeepConvNet_ELU': maxacc1, 'DeepConvNet_relu': maxacc2, 'DeepConvNet_RELU': maxacc3}
    
    plt.figure(figsize = (10,4))
    plt.plot(range(Epochs), test_acc1, label = "relu_test")
    plt.plot(range(Epochs), test_acc2, label = "leaky_relu_test")
    plt.plot(range(Epochs), test_acc3, label = "elu_test")
    plt.plot(range(Epochs), train_accl1, label = "relu_trian")
    plt.plot(range(Epochs), train_accl2, label = "leaky_relu_train")
    plt.plot(range(Epochs), train_accl3, label = "elu_train")
    plt.hlines(y = 87, xmin = 0, xmax = Epochs, linestyle = "--")
    plt.hlines(y = 85, xmin = 0, xmax = Epochs, linestyle = "--")
    plt.hlines(y = 80, xmin = 0, xmax = Epochs, linestyle = "--")
    plt.legend()
    plt.title('Activation function comparision(EEGNET)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.savefig('{}.png'.format(args.model))
    plt.show()
    max_key = max(max_dic, key=max_dic.get)
    print("最高準確率:",max_dic[max_key],"%",' ,active function為:',max_key)

    
    plt.figure(figsize=(10,4))
    plt.plot(range(Epochs),errl1,label="relu_trian")
    plt.plot(range(Epochs),errl2,label="leaky_relu_train")
    plt.plot(range(Epochs),errl3,label="elu_train")
    plt.legend()
    plt.title('Activation function comparision(EEGNET)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss(%)')
    plt.show()
