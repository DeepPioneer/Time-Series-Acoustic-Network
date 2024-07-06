from wave_cnn import WaveMsNet
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
import time
from torch.utils.data import DataLoader
from wave_util.preprocess import SoundDataset
import torch
import matplotlib.pyplot as plt
import copy
import torch.nn as nn

device = torch.device("cuda:0")  # sanity check

def key_func(model,train_rate,criterion, train_loader,optimizer, EPOCH):
    since = time.time()
    # 复制模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    ##计算训练使用的batch数量
    batch_num = len(train_loader)
    print("总的batch数量",batch_num)
    train_batch_num = round(batch_num * train_rate)
    print("训练使用的batch数量",train_batch_num)
    # 复制模型的参数
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    for epoch in range(EPOCH):
        print('Epoch {}/{}'.format(epoch, EPOCH - 1))
        print('-' * 10)
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        # Iterate over data.
        for i,(audio,label) in enumerate(train_loader):
            audio = audio.unsqueeze(1)
            if i < train_batch_num:
                model.to(device)
                model.train()  # Set model to training mode
                audio = audio.to(device)
                label = label.to(device)
                outputs = model(audio)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * audio.size(0)
                train_corrects += torch.sum(preds == label)
                train_num += audio.size(0)
            else:
                model.to(device)
                model.eval() 
                audio = audio.to(device)
                label = label.to(device)
                output = model(audio)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, label)
                val_loss += loss.item() * audio.size(0)
                val_corrects += torch.sum(pre_lab == label)
                val_num += audio.size(0)
        scheduler.step()
        ##计算一个epoch在训练集和验证集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{} Train Loss:{:.4f}  Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss:{:.4f}  Val Acc:{:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))
        ##拷贝模型最高精度下的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    ##使用最好模型的参数
    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data={
            "epoch": range(EPOCH),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all
        }
    )
    return model, train_process

if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()
    train_dataset = SoundDataset("../audio_project/audio_dataset/ESC")
    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    model = WaveMsNet()
    # 打印网络参数数量
    print("Num Parameters:", sum([p.numel() for p in model.parameters()]))
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80], gamma=0.1)
    model_ft, train_process = key_func(model,0.75,criterion, train_loader,optimizer, EPOCH=30)
    #保存数据
    with open('example.txt', 'w') as file:
        file.write(f'{train_process.train_loss_all}\n')
    print(type(train_process.train_loss_all.to_numpy()))

    ##可视化模型训练过程
    plt.figure(figsize=(12, 4))
    ##损失函数
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch.to_numpy(), train_process.train_loss_all.to_numpy(), "ro-", label="Train loss",MarkerSize=5)
    plt.plot(train_process.epoch.to_numpy(),  train_process.val_loss_all.to_numpy(), "bs-", label="Val loss",MarkerSize=5)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    ##精度
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch.to_numpy(), train_process.train_acc_all.to_numpy(), "ro-", label="Train acc",MarkerSize=5)
    plt.plot(train_process.epoch.to_numpy(),  train_process.val_acc_all.to_numpy(), "bs-", label="Val acc",MarkerSize=5)
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig("result_plot/Wave_CNN.pdf")
    # 保存模型
    torch.save(model.state_dict(),"Wave_CNN.pth")





