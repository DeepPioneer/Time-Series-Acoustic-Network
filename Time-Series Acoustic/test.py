from sklearn.metrics import roc_curve, cohen_kappa_score, auc
import torch
from random import random
import torch.nn as nn
import matplotlib.pyplot as plt
import os, time, argparse,glob
import numpy as np
from util.data_loader import get_data_loaders
from model_method.WaveMsNet import WaveMsNet
from model_method.SimPFs import SimPFs_model
from model_method.wavegram_mel import Wavegram_Logmel
from model_method.wave_transformer import restv2_tiny
from model_method.TSNA import TSLANet
from model_method.ResNets import ResNet38,Res1dNet31,M18
from create_model.Demo import Demo_temporarily
from create_model.new_model import LATSANet
from create_model.TS_Net import TS_model
import torch.optim as optim
from config import get_args_parser
from util.optim_factory import create_optimizer
import warnings
from create_model.original import original_model
warnings.filterwarnings("ignore")

# Set device
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# denormalize and show an image
def imshow_mel(audio, title=None):
    npimg = audio.numpy()
    plt.imshow(npimg, origin='lower', aspect='auto')
    if title:
        plt.title(title)

# 可视化模型函数
def visualize_model(model, class_mapping, test_loader, num_waveplot=6):
    was_training = model.training  # if true, the model is in training mode otherwise in evaluate mode
    model.eval()
    audio_so_far = 0
    fig = plt.figure()
    for step, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(torch.float32).unsqueeze(1).to(device)
        labels = torch.LongTensor(labels.numpy()).to(device)
        outputs = model(inputs).to(device)
        preds = torch.max(outputs, 1)[1]
        for i in range(inputs.size(0)):
            audio_so_far += 1
            plt.subplot(2, 3, audio_so_far)
            # plt.title('predicted: {},expected:{}'.format(class_mapping[preds[i].cpu()], class_mapping[labels[i].cpu()]))
            # 若预测正确，显示为蓝色；若预测错误，显示为红色
            color = 'blue' if class_mapping[preds[i].cpu()] == class_mapping[labels[i].cpu()] else 'red'
            plt.title('predict:{},expected:{}'.format(class_mapping[preds[i].cpu()],class_mapping[labels[i].cpu()]), color=color)
            # plt.title('predict:{}'.format(class_mapping[preds[i].cpu()]), color=color)

            print('predicted: {},expected:{}'.format(class_mapping[preds[i].cpu()], class_mapping[labels[i].cpu()]))
            imshow_mel(inputs.cpu().data[i])
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
            plt.axis('off')
            if audio_so_far == num_waveplot:
                model.train(mode=was_training)
                return

    model.train(mode=was_training)

def test(model, test_loader, test_path, args):
    model.eval()
    start = time.time()
    test_loss = 0.0
    total = 0
    correct = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(torch.float32).unsqueeze(1).to(device)
            labels = torch.LongTensor(labels.numpy()).to(device)
            
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(output.cpu().numpy())

    # print(len(test_loader.dataset), total)
    test_acc = 100. * correct / total

    # Calculate ROC curve and AUC
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(args.num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure()
    for i in range(args.num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    ROC_path = f"{args.noise_level}/{args.data_type}/{args.model_name}/ROC.png"

    plt.savefig(ROC_path, bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()

    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(all_labels, all_preds)

    log_message = (
        f'Test set: Test acc:  {test_acc:.2f}%, Kappa: {kappa:.2f}')

    print(log_message)

    with open(test_path, "a") as file:
        file.write(log_message + "\n")

    return test_acc, kappa


# 初始网络模型
def return_model(args):
    if args.model_name == 'WaveMsNet':
        model = WaveMsNet(num_classes=args.num_classes).to(device)
    elif args.model_name == 'SimPFs':
        model = SimPFs_model(classes_num=args.num_classes).to(device)
    elif args.model_name == 'Wavegram_Logmel':
        model = Wavegram_Logmel(sample_rate=args.sample_rate, window_size=args.window_size,
                                      hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
                                      classes_num=args.num_classes).to(device)

    elif args.model_name == 'WTS':
        model = restv2_tiny(pretrained=False,num_classes=args.num_classes).to(device)
        
    elif args.model_name == 'TSNA':
        model = TSLANet().to(device)
        
    elif args.model_name == 'Res1d':
        model = Res1dNet31(classes_num=args.num_classes).to(device)
        
    elif args.model_name == 'Res2d':
        model = ResNet38(sample_rate=args.sample_rate, window_size=args.window_size,
                                         hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
                                         classes_num=args.num_classes).to(device)
    elif args.model_name == 'M18':
        model = M18().to(device)
        
    elif args.model_name == 'Demo':
        model = Demo_temporarily(pretrained=False, num_classes=5).to(device)
        
    elif args.model_name == 'new_model':
        model = LATSANet(pretrained=False, num_classes=5).to(device)
          
    elif args.model_name == 'TS':
        model = TS_model(pretrained=False, num_classes=5).to(device)

    elif args.model_name == 'original':
        model = original_model(pretrained=False, num_classes=5).to(device)
    return model

# 初始化数据集
def return_data(args):
    if args.no_noise:
        if args.data_type == 'ESC':
            return r'ori_dataSet/Cut_ESC'
        elif args.data_type == 'Cut_ShipEar':
            return r"ori_dataSet/Cut_ShipEar"
        elif args.data_type == 'Deepship':
            return r"ori_dataSet/Cut_deepShip"
        elif args.data_type == 'Whale':
            return r"ori_dataSet/Cut_whale"
        else:
            return None
    else:
        if args.data_type == 'ESC':
            return f'dataSet/{args.noise_path}/ESC'
        elif args.data_type == 'Cut_ShipEar':
            return f"dataSet/{args.noise_path}/Cut_ShipEar"
        elif args.data_type == 'Cut_deepShip':
            return f"dataSet/{args.noise_path}/Cut_deepShip"
        elif args.data_type == 'Cut_whale':
            return f"dataSet/{args.noise_path}/Cut_whale"
        else:
            return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model Training', parents=[get_args_parser()])
    args = parser.parse_args()
    # -------------------------加载模型------------------------#
    model = return_model(args)
    model.load_state_dict(torch.load(
        f'{args.noise_level}_model_save/{args.data_type}/{args.model_name}/{args.data_type}_{args.model_name}.pkl'))
    
    parser = argparse.ArgumentParser('Model Training', parents=[get_args_parser()])
    args = parser.parse_args()

    data_set = return_data(args)
    if not os.path.exists('{}/{}/{}'.format(args.noise_level, args.data_type, args.model_name)): os.makedirs(
        '{}/{}/{}'.format(args.noise_level, args.data_type, args.model_name))
    
    test_path = f"{args.noise_level}/{args.data_type}/{args.model_name}/test_accuracy.txt"
    
    train_loader, test_loader = get_data_loaders(data_set,args.batch_size, num_workers=8)
    
    test_acc, kappa = test(model, test_loader, test_path, args)

    if args.data_type == "ESC":
        CLASS_MAPPING = ['airplane', 'breathing', 'brushing_teeth', 'can_opening', 'car_horn']
    elif args.data_type == "Cut_deepShip":
        CLASS_MAPPING = ['Cargo', 'Passengership', 'Tanker', 'Tug']
    elif args.data_type == "Cut_ShipEar":
        CLASS_MAPPING = ["0", "1", "2", "3", "4"]
    elif args.data_type == "Cut_whale":
        CLASS_MAPPING = ["Background_noise", "Bowhead_whale", "Humpback_whale", "Pilot_whale", "Ship_signal"]
    else:
        CLASS_MAPPING = None

    # -------------------------模型预测结果可视化------------------------#
    plt.figure(figsize=(8, 8))
    visualize_model(model, CLASS_MAPPING, test_loader)
    # plt.show()
    plt.savefig(f'{args.noise_level}/{args.data_type}/{args.model_name}/visualize.png', bbox_inches='tight',
                pad_inches=0, dpi=300)
    plt.close()  # 关闭当前图像
    
