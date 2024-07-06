import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from wave_util.preprocess import my_data
import torch
from wave_cnn import WaveMsNet
from sklearn.metrics import confusion_matrix

# device = torch.device("cuda:0")
model = WaveMsNet()
model.load_state_dict(torch.load("M3.pth"))
# model = torch.load("save_model/model.pth")
correct = 0
total = 0
test_dataset = my_data(r"underwater/test")
with torch.no_grad():
    audio, labels = test_dataset.tensors[0],test_dataset.tensors[1]
    model.eval()
    audio = audio.unsqueeze(1)
    outputs = model(audio)
    _,predicted=torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (predicted==labels).sum().item()
print(classification_report(labels, predicted))
print("Accuracy of the network on the test audio:%.2f %%" % (100 * correct / total))
##计算混淆矩阵并可视化
index = [0,1,2,3,4]
conf_mat = confusion_matrix(labels, predicted)
df_cm = pd.DataFrame(conf_mat, index=index, columns=index)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("config_result/my_LossM2.pdf")
