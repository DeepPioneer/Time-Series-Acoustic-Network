import librosa
from warnings import filterwarnings
filterwarnings('ignore')
import librosa
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import librosa.display
import glob,os
import numpy as np
np.random.seed(123)
import torch.utils.data as Data

def make_frames(filename,folder,frame_length,overlapping_fraction):
    class_id = os.path.basename(folder)
    filename = folder + '/'+filename
    data, sample_rate = librosa.load(filename, sr=16000)
    stride = int((1 - overlapping_fraction) * frame_length)
    num_frames = int((len(data) - frame_length) / stride) + 1
    temp = np.array([data[i * stride:i * stride + frame_length] for i in range(num_frames)])
    if (len(temp.shape) == 2):
        res = np.zeros(shape=(num_frames, frame_length + 1), dtype=np.float32)

        res[:temp.shape[0], :temp.shape[1]] = temp
        res[:, frame_length] = np.array([class_id] * num_frames)
        return res

def make_frames_folder(data_set,folders,frame_length,overlapping_fraction):
    data = []
    for folder in folders:  # folder 0,1,2,3,4
        file_path = data_set + '/' + folder
        files = os.listdir(data_set + '/' + folder)
        for file in files:
            res = make_frames(file, file_path, frame_length, overlapping_fraction)
            if res is not None:
                data.append(res)
    dataset = data[0]
    for i in range(1, len(data)):
        dataset = np.vstack((dataset, data[i]))
    return dataset

frame_length = 1024
overlapping_fraction = 0.75

def my_data(data_set):
    folders = os.listdir(data_set)    
    dataSet = make_frames_folder(data_set,folders, frame_length, overlapping_fraction)
    audio = dataSet[:, 0:frame_length]
    audio = torch.from_numpy(audio)
    label = dataSet[:, frame_length]
    label = torch.tensor(label,dtype=torch.int64)
    train_data = Data.TensorDataset(audio, label)
    return train_data

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 设置GPU cuda:0是3080ti
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# noise_dir = "../../数据集/4"
# noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]
# 只选择文件名以 "83_" 开头的音频文件
# noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.startswith('83_') and f.endswith('.wav')]
categories = ['airplane', 'breathing', 'brushing_teeth', 'can_opening', 'car_horn']
# categories = ['0', '1', '2', '3', '4']
# 创建一个字典，将类别映射为整数
category_to_int = {category: idx for idx, category in enumerate(categories)}
print(category_to_int)

# 映射示例
category = 'breathing'
category_int = category_to_int.get(category, -1)  # 如果类别不存在，可以设置默认值

print(f"{category} 映射为整数: {category_int}")

# 构建数据集
class SoundDataset(Dataset):
    def __init__(self, file_path):
        # files-> ['Ships/Dredger','Ships/Fishboat',...]
        files = [file_path + "/" + x for x in os.listdir(file_path) if os.path.isdir(file_path + "/" + x)]
        # 随机选择噪声文件，并允许重复以匹配目标信号数量
        self.labels = []
        self.audio_name = []
        for idx, folder in enumerate(files):
            for audio in glob.glob(folder + '/*.wav'):
                self.audio_name.append(audio)
                x = os.path.basename(folder) #用于获取指定路径的最后一部分，也就是文件名或文件夹名。
                self.labels.append((int)(category_to_int.get(x, -1)))
                # print(self.labels)
                # print(self.audio_name)
        self.file_path = file_path

    def __getitem__(self, index):
        # 为每个音频文件随机选择一个噪声文件
        # self.selected_noise_files = [random.choice(noise_files) for _ in self.audio_name]
        # print(len(self.selected_noise_files))
        audio = self.audio_name[index]  # + '.wav'
        soundData,fs = librosa.load(audio,sr=16000) # 降采样到16000
        soundData = torch.tensor(soundData)
        return soundData, self.labels[index]

    def __len__(self):
        return len(self.audio_name)

