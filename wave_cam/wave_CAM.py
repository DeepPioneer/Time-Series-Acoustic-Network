import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from wave_cnn import WaveMsNet

def grad_cam(model, input_image, target_class):
    model.eval()

    output = model(input_image)
    model.zero_grad()
    class_loss = output[:, target_class].sum()
    class_loss.backward()

    gradients = model.gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2])
    feature_maps = model.feature_maps

    for i in range(feature_maps.shape[1]):
        feature_maps[:, i, :] *= pooled_gradients[i]

    heatmap = torch.mean(feature_maps, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().numpy(), 0)
    heatmap = heatmap / np.max(heatmap)

    return heatmap

def visualize_grad_cam(heatmap, input_image):
    heatmap = cv2.resize(heatmap, (input_image.shape[-1], 1))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    input_image = input_image.squeeze().detach().numpy()
    input_image = np.expand_dims(input_image, axis=0)
    heatmap = np.float32(heatmap) / 255
    superimposed_img = heatmap + np.expand_dims(input_image, axis=-1)
    superimposed_img = np.uint8(255 * superimposed_img)

    plt.imshow(superimposed_img[0], origin='lower', cmap='viridis', aspect='auto')
    plt.xlabel("Time Steps")
    plt.ylabel("Channels")
    plt.show()

import librosa,torchaudio
if __name__ == "__main__":
    audio,fs = librosa.load("../audio_project/audio_dataset/ESC/car_horn/1-17124-A-43_1.wav", sr=16000)
    # audio,fs = torchaudio.load("audio_project/audio_dataset/Cut_ShipEar/0/15__10_07_13_1.wav", channels_first=True)
    audio = torch.from_numpy(audio)
    audio = audio.unsqueeze(0)
    audio = audio.unsqueeze(0)
    print(audio.shape)
    model = WaveMsNet(num_classes=5)

    output = model(audio)
    target_class = torch.argmax(output[0])

    heatmap = grad_cam(model, audio, target_class)
    visualize_grad_cam(heatmap, audio[0])