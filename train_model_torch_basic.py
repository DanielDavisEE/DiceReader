import numpy as np
import os, time, copy
import cv2
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def import_data():
    dice_types = ['d6']
    
    # Initialise dataset
    all_x = np.ones((0, 50, 50), dtype=np.uint8)
    all_y = np.ones((1), dtype=np.uint8)
    
    # Import images
    for dice in dice_types:
        for face in os.listdir(f'{dice}\\train'):
            for image_name in os.listdir(f'{dice}\\train\\{face}'):
                image = cv2.imread(f'{dice}\\train\\{face}\\{image_name}')
                image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape((1, 50, 50))
                all_x = np.append(all_x, image_grey, axis=0)
                all_y = np.append(all_y, [int(face) - 1])
    
    N = all_x.shape[0]
    index = np.arange(N)
    np.random.shuffle(index)
    
    # Split data into test and train sets
    train_N = int(N * 0.8)
    
    train_images = all_x[index[:train_N]]
    train_labels = all_y[index[:train_N]]
    
    val_images = all_x[index[train_N:]]
    val_labels = all_y[index[train_N:]]
    
    class_names = np.unique(all_y)
    
    # Data Normalization
    # Conversion to float
    train_images = train_images.astype(np.float32) 
    val_images = val_images.astype(np.float32)
    
    # Normalization
    train_images = train_images/255.0
    val_images = val_images/255.0

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(50*50, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)

X = torch.rand(1, 50, 50, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")