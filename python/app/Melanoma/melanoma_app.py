"""
Test script for the benign/malignant classification model. Select a dir, 
"""

from tkinter import filedialog
from PIL import Image
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models,transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#Paths
image_dir = filedialog.askdirectory()
image_list = [os.path.join(image_dir,file) for file in os.listdir(image_dir)]
model_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'models\Melanoma\melanoma_vgg.pt')

#Define image size. Cursory look at data shows image sizes of 224 x 224. Shrink this down to speed up training.
IMG_SIZE = 168

def create_model(device):
    #Instantiate the VGG model with default weights
    model = models.vgg19()

    #Uncomment to print model structure
    """
    child_counter = 0
    for child in model.children():
    print(" child", child_counter, "is:")
    print(child)
    child_counter += 1

    print(f'\n{model.classifier[6]}\n')
    """

    # Modifying final classifier layer
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)

    #Send model to device
    model = model.to(device)

    return model

def main():
    #Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device : {device}')
    model = create_model(device)
    model.load_state_dict(torch.load(model_path))

    transformer = transforms.Compose([
            #Load image into tensor and normalize
            transforms.Resize(size = (IMG_SIZE, IMG_SIZE), antialias = True),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
    )
    
    data = datasets.ImageFolder(image_dir, transformer)
    loader = DataLoader(data, batch_size = 256, shuffle = True)

    # Display image and label.
    inputs, labels = next(iter(loader))
    input, label = inputs.to(device), labels.to(device)
    output = model(input)

    running_accuracy = []

    for idx,pred in enumerate(output):
        img = inputs[idx].T
        new_img = img
        predicted = (torch.sigmoid(pred) > 0.5).float()
        if predicted == label[idx]:
            running_accuracy.append(1)
        else:
            running_accuracy.append(0)
        print(f'Prediction: {int(predicted[0])}', f"Label: {label[idx]}", f'Running Accuracy: {round((sum(running_accuracy)/len(running_accuracy))*100, 2)}%')
        plt.imshow(new_img)
        plt.show()

if __name__ == '__main__':
    main()