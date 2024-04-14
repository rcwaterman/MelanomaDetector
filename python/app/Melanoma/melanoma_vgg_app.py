"""
Test script for the benign/malignant classification model. Select a dir, 
"""

from tkinter import filedialog
import os
import torch
import torch.nn as nn
import time
import numpy as np
from torchvision import transforms, datasets
from torchvision.models import vgg19
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#Paths
image_dir = filedialog.askdirectory()
image_list = [os.path.join(image_dir,file) for file in os.listdir(image_dir)]
model_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'models\Melanoma\melanoma_vgg_v1.pt')

#Define image size. Cursory look at data shows image sizes of 224 x 224. Shrink this down to speed up training.
IMG_SIZE = 168

def create_model(device):
    #Instantiate the VGG model with default weights
    model = vgg19()

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
    model.eval()

    transformer = transforms.Compose([
            #Load image into tensor and normalize
            transforms.Resize(size = (IMG_SIZE, IMG_SIZE), antialias = True),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
    )
    
    print("Loading dataset...")
    data = datasets.ImageFolder(image_dir, transformer)
    loader = DataLoader(data, batch_size = 500, shuffle = True)
    print("Load complete")
    # Display image and label.
    inputs, labels = next(iter(loader))
    
    avg_time = []
    running_accuracy = []

    for i in range(0, len(inputs), 1):
        inp, lab = inputs[i], labels[i]
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                            std = [ 1., 1., 1. ]),
                                        ])
        inp = invTrans(inp)
        inp = inp.unsqueeze(0)
        img = inp.squeeze(0).T
        inp, lab = inp.to(device), lab.to(device)
        start = time.time()
        pred = model(inp)
        end = time.time()
        pred_time = end - start
        avg_time.append(pred_time)
        predicted = (torch.sigmoid(pred) > 0.5).float()
        if predicted == lab:
            running_accuracy.append(1)
        else:
            running_accuracy.append(0)
        print(f'Prediction: {int(predicted[0])}', f"Label: {lab}", f'Evaluation Time: {sum(avg_time)/len(avg_time)}',f'Running Accuracy: {round((sum(running_accuracy)/len(running_accuracy))*100, 2)}%')
        #plt.imshow(img)
        #plt.show()

if __name__ == '__main__':
    main()