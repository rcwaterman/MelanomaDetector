"""
Test script for the benign/malignant classification model. Select a dir, 
"""

from tkinter import filedialog
import os
import cv2
import torch
import torch.nn as nn
from torchvision import models,transforms

#Paths
image_dir = filedialog.askdirectory()
image_list = [os.path.join(image_dir,file) for file in os.listdir(image_dir)]
state_dict_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'models\Melanoma\melanoma_vgg.pt')

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
    model.load_state_dict(state_dict_path)
    model.eval()

    for image in image_list:
        img = cv2.imread(image)
        cv2.imshow("Image", img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = transforms.ToTensor(img)
        output = model(img)
        print(output)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()