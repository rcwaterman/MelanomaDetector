"""
Test script for the benign/malignant classification model. Select a dir, 
"""

from tkinter import filedialog
import os
import cv2
import torch
from scripts.Melanoma.melanoma_vgg import IMG_SIZE, create_model

#Paths
image_dir = filedialog.askdirectory()
image_list = [os.path.join(image_dir,file) for file in os.listdir(image_dir)]
model_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'models\Melanoma\melanoma_vgg.pt')

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
