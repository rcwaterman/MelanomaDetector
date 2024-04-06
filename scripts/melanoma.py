import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from kaggle.api.kaggle_api_extended import KaggleApi

#Pull the api key. Defualt key location is C:\\Users\"your_user"\.kaggle\kaggle.json
api = KaggleApi()
api.authenticate()

#Data path
path = os.path.join(os.path.dirname(os.getcwd()), r'datasets\Melanoma')

#Download data to datapath and unzip it.
api.dataset_download_files("bhaveshmittal/melanoma-cancer-dataset", path=path, unzip=True)
print(f'Data saved to {path}')

train_path = os.path.join(path, "train")
test_path = os.path.join(path, "test")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device : {device}')
