import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vgg19, VGG19_Weights
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from kaggle.api.kaggle_api_extended import KaggleApi

#Define image size. Cursory look at data shows image sizes of 224 x 224. Shrink this down to speed up training.
IMG_SIZE = 168

def create_model(device):
    #Instantiate the VGG model with default weights
    model = vgg19(weights=VGG19_Weights)

    #Uncomment to print model structure
    #This is necessary for modifying the classification layer
    """
    for (name, layer) in model._modules.items():
        #iteration over outer layers
        print((name, layer))
    """

    # Modifying final classifier layer
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)

    #Uncomment to print model structure
    #This is necessary for modifying the classification layer
    """
    for (name, layer) in model._modules.items():
        #iteration over outer layers
        print((name, layer))
    """

    #Send model to device
    model = model.to(device)

    return model

def main():

    #Pull the api key. Defualt key location is C:\\Users\"your_user"\.kaggle\kaggle.json
    api = KaggleApi()
    api.authenticate()

    #Data path
    path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'datasets\Melanoma')

    #Download data to datapath and unzip it.
    if not os.path.exists(path):
        api.dataset_download_files("bhaveshmittal/melanoma-cancer-dataset", path=path, unzip=True)
        print(f'Data saved to {path}')

    #Define the data path
    train_path = os.path.join(path, "train")
    val_path = os.path.join(path, "test")

    #Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device : {device}')

    #Model setup based on this notebook: https://www.kaggle.com/code/bhaveshmittal/cancer-detection-using-pytorch-96-accuracy/notebook

    # Training transformer
    train_transformer = transforms.Compose([
        #Randomly rotate the images
        transforms.RandomRotation(degrees = 20),
        
        #Randomly flip the images
        transforms.RandomHorizontalFlip(p = 0.3),
        transforms.RandomVerticalFlip(p = 0.3),
        
        #Resize all images to the previously defined image size
        transforms.Resize(size = (IMG_SIZE, IMG_SIZE), antialias = True),
        transforms.CenterCrop(size = (IMG_SIZE, IMG_SIZE)),
        
        #Load image into tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    # Validation transformer
    val_transformer = transforms.Compose([
        #Resize all images to the previously defined image size
        transforms.Resize(size = (IMG_SIZE, IMG_SIZE), antialias = True),
        transforms.CenterCrop(size = (IMG_SIZE, IMG_SIZE)),

        #Load image into tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    #Load the datasets
    train_data = datasets.ImageFolder(train_path, train_transformer)
    val_data = datasets.ImageFolder(val_path, val_transformer)

    #Define the training parameters and load the data
    batch_size = 256
    print("Loading Data...")
    trainLoader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 4)
    valLoader = DataLoader(val_data, batch_size = batch_size, shuffle = False, num_workers = 4)

    model = create_model(device)

    # Defining the loss, optimizer, and annealer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = ReduceLROnPlateau(optimizer, threshold = 0.01, factor = 0.1, patience = 3, min_lr = 1e-5)

    patience = 5
    minDelta = 0.01
    currentPatience = 0
    bestLoss = float('inf')

    # Gradient scaler for mixed-precision training
    scaler = GradScaler()

    # Lists to store training and validation metrics
    trainLosses = []
    valLosses = []
    valAccs = []

    # Training loop
    epochs = 30

    print("Starting training...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        runningLoss = 0.0

        print(f'Beginning epoch {epoch+1}...')

        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            runningLoss += loss.item()
            print(f'Batch Loss: {loss.item()}')

        trainLoss = runningLoss / len(trainLoader)
        print(f'Epoch {epoch + 1}/{epochs} - Training Loss : {trainLoss:.2f}')
        trainLosses.append(trainLoss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            valLoss = 0.0
            correct = total = 0

            for inputs, labels in valLoader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.unsqueeze(1).float()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valLoss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avgLoss = valLoss / len(valLoader)
            accuracy = correct / total * 100

            print(f'Validation Loss : {avgLoss:.2f} Validation Accuracy : {accuracy:.2f}%\n')
            valLosses.append(avgLoss)
            valAccs.append(accuracy)

            # Early stopping
            if avgLoss < bestLoss - minDelta:
                bestLoss = avgLoss
                currentPatience = 0
            else:
                currentPatience += 1
                if currentPatience >= patience:
                    print('Early stopping triggered.')
                    break

            scheduler.step(avgLoss)

            #Save checkpoints
            if (epoch+1)%2 == 0:
                model_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'models\Melanoma\melanoma_vgg19.pt')
                torch.save(model.state_dict(), model_path)

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'models\Melanoma\melanoma_vgg_19.pt')

    torch.save(model.state_dict(), model_path)
                

if __name__ == '__main__':
    main()
    