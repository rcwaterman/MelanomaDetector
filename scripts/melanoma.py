import os
from kaggle.api.kaggle_api_extended import KaggleApi

#Pull the api key. Defualt key location is C:\\Users\"your_user"\.kaggle\kaggle.json
api = KaggleApi()
api.authenticate()

#Data path
path = os.path.join(os.path.dirname(os.getcwd()), r'datasets\Melanoma')

#Download data to datapath and unzip it.
api.dataset_download_files("bhaveshmittal/melanoma-cancer-dataset", path=path, unzip=True)



