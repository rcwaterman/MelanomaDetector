# Melanoma Models

The melanoma models are models, ranging from VGG19 to EfficientNet V2, that have been fine tuned to identify melanoma based on training data from the following resources:

* [PH2 Dataset](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar)

* [Kaggle Dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset)

An additional YOLO_V8 model will be fine tuned to identify moles and skin growths in a general image, then crop and analyze the whether the growth is malignant. 

Implementation of this application is done via the melanoma_app.py script, which reads frames from the camera in real time, pushes it through the YOLO_V8 model, then pushes all found skin growths through the melanoma identifying model. 

Testing will be needed upon completion of the application script to determine performance on different hardware. Ideally, the application will be performant enough for general consumer use on most modern cell phones.