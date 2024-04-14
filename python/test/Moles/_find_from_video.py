from ultralytics import YOLO
import os
import cv2

def main():
    #Load the model
    model = YOLO(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'models\Moles\runs\detect\training\weights\best.pt'))

    results = model(source="0", show=True, conf=0.75)

if __name__=='__main__':
    main()