from ultralytics import YOLO
import os
import time
import cv2

def main():
    model = YOLO(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'scripts\Moles\runs\detect\train6\weights\best.pt'))
    test_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'datasets\Moles\mole_data\images\test')
    test_images = [os.path.join(test_path,img) for img in os.listdir(test_path)]
    for img in test_images:
        results = model.predict(img)
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            annotated = result.plot()  # display to screen
            cv2.imshow("Image", annotated)
            cv2.waitKey(0)

if __name__=='__main__':
    main()