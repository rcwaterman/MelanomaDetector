from ultralytics import YOLO
import os

model = YOLO('yolov8m.pt')

yaml_file = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'datasets\Moles\mole_data\yaml.yaml')

results = model.train(data=yaml_file, epochs=100, imgsz=640, verbose=True, device='cuda', optimizer='Adam')
