from ultralytics import YOLOv10


model = YOLOv10('yolov10s.pt')
model.train(data='data/data.yaml', epochs=100, batch=16, imgsz=640, workers=0)