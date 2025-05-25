from ultralytics import YOLOv10

model = YOLOv10('runs/detect/train/weights/best.pt')
model.val(data='data/data.yaml', batch=8, workers=0)