from ultralytics import YOLO

model = YOLO("/home/rocknroll1397/Image_Analyzer/runs/segment/train_veo3_v2/weights/last.pt")
model.train(resume=True)
