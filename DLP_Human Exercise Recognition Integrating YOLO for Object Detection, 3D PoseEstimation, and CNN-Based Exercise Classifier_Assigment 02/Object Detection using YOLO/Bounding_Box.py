import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image


# Func to load the model
def load_yolo_model():
    # Loading yolov5 from torch hub
    model=torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)
    return model

# Func to detect persons from the image
def detect_persons(model, img_path, conf_threshold = 0.5):    
    #loading image
    img =Image.open(img_path)
    #performing detection
    results= model(img)
    
    #parsing results
    detections=results.xyxy[0]  #bounding boxes with scores and class labels
    person_boxes=[]
    for *box, conf, cls in detections:
        if int(cls)==0 and conf >=conf_threshold:  # Class 0 is 'person' in COCO vocab
            person_boxes.append(box)
    return person_boxes, results.render()[0]

# FUnc to draw BBoxes
def draw_bounding_boxes(image , boxes):
    # xmin, ymin, xmax, ymax
    for box in boxes:
        x1,y1,x2,y2 = map(int,box)
        # drawing rectangle around the detected person or object
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

#loading yolo model from the hub
model=load_yolo_model()
person_boxes,rendered_image =detect_persons(model,'C:\\Users\\outco\Desktop\\i191742_Assignment02_DL_Perception\\Object Detection using YOLO\\a.jpg')

#if detected
if not person_boxes:
    print("No person detected in the image.")
else:
    rendered_pil=Image.fromarray(rendered_image)
    rendered_pil.show()
    print("Detected bounding boxes:",person_boxes)
