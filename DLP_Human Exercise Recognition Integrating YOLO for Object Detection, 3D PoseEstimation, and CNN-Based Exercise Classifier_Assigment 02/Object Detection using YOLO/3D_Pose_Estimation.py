import torch
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os

# FUnc to load yolov7 model
def load_yolo_model():

    print("Loading YOLOv5 model...")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #loading model from torch hub
    model =torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    print("***YOLOv5 model loaded successfully***")
    #returning model
    return model

# Func to detect persons
def detect_persons(model, img_path, conf_threshold=0.5):
    
    print(f"Applying Person detection on {img_path}...")
    img=Image.open(img_path)
    results=model(img)
    
    #extracting detections
    detections = results.xyxy[0]  # (x1, y1, x2, y2,confidence,class)
    person_boxes = []
    for det in detections:
        x1,y1,x2,y2,conf,cls = det
        if int(cls)==0 and conf>=conf_threshold:  # Class 0 is 'person' in COCO
            person_boxes.append([int(x1),int(y1),int(x2),int(y2)])
    
    print(f"Detected {len(person_boxes)} person(s) with confidence >= {conf_threshold}")
    
    #rendering detections on image
    rendered_image=np.array(results.render()[0])
    
    return person_boxes,rendered_image

#cropping image
def crop_image(img_path, bbox):
    #reading image
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image from {img_path}")
    x1,y1,x2,y2 = bbox
    cropped_image = image[y1:y2 , x1:x2]
    return cropped_image,(x1,y1)

#estimating poses
def estimate_pose(cropped_image):
    # pose estimation
    mp_pose=mp.solutions.pose
    pose=mp_pose.Pose(static_image_mode=True, model_complexity=2,enable_segmentation=False,min_detection_confidence=0.5)
    
    #converting BGR image to RGB
    image_rgb =cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    results=pose.process(image_rgb)
    
    # Not detected
    if not results.pose_landmarks:
        raise ValueError("No Pose-Landmarks detected.")
    
    landmarks = results.pose_landmarks.landmark
    landmark_coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
    #returing landmark coordinates
    return landmark_coords, results.pose_landmarks

#mapping landmarks to the original
def map_landmarks_to_original(landmarks, bbox, original_image_shape):
    # xmin, ymin, xmax, ymax
    x1,y1,x2,y2 = bbox
    cropped_width = x2-x1
    cropped_height = y2-y1
    mapped_landmarks=[]
    
    for lm in landmarks:
        orig_x=int(lm[0]*cropped_width)+x1
        orig_y=int(lm[1]*cropped_height)+y1
        mapped_landmarks.append((orig_x,orig_y,lm[2]))
    # returing mapped landmarks
    return mapped_landmarks

#FUnct to visualize the landmarks on the image
def visualize_landmarks(original_img_path, mapped_landmarks, output_path=None):
    # reading image
    image=cv2.imread(original_img_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image from {original_img_path}")
    
    #drawing landmarks
    for lm in mapped_landmarks:
        x,y,z = lm
        cv2.circle(image, (int(x),int(y)),3,(0, 255, 0), -1)
    
    #drawing connections
    mp_pose=mp.solutions.pose
    pose_connections = mp_pose.POSE_CONNECTIONS
    for connection in pose_connections:
        start_idx,end_idx=connection
        if start_idx < len(mapped_landmarks) and end_idx<len(mapped_landmarks):
            start_point=(int(mapped_landmarks[start_idx][0]) , int(mapped_landmarks[start_idx][1]))
            end_point=(int(mapped_landmarks[end_idx][0]) , int(mapped_landmarks[end_idx][1]))
            cv2.line(image,start_point,end_point, (255,0,0), 2)
    
    #displaying the image
    cv2.imshow('3D Pose Estimation',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if output_path:
        cv2.imwrite(output_path,image)
        print(f"Pose estimation result saved at {output_path}")
    
    return image


#main function
def main(img_path, output_dir=None, conf_threshold=0.5):
    #if img not exists
    if not os.path.exists(img_path):
        print(f"Image path '{img_path}' does not exist.")
        return
    #output directory
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory at {output_dir}")
    
    #loading YOLOv5 model
    model=load_yolo_model()
    #detecting persons in the image
    person_boxes,rendered_image = detect_persons(model,img_path,conf_threshold)
    # no person detected
    if not person_boxes:
        print("No person detected in the image.")
        return
    
    # with bounding boxes
    if output_dir:
        rendered_img_path=os.path.join(output_dir, "detected_persons.jpg")
        cv2.imwrite(rendered_img_path,cv2.cvtColor(rendered_image,cv2.COLOR_RGB2BGR))
        print(f"Detected persons image saved at {rendered_img_path}")
    else:
        #convert RGB to BGR
        rendered_image_bgr=cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Detected Persons',rendered_image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    #loading original image to get its dimensions
    original_image=cv2.imread(img_path)
    if original_image is None:
        print(f"Failed to load original image from {img_path}")
        return
    original_height,original_width, _ = original_image.shape
    
    #process each detected person
    for idx, bbox in enumerate(person_boxes):
        print(f"\nProcessing person {idx+1}...")
        try:
            cropped_img, top_left =crop_image(img_path, bbox)
        except FileNotFoundError as e:
            print(e)
            continue
        
        try:
            landmarks, pose_landmarks=estimate_pose(cropped_img)
            print("Pose landmarks estimated successfully.")
        except ValueError as e:
            print(e)
            continue
        
        #mapping landmarks back to original image coordinates
        mapped_landmarks=map_landmarks_to_original(landmarks, bbox, original_image.shape)
        
        #visualizing landmarks on the original image
        if output_dir:
            output_img_path = os.path.join(output_dir, f"pose_person_{idx+1}.jpg")
        else:
            output_img_path = None
        try:
            visualize_landmarks(img_path, mapped_landmarks, output_img_path)
        except FileNotFoundError as e:
            print(e)
            continue
        
        #landmarking coordinates
        print(f"3D Landmark Coordinates for person {idx+1}:")
        for lm_idx, coord in enumerate(mapped_landmarks):
            print(f"  Landmark {lm_idx}: (x: {coord[0]}, y: {coord[1]}, z: {coord[2]})")

if __name__ == "__main__":
    img_path ='C:\\Users\\outco\\Desktop\\i191742_Assignment02_DL_Perception\\Object Detection using YOLO\\2.jpg'  #image path
    output_directory = 'pose_estimation_results' # Output directory
    confidence_threshold=0.5  # Hyper-Parameter
    #calling function
    main(img_path,output_directory,confidence_threshold)
