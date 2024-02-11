import torch
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pyautogui
import keyboard
import time
import autoit

####Setup environment (make sure you have all necessary libraries installed)
yolov5_repo_path = r'C:\Users\Username\yolov5'

#### Load Yolov5 model OR load 'custom' model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#model_path = r'yolov5\weights\yolo_weights.pt'
#model = torch.hub.load(yolov5_repo_path, 'custom', path=model_path, source='local')

screenshot_number = 0
pyautogui.PAUSE = 0.01

save_image_path = f'C:\\Users\\Username\yolov5\\data\\images\\screenshot_{screenshot_number}.png'

while True:

    # Mouse automation

    if keyboard.is_pressed("shift"):
        screenshot = pyautogui.screenshot(region=(1280,0,2560,1440))
        frame_array = np.array(screenshot)
        frame = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        results = model(frame)
        # Extracting coordinates of detected objects and print them
        detected_objects = results.xyxy[0]

        # Filter the data to make loop faster -> Filter for class 0 and confidence > 0.4
        filtered_objects = detected_objects[(detected_objects[:, 5] == 0) & (detected_objects[:, 4] > 0.4)]

        if len(filtered_objects) > 0:
            highest_confidence_object = filtered_objects[filtered_objects[:, 4].argmax()]
            x1, y1, x2, y2, conf, _ = highest_confidence_object

        try:
            # Increments of Y goes down
            # Increments of X goes right
            # Calculating positions x1,y1 -> top left corner , x2,y2 bottom right corner
            mouse_x = 1280 + int(x1 + (x2-x1)/2)  # screen width 5120/4 = 1280
            mouse_y = int(y1 + 10) # looking for headshot
            speed = int(1) # number 1 instant , number 100 very slow

            autoit.mouse_move(mouse_x, mouse_y, speed)
        except:
            print("No enemy..")
    
    # Save predict picture with retangles and confidence
        
    if keyboard.is_pressed("o"):
        screenshot = pyautogui.screenshot(region=(1280,0,2560,1440))
        frame_array = np.array(screenshot)
        frame = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        results = model(frame)
        # Extracting coordinates of detected objects and print them
        detected_objects = results.xyxy[0]

        # Filter the data to make loop faster -> Filter for class 0 and confidence > 0.4
        filtered_objects = detected_objects[(detected_objects[:, 5] == 0) & (detected_objects[:, 4] > 0.4)]
        
        for obj in filtered_objects:
            x1, y1, x2, y2, conf, _ = obj

            # Draw bounding boxes and labels on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'Confidence: {conf:.2f}', (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ###### SAVING SCREENSHOT#####
        # Generate a filename and check if it already exists
        while os.path.exists(save_image_path) == True:
            screenshot_number += 1
        # Create a filename with the unique number
        filename = f'screenshot_{screenshot_number}.png'
        # Save the screenshot with predictions
        cv2.imwrite(filename, frame)
        ##############################

    # Stop the loop

    if keyboard.is_pressed("l"):
        print("Exiting..")
        break

    # Loop sleep time
    time.sleep(0.01)

