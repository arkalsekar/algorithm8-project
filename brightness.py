import cv2
import mediapipe as mp
import time
import pyautogui
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np 
import math

# Min Hand Distance , Max hand distance 15 , 55
# minimum vol , maximum = -65, 0 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# min and max vol
minVol = 0
maxVol = 100

vol = 0
volBar = 80

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    
    results = hands.process(image)
    image_hight, image_width, _ = image.shape
    
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    imgL = results.multi_hand_landmarks

    # Extracting Hand Landmarks out of provided landmarks
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:            

            indexX, indexY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight
            ThX, ThY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_hight
            
            # thumb and index coordinates
            ThCor = int(ThX), int(ThY)
            indexCor = int(indexX), int(indexY)
            
            cv2.circle(image, tuple(indexCor), 20, (0, 0, 256), -1)
            length = math.hypot(ThX - indexX, ThY - indexY)
            
            forbright = np.interp(length, [15, 155], [minVol, maxVol]) 
            print(forbright)

            # set brightness 
            sbc.set_brightness(int(forbright))
            
            vol = np.interp(length, [15, 155], [minVol, maxVol]) 
            volBar = np.interp(length, [3, 155], [380, 80]) 
            # print(vol)
            # if pyautogui.press('s'):
            #       time.sleep(3)

            # else:    
           
            if indexCor is not None:
                  cv2.line(image, indexCor, ThCor, (0, 0, 256), 2)


            cv2.circle(image, tuple(ThCor), 20, (0, 0, 256), -1)
    
    
    cv2.rectangle(image, (60, 380), (80, 80), (0, 256, 0), 2)
    cv2.rectangle(image, (60, int(380)), (80, int(volBar)), (0, 256, 0), -1)
    
    
    cv2.putText(image, "Brightness Controller", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (256, 0, 0), 2)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()