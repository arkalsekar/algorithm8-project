import cv2
import mediapipe as mp
import time
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np 
import math
import random

class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    def draw(self, img):
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                      (225, 225, 225), cv2.FILLED)
        # cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
        #               (50, 50, 50), 3)
        cv2.putText(img, self.value, (self.pos[0] + 30, self.pos[1] + 70), cv2.FONT_HERSHEY_PLAIN,
                    2, (50, 50, 50), 2)

   
    def checkClick(self, x, y, img):
        if self.pos[0] < x < self.pos[0] + self.width and \
                self.pos[1] < y < self.pos[1] + self.height:
            cv2.rectangle(img, (self.pos[0] + 3, self.pos[1] + 3),
                          (self.pos[0] + self.width - 3, self.pos[1] + self.height - 3),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(img, self.value, (self.pos[0] + 25, self.pos[1] + 80), cv2.FONT_HERSHEY_PLAIN,
                        5, (0, 0, 0), 5)
            return True
        else:
            return False

def draw_rectangle(image, x, y, width, height):
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), -1)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# btn = Button([10, 10], 20, 20, "Play")
# btn1 = Button([20, 10], 20, 20, "Pause")
# btn2 = Button([30, 10], 20, 20, "Work")

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
    
    # Extracting Hand Landmarks out of provided landmarks
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:            

            indexX, indexY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight
            middleX, middleY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_hight
            ThX, ThY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_hight

            # thumb and index coordinates
            indexCor = int(indexX), int(indexY)
            middleCor  = int(middleX), int(middleY)
            ThCor = int(ThX), int(ThY)

            cv2.circle(image, tuple(indexCor), 20, (0, 0, 256), -1)

            length_btw_index_middle = math.hypot(middleX - indexX, middleY - indexY)
            length_btw_index_thumb = math.hypot(ThX - indexX, ThY - indexY)

            cursorPosX = np.interp(indexX, [50, image_width - 50], [0, 1920])
            cursorPosY = np.interp(indexY, [50, image_hight - 50], [0, 1080])
            # print("Cursor is Currently at ", cursorPosX, cursorPosY)


            if (indexCor and middleCor and ThCor) is not None:
                # cv2.line(image, indexCor, middleCor, (0, 0, 256), 2)
                # cv2.line(image, indexCor, ThCor, (0, 0, 256), 2)
                # cv2.circle(image, tuple(middleCor), 20, (0, 0, 256), -1)
                cv2.circle(image, tuple(ThCor), 20, (0, 0, 256), -1)
                print(length_btw_index_thumb)
                if length_btw_index_thumb < 28:
                    print("Key Pressend down")
                    pyautogui.press('up')

                # cv2.line(image, ThCor, middleCor, (0, 0, 256), 2)


    
    cv2.putText(image, "Virtual Controller", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (256, 0, 0), 2)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
