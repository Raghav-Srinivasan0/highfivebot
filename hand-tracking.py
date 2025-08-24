# Wifi: RoArm-M2
# Passwd: 12345678
# IP: 192.168.4.1

import cv2
import mediapipe as mp
import threading
import numpy as np
from matplotlib import pyplot as plt
import requests
import keyboard
import time
import json

leftcapnum = 0
rightcapnum = 1

capleft = cv2.VideoCapture(leftcapnum)
capright = cv2.VideoCapture(rightcapnum)

handX_left, handY_left = 0,0
handX_right, handY_right = 0,0
distance = 0

robotIP = "192.168.4.1"

robotEnabled = True

robothomeX = 100
robothomeY = 0
robothomeZ = 100

robotYoffset = 50
robotZoffset = -200
robotXoffset = -32

robotX = 100
robotY = 0
robotZ = 100

ret, frame = capleft.read()

cx=frame.shape[1]/2
cy=frame.shape[0]/2

print(f"cx={cx}, cy={cy}")

def depth(camidleft, camidright):
    global handX_left
    global handX_right
    global distance
    f = 2.1
    d = 0.0014
    T = 65
    while True:
        D = handX_left-handX_right
        if D == 0:
            continue
        distance = (f/d) * (T/D)
        distance_scalar = 0.2
        distance *= -distance_scalar
        # print(f"Hand distance is {distance}")
        if cv2.waitKey(10) == ord('q'):
            break

def highfive():
    global robotX
    global robotY
    global robotZ
    global robotYoffset
    global robotZoffset
    global robotXoffset
    global robothomeX
    global robothomeY
    global robothomeZ
    global robotIP
    global robotEnabled
    print('Ready to high five!')
    while True:
        time.sleep(5)
        if robotEnabled and keyboard.is_pressed('g'):
            command = f"{{\"T\":104,\"x\":{robotX+robotXoffset},\"y\":{robotY+robotYoffset},\"z\":{robotZ+robotZoffset},\"t\":3.14,\"spd\":0.25}}"
            # print(command)
            url = "http://" + robotIP + "/js?json=" + command
            response = requests.get(url)
            content = response.text
            command = f"{{\"T\":105}}"
            # print(command)
            url = "http://" + robotIP + "/js?json=" + command
            response = requests.get(url)
            content = response.text
            json_obj = json.loads(content)
            newtilt = -float(json_obj["e"])-float(json_obj["s"])+((float(3)*np.pi)/float(2))
            command = f"{{\"T\":101,\"joint\":4,\"rad\":{newtilt},\"spd\":0,\"acc\":10}}"
            # print(command)
            url = "http://" + robotIP + "/js?json=" + command
            response = requests.get(url)
            content = response.text
            time.sleep(2)
            command = f"{{\"T\":104,\"x\":{robotX+robotXoffset+50},\"y\":{robotY+robotYoffset},\"z\":{robotZ+robotZoffset},\"t\":{newtilt},\"spd\":0.25}}"
            # print(command)
            url = "http://" + robotIP + "/js?json=" + command
            response = requests.get(url)
            content = response.text
            time.sleep(1)
            command = f"{{\"T\":104,\"x\":{robotX+robotXoffset-20},\"y\":{robotY+robotYoffset},\"z\":{robotZ+robotZoffset},\"t\":{newtilt},\"spd\":0.25}}"
            # print(command)
            url = "http://" + robotIP + "/js?json=" + command
            response = requests.get(url)
            content = response.text
            time.sleep(2)
            command = f"{{\"T\":104,\"x\":{robothomeX},\"y\":{robothomeY},\"z\":{robothomeZ},\"t\":3.14,\"spd\":0.25}}"
            url = "http://" + robotIP + "/js?json=" + command
            response = requests.get(url)
            content = response.text

def getXYZ():
    global handX_left
    global handY_left
    global distance
    global cx
    global cy
    global robotX
    global robotY
    global robotZ
    fx=2.1
    fy=2.1
    X_offset = 10500
    Y_offset = 8500
    scalar = 0.01*(5/8)
    scalarY = 0.01*(5/8)
    while True:
        X = distance * (handX_left-cx)/fx
        Y = distance * (handY_left-cy)/fy
        X *= scalar
        Y *= -scalarY
        Z = distance
        robotX = Z
        robotY = X
        robotZ = Y
        print(f"({int(robotX)},{int(robotY)},{int(robotZ)})")

def tracking(camid):
    global handX_left
    global handY_left
    global handX_right
    global handY_right
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = None
    if camid == leftcapnum:
        cap = capleft
    elif camid == rightcapnum:
        cap = capright

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a more natural mirror-like view
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and get hand landmarks
            results = hands.process(image)

            # Draw the hand landmarks on the image
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Calculate and print center of hand
                    x, y = 0, 0
                    for lm in hand_landmarks.landmark:
                        x += lm.x * frame.shape[1]
                        y += lm.y * frame.shape[0]
                    x /= len(hand_landmarks.landmark)
                    y /= len(hand_landmarks.landmark)
                    # print("Center of hand:", int(x), int(y))
                    if camid == leftcapnum:
                        handX_left = x
                        handY_left = y
                    elif camid == rightcapnum:
                        handX_right = x
                        handY_right = y

            # Show the image
            # cv2.imshow(f'Hand Tracking {camid}', frame)
            if cv2.waitKey(10) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
t1 = threading.Thread(target=tracking,args=(leftcapnum,))
t2 = threading.Thread(target=tracking,args=(rightcapnum,))
t3 = threading.Thread(target=depth,args=(leftcapnum,rightcapnum,))
t4 = threading.Thread(target=getXYZ)
t5 = threading.Thread(target=highfive)

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()

print("Done!")