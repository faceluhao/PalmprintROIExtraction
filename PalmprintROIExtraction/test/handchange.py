#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import cv2
import mediapipe as mp
# import time
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_utils.DrawingSpec
mp_hands = mp.solutions.hands


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a camera, use 'continue' instead of 'break'.
      break

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles(),
            mp_drawing_styles())
        points = [
            int(hand_landmarks.landmark[i].z * 100)
            for i in range(len(hand_landmarks.landmark))
        ]
        pointsx = [
            int(hand_landmarks.landmark[i].x * 100)
            for i in range(len(hand_landmarks.landmark))
        ]
        pointsy = [
            int(hand_landmarks.landmark[i].y * 100)
            for i in range(len(hand_landmarks.landmark))
        ]
        ##判断手掌前后左右倾斜，
        sum_check_less = sum(i < 0 for i in points)
        sum_check_more = sum(i > 0 for i in points)
        check_front = points[1] + points[2]+points[3]+points[4]+points[5]+points[6]+points[7]+points[8]
        check_after = points[14] + points[15]+points[16]+points[17]+points[18]+points[19]+points[20]+points[13]
        check_rightleft = check_front - check_after
        if sum_check_less > 15:
            print("手掌前倾")
            cv2.putText(image,"front",(100,300), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,255,255), 3, cv2.LINE_AA)
        elif sum_check_more > 15:
            print("手掌后倾")
            cv2.putText(image,"behind",(100,300), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,255,255), 3, cv2.LINE_AA)
        elif check_rightleft < -20:
            print("手掌左前倾")
            cv2.putText(image,"left",(100,300), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,255,255), 3, cv2.LINE_AA)
        elif check_rightleft > 20:
            print("手掌右前倾")
            cv2.putText(image,"right",(100,300), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,255,255), 3, cv2.LINE_AA)

        # points = [
        #     hand_landmarks.landmark[i].z * 100
        #     for i in range(len(hand_landmarks.landmark))
        # ]
        # x_max = -100000
        # x_min = 100000
        # for point in points[1:]:
        #     if point > x_max:
        #         x_max = point
        #     if point < x_min:
        #         x_min = point
        print(points)
        print(pointsx)
        print(pointsy)
        # print(np.var(points))
        # print(x_max - x_min)


    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
