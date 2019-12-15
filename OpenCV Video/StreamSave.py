# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:53:30 2019

@author: Anuj
"""

import cv2

if __name__ == "__main__":
    # find the webcam
    capture = cv2.VideoCapture(0)

    # video recorder
    # fourcc = cv2.VideoWriter_fourcc(*'XVID') <<--This is for .avi format  
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    videoOut = cv2.VideoWriter("output.mp4", fourcc, 10.0, (640, 480))

    # record video
    while (capture.isOpened()):
        ret, frame = capture.read()
        if ret:
            videoOut.write(frame)
            cv2.imshow('Video Stream', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        else:
            break

        # Tiny Pause
        key = cv2.waitKey(1)

    capture.release()
    videoOut.release()
    cv2.destroyAllWindows()