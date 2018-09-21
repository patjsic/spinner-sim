#!/usr/bin/python
from os import *
import cv2
import numpy as np
import sys
import argparse
import time


filname = input("Insert file name: ")
video_title = input("Insert video title (include the .mp4): ")
#Make directory to store frames
mkdir("analysis-" + filname)

#Cut video using OpenCV2

def vid_to_frame(input, output):
    vidcap = cv2.VideoCapture(str(video_title))
    vid_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    count = 0

    print("Making frames...\n")

    #Keeps track of time for fps
    start_time = time.time()

    #Starts cutting the frames
    while vidcap.isOpened():
        r, frame = vidcap.read()
        cv2.imwrite(output + "/%#05d.jpg" % (count+1), frame)
        count += 1

        if (count >= (vid_length - 1)):
            print("Done making frames.")

            #Checks end time
            end_time = time.time()
            vidcap.release()
            break

    #Calculate the FPS
    time_taken = end_time - start_time
    print('Time taken is: ' + str(time_taken) + 'seconds')
    fps = vid_length / time_taken
    print('fps is : ' + str(fps))

    return fps, vid_length

#input = "C:\\Users\\Patrick\\Documents\\Research\\Spinners\\5min.mp4"
output = "analysis-" + filname
fps, nframes = vid_to_frame(video_title, output)
print(nframes)
print('Now running time-corr.py')
system('python time-corr.py -f' + str(filname) + '-nframes' +  str(nframes) + '-fps' + str(fps))
