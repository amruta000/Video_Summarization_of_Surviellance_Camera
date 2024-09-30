from math import ceil  # pip install python-math  returns round nearest number
import cv2  # pip install opencv-python
import numpy as np  # pip install numpy
###############################

CAM = 0

print('SELECT OPTION FROM BELOW')

print('1. sample video')

response = int(input())

# default resourcs
res = CAM

if response == 1:
    print("Enter Video file name [file sould be in same directory]")
    path = input()
    res = path
else:
    print('Not proper input')

print('click \'Q\' to quit')

######## video input module #########


video = cv2.VideoCapture(res)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
threshold = 20
fps = int(video.get(cv2.CAP_PROP_FPS))


# print(width, height) # for current video 640 X 360
fourcc = 0x7634706d   # sequence of 4 byte code and uniquely identify data formats
writer = cv2.VideoWriter('final.MP4', fourcc, fps, (width, height))

ret, first_frame = video.read()  # first frame form input video

prev_frame = first_frame

unique_frames = 0
common_frames = 0
total_frames = 0

 
#MobileNetSSD object detection model pre-trained using the Caffe framework. 

protopath = 'MobileNetSSD_deploy.prototxt'
modelpath = 'MobileNetSSD_deploy.caffemodel'

# prototxt   -> path to the .prototxt file with text description of the network architecture.
# caffeModel ->	path to the .caffemodel file with learned network.


# Caffe is a deep learning framework made with expression, speed, and modularity in mind.

# loading of caffe model and  used to load pre-trained Caffe models and accepts two arguments

detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

frame_number = 1

importantFrameTimeStampArr = []
# The time at which the frame was captured

while True:
    ret, frame = video.read()

    if not ret:
        break

    ##(H, W) = frame.shape[:2]

    # convert image into binary large object
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # setting input
    detector.setInput(blob)

    # perform classification

    person_detections = detector.forward()

    # returns probability of background or any object
    # print(person_detections.shape[2])

    # to keep track of number of frames
    # print(frame_number)
    frame_number += 1

    for i in np.arange(0, person_detections.shape[2]):

        confidence = person_detections[0, 0, i, 2]
        # print(confidence)

        if confidence > 0.5:

            # class label (id)
            idx = int(person_detections[0, 0, i, 1])

            # if object found is not person then continue
            if CLASSES[idx] != "person":
                continue

            # object found is person but not moving then
            # difference detection
            diff = (np.sum(np.absolute(frame - prev_frame)) / np.size(frame))

            #print('threshold=' +str(ceil(diff)))

            if ((ceil(diff) > threshold)):
                writer.write(frame)
                prev_frame = frame
                unique_frames += 1

                ###  getting timestamp   ###

                sec = int(video.get(cv2.CAP_PROP_POS_MSEC)/1000)
                # return duration of frame on which sec it is
                #print('timeStamp' +str(sec))

                importantFrameTimeStampArr.append(sec)

                # skipping non-moving frame

            else:
                prev_frame = frame
                common_frames += 1
        # print('no person')

    cv2.imshow("Application", frame)
    total_frames += 1

    key = cv2.waitKey(1)
    print('Processing...')

    if key == ord('q'):
        break

print("Total frames: ", total_frames)
print("Unique frames: ", unique_frames)
print("Common frames: ", common_frames)
print('done')
video.release()
writer.release()

finalArr = [importantFrameTimeStampArr[0]]

for i in range(1, len(importantFrameTimeStampArr)):
    finalArr.append(importantFrameTimeStampArr[i])

print(finalArr)

timeStamps = []

k = 0
i = 0
while i < (len(finalArr)):
    clip = []
    clip.append(finalArr[i])
    k = i+1
    while (k < len(finalArr)) and (finalArr[k] == finalArr[k-1]+1):
        k += 1
    i = k
    clip.append(finalArr[k-1])
    timeStamps.append(clip)

print(timeStamps)

########### Output####################

print("Summarized Video is Created")
print("Playing Output Video")

out_video = cv2.VideoCapture('output.mp4')

while True:

    ret, frame = out_video.read()

    if not ret:
        break
    cv2.imshow('Summarized output video', frame)

    if cv2.waitKey(1) == 27:
        break

out_video.release()

print('Output video file name is output.mp4')

cv2.destroyAllWindows()
