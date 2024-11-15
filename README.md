# Video Summarization of Surviellance Camera by using MobileNet.SSD

This project implements an automated video summarization system for surveillance camera footage using the MobileNet SSD (Single Shot Multibox Detector) deep learning model. The goal is to reduce the amount of video data to review by identifying and retaining key events, such as the presence of people or vehicles, while filtering out redundant information. This summarization process is achieved by detecting relevant objects in video frames and creating a concise video summary based on detected activity.

## Features
- **Real-Time Object Detection:** Uses MobileNet SSD to detect people, vehicles, and other objects of interest in each video frame.
  
- **Motion Detection:** Identifies significant motion between frames to detect events, such as a person entering or exiting the scene.
  
- **Timestamping:** Tracks the timestamps of key frames for analysis and organization.
  
- **Efficient Processing:** The system processes frames only when necessary (i.e., when significant movement is detected), reducing redundant frames in the output.
  
- **Summarized Video:** Automatically generates a video summary with only the most relevant frames (e.g., frames with detected movement).
  
- **Activity-Based Filtering:** Only frames with detected objects of interest (e.g., people, vehicles) are included in the summary.
  
- **Lightweight and Efficient:** The solution is designed for real-time processing on devices with limited resources.

## Technologies Used
- **MobileNet SSD:** A lightweight convolutional neural network model for object detection.
- **OpenCV:** A library for computer vision tasks, such as reading video files, processing frames, and drawing bounding boxes.
- **NumPy:** A library for handling numerical operations and data manipulation.
- **Caffe:**  A deep learning framework used to load the MobileNet SSD model.
- **Python 3:** The programming language used for implementing the solution.

## Prerequisites Software
- Python 3.x
- OpenCV
- NumPy
- Caffe (for MobileNet SSD model)

## Installation
1. Install required Python packages:
```bash
pip install opencv-python numpy
```
```bash
pip install opencv-python
```
2. Download MobileNet SSD Files:

You will need the following files for the MobileNet SSD object detection model:

  ```MobileNetSSD_deploy.prototxt``` (Network architecture file)

  ```MobileNetSSD_deploy.caffemodel``` (Pre-trained model weights)

Place these files in the project directory.


## How to Use
### Running the Script
1. Place your video file in the same directory as the script or provide its path when prompted.

2. Run the script:

```bash
python summarize_video.py
```

3. Follow the prompts:

- **Option 1:** Choose the input video file if you want to process a specific file.
- **Option 2:** You can also use a live camera feed by selecting the default camera (CAM = 0).

4. The script will:

- Detect people in the video using MobileNet SSD.
- Detect significant motion and store frames with moving objects.
- Generate a summarized video with key frames.
- Display the final summarized video and its timestamps.

**Video Summarization**
- After processing the video, the summarized video will be saved as final.MP4.
- A list of timestamps corresponding to significant frames will be output.
- The system will also display a preview of the summarized video after creation.
  
## Workflow Diagram 
**1. Input:** Surveillance video file.
  
**2. Frame Extraction:** Frames are extracted at regular intervals from surveillance video files using OpenCV.

**3. Object Detection:** Run each frame through MobileNet SSD to detect objects (e.g., people, cars).

**4. Event Filtering:** Identify frames where significant events happen (people or cars detected).

**5. Summarization:** Select key frames and compile them into a new video.

**6. Output:** A summarized video file with only the relevant frames.

## Configuration
You can adjust parameters such as:

- **Confidence Threshold:** The minimum confidence score for an object to be considered detected (default: 0.5).
- **Summary Length:** Control the frequency and number of frames selected for the summary.

  
