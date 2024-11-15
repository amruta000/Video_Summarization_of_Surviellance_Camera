# Video Summarization of Surviellance Camera

This project implements an automated video summarization system for surveillance camera footage using the MobileNet SSD (Single Shot Multibox Detector) deep learning model. The goal is to reduce the amount of video data to review by identifying and retaining key events, such as the presence of people or vehicles, while filtering out redundant information. This summarization process is achieved by detecting relevant objects in video frames and creating a concise video summary based on detected activity.

## Features
- **Real-Time Object Detection:** Uses MobileNet SSD to detect people, vehicles, and other objects of interest in each video frame.
- **Efficient Video Summarization:** Automatically generates video summaries by selecting frames that contain significant events.
- **Activity-Based Filtering:** Only frames with detected objects of interest (e.g., people, vehicles) are included in the summary.
- **Lightweight and Efficient:** The solution is designed for real-time processing on devices with limited resources.

## Technologies Used
- **MobileNet SSD:** A lightweight convolutional neural network model for object detection.
- **OpenCV:** A library for computer vision tasks, such as reading video files, processing frames, and drawing bounding boxes.
- **NumPy:** A library for handling numerical operations and data manipulation.
- **Python 3:** The programming language used for implementing the solution.

## Installation
Prerequisites
- Python 3.x
- OpenCV
- NumPy
- TensorFlow (for MobileNet SSD model support)
  
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

  
