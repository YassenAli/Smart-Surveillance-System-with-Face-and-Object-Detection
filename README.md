# Smart Surveillance System with Face and Object Detection

Smart Surveillance System is a real-time video processing application built with OpenCV and Python. It combines face detection, human pose estimation, and object tracking to create an intelligent surveillance system.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Model Files](#model-files)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Features

- **Face Detection**: Detects faces in video frames using a pre-trained Caffe model.
- **Pose Estimation**: Estimates human poses using a deep learning model (OpenPose).
- **Object Tracking**: Tracks selected objects with the CSRT tracker.
- **Real-Time Video Processing**: Supports input from a webcam or video file.
- **User Controls**: Pause/resume, fast-forward, and rewind video playback.

## Project Structure

```
Smart-Surveillance-System-with-Face-and-Object-Detection/
├── FaceDetection.py
├── PoseEstimation.py
├── TrackingObject.py
├── main.py
└── models/
    ├── deploy.prototxt
    ├── res10_300x300_ssd_iter_140000_fp16.caffemodel
    ├── pose_deploy_linevec_faster_4_stages.prototxt
    ├── pose_iter_160000.caffemodel
    └── ssd_mobilenet_v2_coco_2018_03_29.pbtxt
```

## Requirements

- Python 3.x
- OpenCV (both `opencv-python` and `opencv-contrib-python`)
- NumPy
- Matplotlib (for visualization, if needed)

Install the required packages using:

```bash
pip install opencv-python opencv-contrib-python numpy matplotlib
```

## Model Files

This project uses pre-trained models for face detection and pose estimation. **These model files are not stored in the repository** due to their size. Instead, they are available as a release asset.

**Download the models:**
1. Go to the [Releases](https://github.com/yourusername/Smart-Surveillance-System-with-Face-and-Object-Detection/releases) page.
2. Download the `models.zip` file from the latest release.
3. Extract the contents of `models.zip` into a folder named `models` in the project root.

The `models` folder should contain the following files:
- **Face Detection Model:**
  - `deploy.prototxt`
  - `res10_300x300_ssd_iter_140000_fp16.caffemodel`
- **Pose Estimation Model:**
  - `pose_deploy_linevec_faster_4_stages.prototxt`
  - `pose_iter_160000.caffemodel`
- **Additional Model (if used):**
  - `ssd_mobilenet_v2_coco_2018_03_29.pbtxt`

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/Smart-Surveillance-System-with-Face-and-Object-Detection.git
   cd Smart-Surveillance-System-with-Face-and-Object-Detection
   ```

2. **Verify/Download Model Files:**

   Ensure that the `models` folder contains the required model files. If not, download them from the following sources:
   - [Face Detection Deploy Prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
   - [Face Detection Caffe Model](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel)
   - [Pose Estimation Prototxt](https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/pose_deploy_linevec_faster_4_stages.prototxt)
   - [Pose Estimation Caffe Model](https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/pose_iter_160000.caffemodel)

3. **Install Python Dependencies:**

   ```bash
   pip install opencv-python opencv-contrib-python numpy matplotlib
   ```

## Usage

1. **Run the Application:**

   You can run the project using:

   ```bash
   python main.py
   ```

2. **Video Source Selection:**

   - The application will prompt you to choose between using a webcam or a video file.
   - If you select a video file, ensure you provide the correct file path.

3. **Object Tracking Setup:**

   - On startup, you'll be prompted to select a region of interest (ROI) for object tracking.
   - Use your mouse to draw a rectangle around the object you wish to track.

4. **Keyboard Controls:**

   - **ESC or Q**: Quit the application.
   - **Space**: Pause/resume video playback.
   - **F**: Fast forward (jump ahead 100 frames).
   - **B**: Rewind (jump back 100 frames).

## Acknowledgements

- [OpenCV](https://opencv.org/) for the computer vision libraries.
- Pre-trained models provided by the OpenCV community.
- Inspiration from various computer vision and deep learning tutorials.

---

Happy coding! If you encounter any issues or have suggestions, please feel free to open an issue or submit a pull request.
