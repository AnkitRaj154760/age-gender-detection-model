# Age & Gender Detection AI Tool

A real-time computer vision application that detects faces via your webcam and predicts the age and gender of each detected person.

## Features
- **Real-time Face Detection**: Uses an OpenCV TensorFlow-based deep neural network to accurately detect faces in real-time.
- **Gender Prediction**: Uses a pre-trained Caffe model to classify gender as Male or Female.
- **Age Prediction**: Uses a pre-trained Caffe model to predict age within one of 8 specific ranges: `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, and `(60-100)`.

## Prerequisites

To run this project, you need Python installed on your system along with the OpenCV library.

```bash
pip install opencv-python
```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AnkitRaj154760/age-gender-detection-model.git
   cd age-gender-detection-model
   ```

2. **Verify Models are Present:**
   The `ML_Models/` directory should contain the pre-trained models required to run the script:
   - **Age Models:** `age_net.caffemodel` & `age_deploy.prototxt`
   - **Gender Models:** `gender_net.caffemodel` & `gender_deploy.prototxt`
   - **Face Models:** `opencv_face_detector_uint8.pb` & `opencv_face_detector.pbtxt`

## Usage

Run the main Python script from your terminal:

```bash
python Face_gender_detect.py
```

A window will open displaying your webcam feed. The script will draw a bounding box around any detected faces and overlay the predicted gender and age. 

**To stop the script and close the window, simply press the `q` key on your keyboard.**

## How it Works
1. The script captures a video stream from your default webcam.
2. It processes each frame through a deep learning face detector (SSD algorithm).
3. Once a face is detected with high confidence, it extracts the Region of Interest (ROI).
4. The face ROI is passed sequentially into the gender and age prediction neural networks.
5. The final predictions are rendered as text overlaid on the video frame.
