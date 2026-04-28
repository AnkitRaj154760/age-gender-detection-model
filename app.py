import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

# Load models (we cache them so they only load once, saving time)
@st.cache_resource
def load_models():
    model_folder = "ML_Models/"
    age_net = cv2.dnn.readNetFromCaffe(model_folder + 'age_deploy.prototxt', model_folder + 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe(model_folder + 'gender_deploy.prototxt', model_folder + 'gender_net.caffemodel')
    face_net = cv2.dnn.readNetFromTensorflow(model_folder + 'opencv_face_detector_uint8.pb', model_folder + 'opencv_face_detector.pbtxt')
    return age_net, gender_net, face_net

age_net, gender_net, face_net = load_models()

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# This function processes each video frame from the browser webcam
def video_frame_callback(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * [width, height, width, height]
            (x, y, x1, y1) = box.astype(int)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
            
            face_roi = img[y:y1, x:x1]
            if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                
                label = f"{gender}, {age}"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Age & Gender Detection AI")
st.write("Click 'START' to use your webcam and detect age and gender.")

webrtc_streamer(
    key="age-gender-detector", 
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun.services.mozilla.com"]},
            {"urls": ["stun:global.stun.twilio.com:3478"]}
        ]
    }
)
