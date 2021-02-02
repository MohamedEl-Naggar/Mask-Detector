from cv2 import cv2 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import time
import os

def detect_predict(frame, face_detector, mask_detector):

    (h,w) = frame.shape[:2]
    # Input blobs from image to facial detection model
    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104.0, 177.0, 123.0)) 
    face_detector.setInput(blob) 
    detections = face_detector.forward()

    faces = []
    locations = []
    predictions = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.9: 
            box = detections[0, 0, i, 3:7]*np.array([w,h,w,h])
            # increase box size by 20%
            X_size = box[2]-box[0]
            Y_size = box[3]-box[1]
            box[0] = box[0] - 0.2*X_size
            box[1] = box[1] - 0.2*Y_size
            box[2] = box[2] + 0.2*X_size
            box[3] = box[3] + 0.2*Y_size

            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Crop face from image and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locations.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype = "float32")
        predictions = mask_detector.predict(faces, batch_size = 32)

    return (locations, predictions)

# Load models
mask_detector = load_model("mask_detector_model")
face_detector = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Globals
classes = np.array(("Incorrect_Mask", "Mask", "No_Mask"))
c = [(0,255,255), (0,255,0), (0,0,255)]
colors = np.empty(len(c), dtype=object)
colors[:] = c

# Start video streaming
vid = cv2.VideoCapture(0)
time.sleep(2) 

while(True):
    ret,frame = vid.read()
    (boxes, predictions) = detect_predict(frame, face_detector, mask_detector)

    for (box, pred) in zip(boxes, predictions): # loop over predictions with their respective boxes
        (startX, startY, endX, endY) = box # box coordinates

        index = np.argmax(pred) # index of predicted class
        confidence = pred[index] * 100
        classified = classes[index]
        color = colors[index]

        # Format label and place it on the frame above the bounding box
        label = "{}: {:.2f}%".format(classified, confidence) 
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to quit
        break

cv2.destroyAllWindows()