import cv2
import numpy as np
import os

# net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")

# if the above line not work with you try the following lines

model_path = r"Y:\Yassin Projects\Smart-Surveillance-System-with-Face-and-Object-Detection\models"
net = cv2.dnn.readNetFromCaffe(
    os.path.join(model_path, "deploy.prototxt"),
    os.path.join(model_path, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
)

def detectFace(frame):
    h, w = frame.shape[:2]
    dim = (300, 300)
    confThreshold = 0.5

    blob = cv2.dnn.blobFromImage(frame, 1.0, dim, (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detectionFaces = net.forward()

    for i in range(detectionFaces.shape[2]):
        confidenceScore = detectionFaces[0, 0, i, 2]
        if confidenceScore > confThreshold:
            box = detectionFaces[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            label = "Face: {:.2f}%".format(confidenceScore * 100)
            labelSize, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x1, y1 - labelSize[1]), (x1 + labelSize[0], y1 + baseline), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
