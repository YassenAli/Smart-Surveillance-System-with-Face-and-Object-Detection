import cv2
import numpy as np

nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

protoFile = "models/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "models/pose_iter_160000.caffemodel"
netPose = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

def poseEstimation(frame):
    """
    Performs pose estimation on the input frame, detecting keypoints and optionally drawing skeleton lines.

    Parameters:
        frame (np.ndarray): Input image/frame (BGR format).

    Returns:
        frame (np.ndarray): Output image with detected keypoints (and skeleton lines) drawn.
    """
    h, w = frame.shape[:2]
    netInputSize = (368, 368)
    threshold = 0.1

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=False, crop=False)
    netPose.setInput(inpBlob)
    output = netPose.forward()

    points = []
    for i in range(nPoints):
        probMap = output[0, int(i), :, :]
        # displayMap = cv2.resize(probMap, (w, h)) # Optionally, resize the probability map for visualization:
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        # Scale the point to the dimensions of the original image.
        x, y = int(w * point[0] / output.shape[3]), int(h * point[1] / output.shape[2])

        if prob > threshold:
            cv2.circle(frame, (x, y), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frame, "{}".format(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            points.append((x, y))
        else:
            points.append(None)

    for pair in POSE_PAIRS:
        partA, partB = pair
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3)

    return frame
