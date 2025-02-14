import cv2
import numpy as np

initBB = None
tracker = None

def trackObject(frame):
    """
    Updates the object tracker with the current frame and draws the tracking bounding box.

    The function uses global variables 'initBB' (initial bounding box) and 'tracker'.
    If 'tracker' is not initialized, it creates a CSRT tracker.
    If tracking is successful, a bounding box is drawn; otherwise, a "Tracking Failed" message is shown.

    Parameters:
        frame (np.ndarray): The current video frame.

    Returns:
        frame (np.ndarray): Frame annotated with the tracking result.
    """
    global initBB, tracker

    if tracker is None:
        tracker = cv2.TrackerCSRT_create()

    if initBB is not None:
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking Failed", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No object selected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame
