import sys
import cv2
import numpy as np
import time
import PoseEstimation
import FaceDetection
import TrackingObject

def main():
    # --- Welcome Message and Source Selection ---
    print("Welcome to Smart Surveillance System")
    print("Press 'w' to use webcam")
    print("Press 'v' to use video file")
    print("Press 'q' to quit")
    
    choice = input("Enter your choice (w/v): ").strip().lower()
    
    if choice == 'q':
        sys.exit("Exiting the system.")
    elif choice == 'w':
        source_index = 0
    elif choice == 'v':
        source_index = input("Enter the video file path: ").strip()
    else:
        sys.exit("Invalid choice. Exiting.")
    
    # --- Open the Video Source ---
    source = cv2.VideoCapture(source_index)
    if not source.isOpened():
        sys.exit("Error: Could not open video source.")
    
    # --- Create a Resizable Window ---
    WindowName = "Smart Surveillance System"
    cv2.namedWindow(WindowName, cv2.WINDOW_NORMAL)
    
    # --- Read the First Frame for Initialization ---
    ret, frame = source.read()
    if not ret:
        sys.exit("Error: Unable to read from video source.")
    
    # --- Select ROI for Object Tracking ---
    initBB = cv2.selectROI(WindowName, frame, fromCenter=False, showCrosshair=True)
    if initBB == (0, 0, 0, 0):
        sys.exit("No ROI selected. Exiting.")
    
    # --- Initialize the Tracker (Global Variables in TrackingObject) ---
    TrackingObject.initBB = initBB
    TrackingObject.tracker = cv2.TrackerCSRT_create()
    TrackingObject.tracker.init(frame, TrackingObject.initBB)
    
    # --- FPS Calculation Setup ---
    prev_time = time.time()
    
    # --- Main Loop for Processing Frames ---
    while True:
        ret, frame = source.read()
        if not ret:
            print("End of video or unable to fetch frame.")
            break
        
        # --- Process the frame with different modules ---
        frame = PoseEstimation.poseEstimation(frame)
        frame = FaceDetection.detectFace(frame)
        frame = TrackingObject.trackObject(frame)
        
        # --- Calculate and Overlay FPS ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # --- Display the Processed Frame ---
        cv2.imshow(WindowName, frame)
        
        # --- Handle Keyboard Events ---
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q' to quit
            break
        elif key == ord(' '):  # Pause/Play with space bar
            print("Paused. Press space to resume.")
            while True:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord(' '):
                    break
        elif key == ord('f'):  # Fast forward
            current_frame = source.get(cv2.CAP_PROP_POS_FRAMES)
            source.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 100)
        elif key == ord('b'):  # Rewind
            current_frame = source.get(cv2.CAP_PROP_POS_FRAMES)
            new_frame = max(current_frame - 100, 0)
            source.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
    
    # --- Release Resources ---
    source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
