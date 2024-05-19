from collections import defaultdict
import cv2
from ultralytics import YOLO
import numpy as np
import os

# Load the YOLOv8 model
# model = YOLO("yolov8n-pose.pt")
model = YOLO("yolov8x-pose.pt")
# model = YOLO("yolov8x-pose-p6.pt")
model.to("cuda")

# Open the video file
video_path = os.path.abspath("../sample/7-4_cam02_assault01_place04_day_summer.mp4")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cv2.namedWindow('YOLOv8 Tracking', cv2.WINDOW_NORMAL)

# Store the track history
track_history = defaultdict(lambda: [])

visual = False
count = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    count += 1

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        if count % 30 != 0:
            continue

        # Get the boxes and track IDs
        print(results[0].boxes)
        boxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.id is None:
            track_ids = []
        else:
            track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

    key = cv2.waitKeyEx(1)  # waitKeyEx()를 사용하여 확장된 키 코드를 얻습니다.

    if key == 27:  # ESC 키
        break
    elif key == 2424832:  # 왼쪽 화살표 키
        frame_id = max(cap.get(cv2.CAP_PROP_POS_FRAMES) - fps * 5, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    elif key == 2555904:  # 오른쪽 화살표 키
        frame_id = min(cap.get(cv2.CAP_PROP_POS_FRAMES) + fps * 5, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    elif key == 32:  # 스페이스바
        while True:
            key = cv2.waitKeyEx(0)
            if key == 32 or key == 27:
                break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()