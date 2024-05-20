import os
import json
import cv2
import keyboard
from ultralytics import YOLO
import torch

model = YOLO('./model/yolov8n-pose.pt')
# model.to('cuda')
# model = YOLO('./model/yolov8x-pose-p6.pt')

# 비디오 파일을 읽어옵니다.
video_path = os.path.abspath("./sample/7-4_cam02_assault01_place04_day_summer.mp4")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow('Results', cv2.WINDOW_NORMAL)

frame_data = []

while cap.isOpened():
	ret, frame = cap.read()
	
	if ret:
		results = model(frame)
		data = {
			"frame_id": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
			"box": results[0].boxes.xyxy.numpy().tolist(),
			"keypoints": results[0].keypoints_xy.xy.numpy().tolist(),
		}
		frame_data.append(data)
		
		img = results[0].plot()
		cv2.imshow("Results", img)
	else:
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

# 프로그램 종료 전에 frame_data를 JSON 파일로 저장합니다.
with open('frame_data.json', 'w') as f:
	json.dump(frame_data, f)

cv2.destroyAllWindows()
cap.release()

