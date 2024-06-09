import cv2
import numpy as np
from ultralytics import YOLO
import torch
import joblib
from bson_to_data import get_bson_for_frame

# YOLOv8 모델 로드
model_name = "yolov8x-pose-p6"
model = YOLO(f"{model_name}.pt")
model.to("cuda")

# 스켈레톤 및 바운딩 박스 표시 색상과 점 크기 설정
thickness = 2

# 비디오 파일 경로
video_path = "1.mp4"

# 비디오 파일 초기화
cap = cv2.VideoCapture(video_path)


def get_color_from_id(track_id):
	np.random.seed(track_id)
	color = np.random.randint(0, 255, 3).tolist()
	return (color[0], color[1], color[2])


label_encoder = joblib.load('./random/label_encoder.joblib')

# 저장된 모델 불러오기
rf_model = joblib.load('./random/random_forest_model.joblib')

results_data = []

frame_idx = 30 * 200  # 30 FPS * 100초
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break

	# frame ,해상도 1/4로 줄이기
	frame = cv2.resize(frame, (1280, 720))

	# YOLOv8 추적 수행
	results = model.track(frame, persist=True, verbose=False)

	# 박스, 트랙 ID 및 키포인트 가져오기
	boxes = results[0].boxes.xywh.cpu().tolist()
	if results[0].boxes.id is None:
		track_ids = []
	else:
		track_ids = results[0].boxes.id.int().cpu().tolist()

	keypoints_xy = results[0].keypoints.xy.cpu().numpy().tolist() if results[0].keypoints is not None else []

	# 프레임에 결과 표시
	for i, box in enumerate(boxes):
		x, y, w, h = map(int, box)
		track_id = track_ids[i] if i < len(track_ids) else None
		color = get_color_from_id(track_id) if track_id is not None else (0, 255, 0)

		# 해당 프레임의 결과 저장
		frame_results = {
			"boxes": boxes,
			"track_ids": track_ids,
			"keypoints_xy": keypoints_xy
		}
		results_data.append(frame_results)
		results_data = results_data[-8:]

		predicted_labels = []
		if len(results_data) > 6:
			obj_datas = get_bson_for_frame(results_data, 3, 30)

			for obj_data in obj_datas.values():
				data_merged = obj_data["angles_before"] + obj_data["angles_after"] + obj_data["lengths_before"] + \
				              obj_data[
					              "lengths_after"]  # + obj_data["angular_velocities"] + obj_data["linear_velocities"]

				# data_merged로 예측
				data_merged = np.array(data_merged).reshape(1, -1)
				prediction = rf_model.predict(data_merged)
				predicted_label = label_encoder.inverse_transform(prediction)[0]

				# 예측 결과를 프레임에 표시
				# color = (255, 255, 255) if predicted_label=="non" else (0, 0, 255)
				# cv2.putText(frame, f'Action: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
				#             2)
				predicted_labels += [predicted_label]

		flag = False
		for p in predicted_labels:
			if p != "nan":
				flag = True
				color = (0, 0, 255)
				cv2.putText(frame, f'Action: {p}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
				#프레임 기록
				print(frame_idx, p)
				break
		if not flag:
			cv2.putText(frame, f'Action: nan', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

		# 바운딩 박스 그리기
		cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, thickness)
		if track_id is not None:
			cv2.putText(frame, f'ID: {track_id}', (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
			            2)

		# 키포인트 그리기
		if i < len(keypoints_xy):
			keypoints = keypoints_xy[i]
			for point in keypoints:
				px, py = map(int, point)
				cv2.circle(frame, (px, py), 5, color, -1)

	# 결과 프레임 표시
	cv2.imshow('YOLOv8 Skeleton Tracking', frame)

	key = cv2.waitKeyEx(1)  # waitKeyEx()를 사용하여 확장된 키 코드를 얻습니다.

	if key == 27:  # ESC 키
		break
	elif key == 2424832:  # 왼쪽 화살표 키
		frame_idx = max(frame_idx - int(5 * cap.get(cv2.CAP_PROP_FPS)), 0)
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
	elif key == 2555904:  # 오른쪽 화살표 키
		frame_idx = min(frame_idx + int(5 * cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1))
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
	elif key == 32:  # 스페이스바
		while True:
			key = cv2.waitKeyEx(0)
			if key == 32 or key == 27:
				break
	else:
		frame_idx += 1
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# 캡처 객체 및 창 해제
cap.release()
cv2.destroyAllWindows()
