import bson
import cv2
import numpy as np
import os
from glob import glob
import joblib
from bson_to_data import get_bson_for_frame

# 기본 경로 설정
# base_path = r"G:\abnormal_behavior_wsl\Dataset\assult\outsidedoor_06\23-1\1"
# base_path = "E:/내 드라이브/machine_learning/dataset/assult/outsidedoor_06/23-1/2"
base_path = "G:/내 드라이브/machine_learning/dataset/assult/outsidedoor_06/23-1/2"

# .bson 파일과 .mp4 파일 찾기
base_only_filename = os.path.splitext(os.path.basename(base_path))[0]
bson_files = glob(os.path.join(base_path, f"{base_only_filename}_*.bson"))
mp4_files = glob(base_path + "*.mp4")

print(f"base_only_filename: {base_only_filename}")
print(f"bson_files: {bson_files}")
print(f"mp4_files: {mp4_files}")

# .bson 파일 선택
if len(bson_files) == 1:
	path = bson_files[0]
elif len(bson_files) > 1:
	print("여러 개의 .bson 파일이 발견되었습니다. 사용할 파일을 선택하세요:")
	for idx, file in enumerate(bson_files):
		print(f"{idx + 1}: {file}")
	selected_idx = int(input("파일 번호를 입력하세요: ")) - 1
	path = bson_files[selected_idx]
else:
	raise FileNotFoundError("해당 경로에 .bson 파일이 없습니다.")

# .mp4 파일 선택
if len(mp4_files) == 1:
	video_path = mp4_files[0]
elif len(mp4_files) > 1:
	print("여러 개의 .mp4 파일이 발견되었습니다. 사용할 파일을 선택하세요:")
	for idx, file in enumerate(mp4_files):
		print(f"{idx + 1}: {file}")
	selected_idx = int(input("파일 번호를 입력하세요: ")) - 1
	video_path = mp4_files[selected_idx]
else:
	raise FileNotFoundError("해당 경로에 .mp4 파일이 없습니다.")

# BSON 파일에서 데이터를 로드
with open(path, "rb") as f:
	data = bson.decode(f.read())

# 각 프레임의 결과를 담고 있는 리스트
results = data['results']

label_encoder = joblib.load('./random/label_encoder.joblib')

# 저장된 모델 불러오기
rf_model = joblib.load('./random/random_forest_model.joblib')

results_data = []

# 비디오 파일을 로드
cap = cv2.VideoCapture(video_path)

# 프레임당 밀리초
frame_time_ms = int(1000 / cap.get(cv2.CAP_PROP_FPS))

# 목표 해상도
target_width = 1280
target_height = 720
cv2.namedWindow('Skeleton Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Skeleton Video', target_width, target_height)

frame_idx = 7800

# 스켈레톤 및 바운딩 박스 표시 색상과 점 크기 설정
thickness = 2

def get_color_from_id(track_id):
	np.random.seed(track_id)
	color = np.random.randint(0, 255, 3).tolist()
	return (color[0], color[1], color[2])


while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break

	# 원본 해상도
	orig_height, orig_width = frame.shape[:2]

	# 해상도 변경 비율 계산
	scale_x = target_width / orig_width
	scale_y = target_height / orig_height

	# 프레임 리사이즈
	frame = cv2.resize(frame, (target_width, target_height))

	if frame_idx < len(results):
		result = results[frame_idx]
		# 박스, 트랙 ID 및 키포인트 가져오기
		boxes = result['boxes']
		track_ids = result['track_ids']
		keypoints_xy = result['keypoints_xy']

		# #frame에 원 그리기
		# cv2.circle(frame, (640, 360), 5, (0, 255, 0), -1)

		predicted_labels = []

		# 해당 프레임의 결과 저장
		frame_results = {
			"boxes": boxes,
			"track_ids": track_ids,
			"keypoints_xy": keypoints_xy
		}
		results_data.append(frame_results)
		results_data = results_data[-8:]

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
				predicted_labels += [predicted_label]

		# 프레임에 결과 표시
		for i, box in enumerate(boxes):
			x, y, w, h = map(int, box)
			x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
			track_id = track_ids[i] if i < len(track_ids) else None
			color = get_color_from_id(track_id) if track_id is not None else (0, 255, 0)

			if i < len(predicted_labels) and predicted_labels[i] != "nan":
				color = (0, 0, 255)

			# 바운딩 박스 그리기
			cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, thickness)
			if track_id is not None:
				cv2.putText(frame, f'ID: {track_id}', (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
				color,2)

			# 키포인트 그리기
			if i < len(keypoints_xy):
				keypoints = keypoints_xy[i]
				for point in keypoints:
					px, py = map(int, point)
					px, py = int(px * scale_x), int(py * scale_y)
					cv2.circle(frame, (px, py), 5, color, -1)

		flag = False
		predicted_label_xy = (target_width // 2 - 20, 30)
		for p in predicted_labels:
			if p != "nan":
				flag = True
				cv2.putText(frame, f'Action: {p}', predicted_label_xy, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
				# 프레임 기록
				print(frame_idx, p)
				break
		if not flag:
			cv2.putText(frame, f'Action: nan', predicted_label_xy, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

	cv2.imshow('Skeleton Video', frame)

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

cap.release()
cv2.destroyAllWindows()
