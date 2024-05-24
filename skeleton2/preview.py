import bson
import cv2
import numpy as np
import os
from glob import glob

# 기본 경로 설정
# base_path = r"G:\abnormal_behavior_wsl\Dataset\assult\outsidedoor_06\23-1\23-1_cam02_assault01_place02_night_summer"
base_path = "E:/내 드라이브/machine_learning/dataset/assult/outsidedoor_06/23-1/2"

# .bson 파일과 .mp4 파일 찾기
base_only_filename = os.path.splitext(os.path.basename(base_path))[0]
bson_files = glob(os.path.join(base_path, f"{base_only_filename}_*.bson"))
mp4_files = glob(base_path + "*.mp4")

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

# 비디오 파일을 로드
cap = cv2.VideoCapture(video_path)

# 프레임당 밀리초
frame_time_ms = int(1000 / cap.get(cv2.CAP_PROP_FPS))

# 목표 해상도
target_width = 1280
target_height = 720
cv2.namedWindow('Skeleton Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Skeleton Video', target_width, target_height)

frame_idx = 0

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
		keypoints_list = result['keypoints_xy']
		print(keypoints_list)
		boxes = result['boxes']
		track_ids = result['track_ids']

		for keypoints, box, track_id in zip(keypoints_list, boxes, track_ids):
			# 사람마다 고유한 색상 가져오기
			color = get_color_from_id(track_id)

			# 바운딩 박스 그리기
			x1, y1, w, h = box
			x1, y1, x2, y2 = int((x1 - w / 2) * scale_x), int((y1 - h / 2) * scale_y), int((x1 + w / 2) * scale_x), int(
				(y1 + h / 2) * scale_y)
			cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
			cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

			# 스켈레톤 그리기
			for i in range(len(keypoints)):
				x, y = keypoints[i]
				if x != 0.0 and y != 0.0:
					# 키포인트 위치를 새로운 해상도로 조정
					x = int(x * scale_x)
					y = int(y * scale_y)
					cv2.circle(frame, (x, y), 4, color, thickness)

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
