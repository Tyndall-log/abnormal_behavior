import os
import bson
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob

# 기본 경로 설정
# base_path = r"G:\abnormal_behavior_wsl\Dataset\assult\outsidedoor_06"
base_path = "E:/내 드라이브/machine_learning/dataset/assult/outsidedoor_06"
# base_path = "E:/내 드라이브/machine_learning/dataset/fight/outsidedoor_12"

# 재귀적으로 모든 mp4 파일 찾기
mp4_files = glob(os.path.join(base_path, "**", "*.mp4"), recursive=True)

# skeletons.mp4 파일 제외
mp4_files = [file for file in mp4_files if "skeletons.mp4" not in file]
mp4_files = sorted(mp4_files)
for mp4_file in mp4_files:
	print(f"Found mp4 file: {mp4_file}")
input("Press Enter to continue...")


def get_color_from_id(track_id):
	np.random.seed(track_id)
	color = np.random.randint(0, 255, 3).tolist()
	return (color[0], color[1], color[2])


# 전체 진행도 표시를 위한 tqdm 설정
with tqdm(total=len(mp4_files), desc="Processing mp4 files") as overall_pbar:
	for mp4_file in mp4_files:
		# BSON 파일 찾기
		mp4_file_only_filename = os.path.basename(mp4_file).replace(".mp4", "")
		bson_pattern = mp4_file.replace(".mp4", os.path.sep + f"{mp4_file_only_filename}_*.bson")
		bson_files = glob(bson_pattern)

		# 대응되는 BSON 파일 선택
		if len(bson_files) == 1:
			bson_file = bson_files[0]
		elif len(bson_files) > 1:
			print(f"여러 개의 BSON 파일이 발견되었습니다. 사용할 파일을 선택하세요:")
			for idx, file in enumerate(bson_files):
				print(f"{idx + 1}: {file}")
			selected_idx = int(input("파일 번호를 입력하세요: ")) - 1
			bson_file = bson_files[selected_idx]
		else:
			print(f"{mp4_file}에 대한 BSON 파일이 발견되지 않았습니다. 건너뜁니다...")
			overall_pbar.update(1)
			continue

		# BSON 파일 열기
		with open(bson_file, "rb") as f:
			data = bson.decode(f.read())

		# 결과 리스트 가져오기
		results = data['results']

		# 비디오 파일 열기
		cap = cv2.VideoCapture(mp4_file)
		if not cap.isOpened():
			print(f"Failed to open video file {mp4_file}. Skipping...")
			overall_pbar.update(1)
			continue

		# 전체 프레임 수
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		# BSON 파일이 완성되었는지 확인
		if results[-1]["frame_id"] < total_frames:
			print(f"BSON file {bson_file} is not complete. Skipping...")
			overall_pbar.update(1)
			cap.release()
			continue

		# 결과를 저장할 비디오 파일 이름
		output_file = mp4_file.replace(".mp4", os.path.sep + "skeletons.mp4")

		# 이미 처리된 비디오 파일이 존재하는 경우 모든 프레임이 생성되었는지 확인
		if os.path.exists(output_file):
			out = cv2.VideoCapture(output_file)
			if int(out.get(cv2.CAP_PROP_FRAME_COUNT)) == total_frames:
				print(f"Skipping {mp4_file}, already processed.")
				overall_pbar.update(1)
				cap.release()
				out.release()
				continue
		else:
			# 만약 폴더가 없다면 생성
			if not os.path.exists(os.path.dirname(output_file)):
				os.makedirs(os.path.dirname(output_file))

		# 프레임당 밀리초
		frame_time_ms = int(1000 / cap.get(cv2.CAP_PROP_FPS))

		# 목표 해상도
		target_width = 1280
		target_height = 720

		# 비디오 저장 설정
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
		out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), (target_width, target_height))

		frame_idx = 0

		# 스켈레톤 및 바운딩 박스 표시 색상과 점 크기 설정
		thickness = 2

		# 개별 영상의 진행도 표시를 위한 tqdm 설정
		show_filename = os.path.join(os.path.basename(os.path.dirname(mp4_file)), os.path.basename(mp4_file)).replace("\\", "/")
		with tqdm(total=total_frames, desc=f"Processing {show_filename}") as video_pbar:
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
					boxes = result['boxes']
					track_ids = result['track_ids']

					for keypoints, box, track_id in zip(keypoints_list, boxes, track_ids):
						# 사람마다 고유한 색상 가져오기
						color = get_color_from_id(track_id)

						# 바운딩 박스 그리기
						x1, y1, w, h = box
						x1, y1, x2, y2 = int((x1 - w / 2) * scale_x), int((y1 - h / 2) * scale_y), int((x1 + w / 2) * scale_x), int((y1 + h / 2) * scale_y)
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

				# 프레임 저장
				out.write(frame)

				frame_idx += 1

				# video_pbar.update(1)
				if frame_idx % 30 == 0:
					video_pbar.update(30)

			cap.release()
			out.release()

		overall_pbar.update(1)

print("모든 BSON 데이터가 영상으로 저장되었습니다.")
