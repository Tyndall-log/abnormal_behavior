import os
from glob import glob
import cv2
import bson
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET


# print(os.path.exists("E:/내 드라이브/machine_learning/dataset/assult/outsidedoor_06/23-1"))

video_folder_paths = [
	"E:/내 드라이브/machine_learning/dataset/assult/outsidedoor_06/29-1",
]

target_width = 1280
target_height = 720

# 스켈레톤 및 바운딩 박스 표시 색상과 점 크기 설정
thickness = 2

def time_to_frame(time_str, fps):
	h, m, s = map(float, time_str.split(':'))
	total_seconds = h * 3600 + m * 60 + s
	return int(total_seconds * fps)


def get_color_from_id(track_id):
	np.random.seed(track_id)
	color = np.random.randint(0, 255, 3).tolist()
	return (color[0], color[1], color[2])


# 재귀적으로 모든 MP4 파일 찾기
video_files = []
for video_folder_path in video_folder_paths:
	for root, dirs, files in os.walk(video_folder_path):
		for file in files:
			if file.endswith(".mp4"):
				video_files.append(os.path.join(root, file))


# skeletons.mp4 파일 제외
video_files = [file for file in video_files if "skeletons.mp4" not in file]

# for video_file in video_files:
# 	print(f"Found video file: {video_file}")

# 전체 진행도 표시를 위한 tqdm 설정
with tqdm(total=len(video_files), desc="data labeling") as overall_pbar:
	for video_file in video_files:
		video_folder_path = os.path.dirname(video_file)
		only_filename = os.path.splitext(os.path.basename(video_file))[0]
		subfolder = os.path.join(video_folder_path, only_filename)

		bson_filename = os.path.join(subfolder, f"{only_filename}_yolov8x-pose-p6.bson")
		xml_filename = os.path.join(subfolder, f"{only_filename}.xml")
		txt_filename = os.path.join(subfolder, f"{only_filename}.txt")
		# skeleton_filename = os.path.join(subfolder, f"skeletons.mp4")

		if not os.path.exists(bson_filename):
			print(f"Skipping {video_file}, no bson file.")
			overall_pbar.update(1)
			continue

		if not os.path.exists(xml_filename):
			print(f"Skipping {video_file}, no xml file.")
			overall_pbar.update(1)
			continue

		# if not os.path.exists(skeleton_filename):
		# 	print(f"Skipping {video_file}, no skeleton file.")
		# 	overall_pbar.update(1)
		# 	continue

		if os.path.exists(txt_filename):
			print(f"Skipping {video_file}, already labeled.")
			overall_pbar.update(1)
			continue

		# bosn 파일 읽기
		with open(bson_filename, "rb") as f:
			bson_data = bson.decode(f.read())
			results = bson_data["results"]
		frame_count = len(results)

		# xml 파일 읽기
		tree = ET.parse(xml_filename)
		root = tree.getroot()
		fps = float(root.find('header/fps').text)
		event_start_time = root.find('event/starttime').text
		event_duration = root.find('event/duration').text
		event_start_frame = time_to_frame(event_start_time, fps)
		event_end_frame = min(event_start_frame + time_to_frame(event_duration, fps), frame_count)
		objects = root.findall('object')
		num_objects = len(objects)
		object_names = [obj.find('objectname').text for obj in objects]
		xml_video_width = int(root.find('size/width').text)
		xml_video_height = int(root.find('size/height').text)

		# video_file 파일 읽기
		cap = cv2.VideoCapture(video_file)
		cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		# 영상 크기 얻기
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		# 오브젝트의 위치 표시
		for obj in objects:
			object_name = obj.find('objectname').text
			keyframe = int(obj.find('position/keyframe').text)
			x = float(obj.find('position/keypoint/x').text)
			y = float(obj.find('position/keypoint/y').text)
			cap.set(cv2.CAP_PROP_POS_FRAMES, keyframe)
			ret, frame = cap.read()
			if not ret:
				print(f"Skipping {video_file}, no frame.")
				overall_pbar.update(1)
				continue

			# frame에 x, y 위치에 원 그리기
			x = int(x / xml_video_width * width)
			y = int(y / xml_video_height * height)
			cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
			cv2.imshow("frame", frame)

			print(f"빨간 점의 사람이 {object_name}입니다.")
			# 키보드 입력 대기
			key = cv2.waitKey(0)

		frame_idx = event_start_frame
		# cap.set(cv2.CAP_PROP_POS_FRAMES, event_start_frame)

		# 라벨링 해야할 id dict
		labeling_ids = {}
		for i in range(event_start_frame, event_end_frame):
			result = results[i]
			track_ids = result['track_ids']
			for track_id in track_ids:
				if track_id not in labeling_ids:
					labeling_ids[track_id] = ""

		# persons = {name: [] for name in object_names}
		while True:
			if not frame_idx < len(results):
				break

			result = results[frame_idx]
			keypoints_list = result['keypoints_xy']
			boxes = result['boxes']
			track_ids = result['track_ids']

			# need_labeling_id_list = []
			for i, (keypoints, box, track_id) in enumerate(zip(keypoints_list, boxes, track_ids)):
				if track_id not in labeling_ids:
					continue

				if labeling_ids[track_id] != "":
					continue

				_frame_idx = frame_idx
				_last_id = track_id
				while True:
					_result = results[_frame_idx]
					cap.set(cv2.CAP_PROP_POS_FRAMES, _frame_idx)
					ret, frame = cap.read()

					# 원본 해상도
					orig_height, orig_width = frame.shape[:2]

					# 해상도 변경 비율 계산
					scale_x = target_width / orig_width
					scale_y = target_height / orig_height

					# 프레임 리사이즈
					frame = cv2.resize(frame, (target_width, target_height))

					# track_id
					if track_id in _result['track_ids']:
						idx = _result['track_ids'].index(track_id)
						_keypoints = _result['keypoints_xy'][idx]
						_box = _result['boxes'][idx]
						# 사람마다 고유한 색상 가져오기
						color = get_color_from_id(track_id)

						# 바운딩 박스 그리기
						x1, y1, w, h = _box
						x1, y1, x2, y2 = int((x1 - w / 2) * scale_x), int((y1 - h / 2) * scale_y), int(
							(x1 + w / 2) * scale_x), int(
							(y1 + h / 2) * scale_y)
						cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
						cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

						# # 스켈레톤 그리기
						# for i in range(len(_keypoints)):
						# 	x, y = _keypoints[i]
						# 	if x != 0.0 and y != 0.0:
						# 		# 키포인트 위치를 새로운 해상도로 조정
						# 		x = int(x * scale_x)
						# 		y = int(y * scale_y)
						# 		cv2.circle(frame, (x, y), 4, color, thickness)
					cv2.imshow('frame', frame)

					key = cv2.waitKeyEx(0)
					if key == 48:
						labeling_ids[track_id] = "unknown"
						print(f"ID: {track_id} is labeled as unknown")
						break
					elif 48 < key < 58:
						num = key - 49
						if num < len(object_names):
							labeling_ids[track_id] = object_names[num]
							print(f"ID: {track_id} is labeled as {object_names[num]}")
							break
						print("Invalid number")
					elif key == 2424832: # 왼쪽 화살표 키
						_frame_idx -= 1
					elif key == 2555904:  # 오른쪽 화살표 키
						_frame_idx += 1
			frame_idx += 1

		# 프로그램 종료 전에 labeling_ids를 txt 파일로 저장합니다.
		with open(txt_filename, 'w') as f:
			for object_name in object_names:
				f.write(f"{object_name}: ")
				T = ""
				for track_id, label in labeling_ids.items():
					if label == object_name:
						T += f"{track_id}, "
				f.write(T[:-2])
				f.write("\n")

		print(f"Labeling {video_file} is done.")
		overall_pbar.update(1)