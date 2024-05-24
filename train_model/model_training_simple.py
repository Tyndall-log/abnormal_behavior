import os
from glob import glob
import cv2
import bson
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET
from bson_to_data import get_bson_for_frame
import pandas as pd
import skeleton.model as model

video_folder_paths = [
	"E:/내 드라이브/machine_learning/dataset/assult/outsidedoor_06/",
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

# columns_1
# columns = [f"angles_before{i}" for i in range(1, 19)] \
# + [f"angles_after{i}" for i in range(1, 19)] \
# + [f"lengths_before{i}" for i in range(1, 19)] \
# + [f"lengths_after{i}" for i in range(1, 19)] \

train_data_list = []

# 전체 진행도 표시를 위한 tqdm 설정
with tqdm(total=len(video_files), desc="training") as overall_pbar:
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

		if not os.path.exists(txt_filename):
			print(f"Skipping {video_file}, no txt file.")
			overall_pbar.update(1)
			continue

		# bson 파일 읽기
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
		# cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		# 영상 크기 얻기
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		# cap.set(cv2.CAP_PROP_POS_FRAMES, event_start_frame)

		# txt 파일 열기
		id_to_name: dict = {}
		with open(txt_filename, "r") as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip()
				if line:
					object_name, target_id_str = line.split(":")
					target_id_list = [int(target_id) for target_id in target_id_str.split(",")]
					for target_id in target_id_list:
						id_to_name[target_id] = object_name

		# (객체이름:(시작 프레임, 종료 프레임, 라벨(엑션))) 형태로 저장
		action_dict = {}
		action_dict_n = {}
		for obj_id in objects:
			object_name = obj_id.find('objectname').text
			actions = obj_id.findall('action')
			for action in actions:
				action_name = action.find('actionname').text
				start_frame = int(action.find('frame/start').text)
				end_frame = int(action.find('frame/end').text)
				if object_name not in action_dict:
					action_dict[object_name] = []
					action_dict_n[object_name] = 0
				action_dict[object_name].append((start_frame, end_frame, action_name))

		for frame_idx in range(event_start_frame, event_end_frame):
			# 프레임 읽기
			data = get_bson_for_frame(results, frame_idx, fps)
			for obj_id, obj_data in data.items():
				data_merged = obj_data["angles_before"] + obj_data["angles_after"] + obj_data["lengths_before"] + obj_data["lengths_after"]\
				+ obj_data["angular_velocities"] + obj_data["linear_velocities"]
				if obj_id in id_to_name:
					obj_name = id_to_name[obj_id]
					if obj_name in action_dict:
						for action_tuple in action_dict[obj_name]:
							start_frame, end_frame, action_name = action_tuple
							if start_frame <= frame_idx <= end_frame:
								train_data_list.append(data_merged + [action_name])
				else:
					train_data_list.append(data_merged + ["None"])
		overall_pbar.update(1)

	print("Training data length:", len(train_data_list))
	save_filename = "train_data(velocities).csv"
	df = pd.DataFrame(train_data_list)
	df.to_csv(save_filename, index=False, header=False)


