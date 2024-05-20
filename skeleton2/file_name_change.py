from glob import glob
import os
from tqdm import tqdm

# 기본 경로 설정
base_path = r"G:\abnormal_behavior_wsl\Dataset\assult\outsidedoor_06"

# 재귀적으로 모든 mp4 파일 찾기
mp4_files = glob(os.path.join(base_path, "**", "*.mp4"), recursive=True)

#output_with_skeletons.mp4 파일 제외
mp4_files = [file for file in mp4_files if "output_with_skeletons.mp4" not in file]
mp4_files = [file for file in mp4_files if "skeletons.mp4" not in file]

# 전체 진행도 표시를 위한 tqdm 설정
parent_folder_dict = {}
with tqdm(total=len(mp4_files), desc="Processing mp4 files") as overall_pbar:
	for mp4_file in mp4_files:
		only_file_name = os.path.basename(mp4_file).replace(".mp4", "")
		parent_folder = os.path.dirname(mp4_file)

		if parent_folder not in parent_folder_dict:
			parent_folder_dict[parent_folder] = 0
		parent_folder_dict[parent_folder] += 1
		num = parent_folder_dict[parent_folder]

		if len(only_file_name) < 4:
			overall_pbar.update(1)
			continue

		# 새로운 파일 이름
		new_file_name = f"{num:1d}"

		# parent_folder/only_file_name*로 시작하는 모든 파일 및 폴더 찾기
		# parent_folder/new_file_name*으로 변경
		new_file_list = []
		for file in glob(os.path.join(parent_folder, only_file_name + "*")):
			new_file = os.path.join(parent_folder, os.path.basename(file).replace(only_file_name, new_file_name))
			new_file_list.append(new_file)
			os.rename(file, new_file)

		new_file_list = [file for file in new_file_list if os.path.isfile(file)]

		#하위 폴더 생성
		sub_folder = os.path.join(parent_folder, new_file_name)
		if not os.path.exists(sub_folder):
			os.makedirs(sub_folder)

		# 하위 폴더에 저장
		for file in new_file_list:
			ext = os.path.splitext(file)[1]
			if ext == ".bson":
				os.rename(file, os.path.join(sub_folder, os.path.basename(file)))
			if ext == ".xml":
				os.rename(file, os.path.join(sub_folder, os.path.basename(file)))

		# 하위 폴더의 "output_with_skeletons.mp4"을 "skeletons.mp4"로 변경
		if os.path.exists(os.path.join(sub_folder, "output_with_skeletons.mp4")):
			os.rename(os.path.join(sub_folder, "output_with_skeletons.mp4"), os.path.join(sub_folder, "skeletons.mp4"))

		overall_pbar.update(1)
