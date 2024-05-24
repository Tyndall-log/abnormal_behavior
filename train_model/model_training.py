import os
from glob import glob
import cv2
import bson
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


# print(os.path.exists("E:/내 드라이브/machine_learning/dataset/assult/outsidedoor_06/23-1"))

video_folder_paths = [
    "E:/내 드라이브/machine_learning/dataset/assult/outsidedoor_06/23-1",
]

# 재귀적으로 모든 MP4 파일 찾기
video_files = []
for video_folder_path in video_folder_paths:
    for root, dirs, files in os.walk(video_folder_path):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))


# skeletons.mp4 파일 제외
video_files = [file for file in video_files if "skeletons.mp4" not in file]

for video_file in video_files:
    print(f"Found video file: {video_file}")

# 전체 진행도 표시를 위한 tqdm 설정
with tqdm(total=len(video_files), desc="Make training data") as overall_pbar:
    for video_file in video_files:
        video_folder_path = os.path.dirname(video_file)
        only_filename = os.path.splitext(os.path.basename(video_file))[0]
        subfolder = os.path.join(video_folder_path, only_filename)

        bson_filename = os.path.join(subfolder, f"{only_filename}_yolov8x-pose-p6.bson")
        xml_filename = os.path.join(subfolder, f"{only_filename}.xml")
        txt_filename = os.path.join(subfolder, f"{only_filename}.txt")

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

        break  # 임시