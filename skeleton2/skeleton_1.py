from collections import defaultdict
import cv2
from ultralytics import YOLO
import numpy as np
import os
import bson  # pip install pymongo
from tqdm import tqdm

# YOLOv8 모델 로드
model_name = "yolov8x-pose-p6"
model = YOLO(f"{model_name}.pt")
model.to("cuda")

# 비디오 파일이 있는 폴더 경로
# video_folder_path = os.path.abspath("G:/abnormal_behavior_wsl/Dataset/assult/outsidedoor_06/23-3")
# video_folder_path = os.path.abspath("E:/내 드라이브/machine_learning/dataset/assult/outsidedoor_06/23-6")
video_folder_paths = [
    "E:/내 드라이브/machine_learning/dataset/assult/outsidedoor_06/25-2",
    "E:/내 드라이브/machine_learning/dataset/assult/outsidedoor_06/25-3",
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
with tqdm(total=len(video_files), desc="Processing videos") as overall_pbar:
    for video_file in video_files:
        # 결과를 저장할 BSON 파일 이름
        video_folder_path = os.path.dirname(video_file)
        only_filename = os.path.splitext(os.path.basename(video_file))[0]
        output_file = os.path.join(video_folder_path, only_filename, f"{only_filename}_{model_name}.bson")

        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 이전에 처리된 프레임 위치와 데이터를 가져오기
        if os.path.exists(output_file):
            with open(output_file, "rb") as f:
                existing_data = bson.decode(f.read())
                if existing_data:
                    last_frame = existing_data["results"][-1]["frame_id"]
                    # 영상이 이미 완료되었는지 확인
                    if last_frame >= total_frames:
                        print(f"Skipping {video_file}, already processed.")
                        overall_pbar.update(1)
                        cap.release()
                        continue

        # 영상을 처음부터 시작
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0

        # 개별 영상의 진행도 표시를 위한 tqdm 설정
        with tqdm(total=total_frames, initial=frame_index, desc=f"Processing {os.path.basename(video_file)}") as video_pbar:
            results_data = []
            while cap.isOpened():
                frame_index += 1

                # 비디오에서 프레임 읽기
                success, frame = cap.read()

                if success:
                    # YOLOv8 추적 수행
                    results = model.track(frame, persist=True, verbose=False)

                    # 박스, 트랙 ID 및 키포인트 가져오기
                    boxes = results[0].boxes.xywh.cpu().tolist()
                    if results[0].boxes.id is None:
                        track_ids = []
                    else:
                        track_ids = results[0].boxes.id.int().cpu().tolist()

                    keypoints_xy = results[0].keypoints.xy.cpu().numpy().tolist() if results[0].keypoints is not None else []

                    # 해당 프레임의 결과 저장
                    frame_results = {
                        "frame_id": frame_index,
                        "boxes": boxes,
                        "track_ids": track_ids,
                        "keypoints_xy": keypoints_xy
                    }
                    results_data.append(frame_results)

                    video_pbar.update(1)  # 진행 상황 업데이트

                    # 30프레임마다 결과를 BSON 파일에 기록
                    if frame_index % 30 == 0:
                        # video_pbar.update(30)  # 진행 상황 업데이트
                        with open(output_file, "wb") as f:
                            f.write(bson.encode({"results": results_data}))
                else:
                    # 비디오가 끝나면 루프 종료
                    break

            # 비디오 캡처 객체 해제
            cap.release()

            # 최종 결과를 BSON 파일에 저장
            with open(output_file, "wb") as f:
                f.write(bson.encode({"results": results_data}))

        overall_pbar.update(1)  # 전체 진행 상황 업데이트

print("모든 스켈레톤 데이터가 BSON 파일에 저장되었습니다.")
