import bson

path = r"G:\abnormal_behavior_wsl\Dataset\assult\outsidedoor_06\23-1\23-1_cam01_assault01_place02_night_spring_yolov8x-pose-p6.bson"
with open(path, "rb") as f:
	data = bson.decode(f.read())
	print(data['results'][1000])  # [{'frame_id': 1, 'box': [[0.0, 0.0, 0.0, 0.0]], 'keypoints': [[0.0, 0.0]]}, ...]