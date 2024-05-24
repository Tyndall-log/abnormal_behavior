import os
from glob import glob
import cv2
import bson
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET
import math

def calculate_angle(p1, p2, p3):
	if any(coord == 0 for coord in p1 + p2 + p3):
		return 0
	v1 = (p1[0] - p2[0], p1[1] - p2[1])
	v2 = (p3[0] - p2[0], p3[1] - p2[1])
	dot_product = v1[0] * v2[0] + v1[1] * v2[1]
	mag_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
	mag_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
	if mag_v1 * mag_v2 == 0:
		return 0
	angle = math.acos(dot_product / (mag_v1 * mag_v2))
	return math.degrees(angle)

def calculate_length(p1, p2):
	if any(coord == 0 for coord in p1 + p2):
		return 0
	return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_angular_velocity(angle1, angle2, time_interval):
	if angle1 == 0 or angle2 == 0:
		return 0
	return (angle2 - angle1) / time_interval

def calculate_linear_velocity(length1, length2, time_interval):
	if length1 == 0 or length2 == 0:
		return 0
	return (length2 - length1) / time_interval

def get_bson_for_frame(data, frame_number, fps=30):
	angle_indices = [(5, 7, 9), (6, 8, 10), (7, 5, 11), (8, 6, 12),
					 (11, 12, 14), (12, 11, 13), (12, 14, 16), (11, 13, 15)]
	length_indices = [(5, 7), (6, 8), (7, 9), (8, 10), (11, 13), (12, 14), (11, 15), (12, 16)]

	mean_n = 3

	track_ids = list(data[frame_number]["track_ids"])
	track_ids_len = len(track_ids)
	track_ids_dict = {obj_id: idx for idx, obj_id in enumerate(track_ids)}

	start_frame = max(frame_number - mean_n, 0)
	end_frame = min(frame_number + mean_n + 1, len(data))

	# before_mean
	before_keypoints_xy = [[[0.0, 0.0] for _ in range(17)] for _ in range(track_ids_len)]
	before_keypoints_xy_n = [[0 for _ in range(17)] for _ in range(track_ids_len)]
	for i in range(start_frame, frame_number):
		for n, obj_id in enumerate(data[i]["track_ids"]):
			if obj_id in track_ids_dict:
				keypoints_xy = data[i]["keypoints_xy"][n]
				oid = track_ids_dict[obj_id]
				for j in range(17):
					if keypoints_xy[j] != [0.0, 0.0]:
						before_keypoints_xy[oid][j][0] += keypoints_xy[j][0]
						before_keypoints_xy[oid][j][1] += keypoints_xy[j][1]
						before_keypoints_xy_n[oid][j] += 1
	for i in range(track_ids_len):
		for j in range(17):
			if before_keypoints_xy_n[i][j] != 0:
				before_keypoints_xy[i][j][0] /= before_keypoints_xy_n[i][j]
				before_keypoints_xy[i][j][1] /= before_keypoints_xy_n[i][j]

	# after_mean
	after_keypoints_xy = [[[0.0, 0.0] for _ in range(17)] for _ in range(track_ids_len)]
	after_keypoints_xy_n = [[0 for _ in range(17)] for _ in range(track_ids_len)]
	for i in range(frame_number, end_frame):
		for n, obj_id in enumerate(data[i]["track_ids"]):
			if obj_id in track_ids_dict:
				keypoints_xy = data[i]["keypoints_xy"][n]
				oid = track_ids_dict[obj_id]
				for j in range(17):
					if keypoints_xy[j] != [0.0, 0.0]:
						after_keypoints_xy[oid][j][0] += keypoints_xy[j][0]
						after_keypoints_xy[oid][j][1] += keypoints_xy[j][1]
						after_keypoints_xy_n[oid][j] += 1
	for i in range(track_ids_len):
		for j in range(17):
			if after_keypoints_xy_n[i][j] != 0:
				after_keypoints_xy[i][j][0] /= after_keypoints_xy_n[i][j]
				after_keypoints_xy[i][j][1] /= after_keypoints_xy_n[i][j]

	result = {}
	time_interval = mean_n / fps

	# 각도, 길이, 각속도, 선속도 계산
	for obj_id, idx in track_ids_dict.items():
		angles_before = [calculate_angle(before_keypoints_xy[idx][i1], before_keypoints_xy[idx][i2], before_keypoints_xy[idx][i3]) for i1, i2, i3 in angle_indices]
		angles_after = [calculate_angle(after_keypoints_xy[idx][i1], after_keypoints_xy[idx][i2], after_keypoints_xy[idx][i3]) for i1, i2, i3 in angle_indices]

		lengths_before = [calculate_length(before_keypoints_xy[idx][i1], before_keypoints_xy[idx][i2]) for i1, i2 in length_indices]
		lengths_after = [calculate_length(after_keypoints_xy[idx][i1], after_keypoints_xy[idx][i2]) for i1, i2 in length_indices]

		angular_velocities = [calculate_angular_velocity(angles_before[i], angles_after[i], time_interval) for i in range(len(angle_indices))]
		linear_velocities = [calculate_linear_velocity(lengths_before[i], lengths_after[i], time_interval) for i in range(len(length_indices))]

		result[obj_id] = {
			"angles_before": angles_before,
			"angles_after": angles_after,
			"lengths_before": lengths_before,
			"lengths_after": lengths_after,
			"angular_velocities": angular_velocities,
			"linear_velocities": linear_velocities
		}

	return result