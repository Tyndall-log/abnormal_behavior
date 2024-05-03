import os
import json
import matplotlib.pyplot as plt
import numpy as np


def get_frame_data():
	# with open('frame_data.json', 'r') as f:
	with open('frame_data_yolov8n-pose.json', 'r') as f:
		return json.load(f)


def calculate_angle(pt1, pt2, pt3):
	x1, y1 = pt1
	x2, y2 = pt2
	x3, y3 = pt3
	
	vector1 = [x1 - x2, y1 - y2]
	vector2 = [x3 - x2, y3 - y2]
	
	unit_vector1 = vector1 / np.linalg.norm(vector1)
	unit_vector2 = vector2 / np.linalg.norm(vector2)
	
	dot_product = np.dot(unit_vector1, unit_vector2)
	angle_radians = np.arccos(dot_product)
	
	angle_degrees = np.degrees(angle_radians)
	return angle_degrees


def calculate_distance(pt1, pt2):
	return np.linalg.norm(np.array(pt1) - np.array(pt2))


def calculate_angular_velocity(angle1, angle2, time1, time2):
	if angle1 == 0.0 or angle2 == 0.0:
		return 0.0
	return (angle2 - angle1) / (time2 - time1)


def calculate_linear_velocity(distance1, distance2, time1, time2):
	if distance1 == 0.0 or distance2 == 0.0:
		return 0.0
	return (distance2 - distance1) / (time2 - time1)


def set_axes_limits(ax, x_data, y_data, x_start, x_end):
	# 주어진 x 범위에 대한 y 데이터 필터링
	y_data_filtered = [y for x, y in zip(x_data, y_data) if x_start <= x <= x_end]
	
	# 필터링된 y 데이터의 최소 및 최대값을 구하고 ylim 설정
	if y_data_filtered:  # 필터링된 데이터가 있을 경우에만 실행
		ax.set_xlim(x_start, x_end)
		ax.set_ylim(np.min(y_data_filtered), np.max(y_data_filtered))


if __name__ == '__main__':
	fps = 60  # Assuming 60 frames per second
	frame_data = get_frame_data()
	
	angles = []
	lengths = []
	angular_velocities = []
	linear_velocities = []
	
	angle_indices = [(5, 7, 9), (6, 8, 10), (7, 5, 11), (8, 6, 12),
					 (11, 12, 14), (12, 11, 13), (12, 14, 16), (11, 13, 15)]
	length_indices = [(5, 7), (6, 8), (7, 9), (8, 10), (11, 13), (12, 14), (11, 15), (12, 16)]
	
	# Process each frame in the data
	for i in range(len(frame_data)):
		angles = []
		lengths = []
		angular_velocities = []
		linear_velocities = []
		keypoints1 = frame_data[i]["keypoints"][0]
		time1 = frame_data[i]["frame_id"] / fps
		
		if len(keypoints1) == 0:
			continue
		
		# print(keypoints1)
		
		for angle_idx in angle_indices:
			if [0.0, 0.0] in [keypoints1[idx] for idx in angle_idx]:
				angles.append(0.0)
				continue
			angle = calculate_angle(keypoints1[angle_idx[0]], keypoints1[angle_idx[1]], keypoints1[angle_idx[2]])
			angles.append(angle)
		
		for length_idx in length_indices:
			if [0.0, 0.0] in [keypoints1[idx] for idx in length_idx]:
				lengths.append(0.0)
				continue
			length = calculate_distance(keypoints1[length_idx[0]], keypoints1[length_idx[1]])
			lengths.append(length)
		
		frame_data[i]["time"] = time1
		frame_data[i]["angles"] = angles
		frame_data[i]["lengths"] = lengths
		
		if i > 0:
			keypoints2 = frame_data[i - 1]["keypoints"][0]
			time2 = frame_data[i - 1]["frame_id"] / fps
			if "angles" not in frame_data[i - 1]:
				continue
			if "lengths" not in frame_data[i - 1]:
				continue
			angles_b = frame_data[i - 1]["angles"]
			lengths_b = frame_data[i - 1]["lengths"]
			
			# calculate_angular_velocity() 함수를 사용하여 각도 속도를 계산합니다.
			for j in range(len(angle_indices)):
				angular_velocity = calculate_angular_velocity(angles[j], angles_b[j], time1, time2)
				angular_velocities.append(angular_velocity)
			
			# calculate_linear_velocity() 함수를 사용하여 선속도를 계산합니다.
			for j in range(len(length_indices)):
				linear_velocity = calculate_linear_velocity(lengths[j], lengths_b[j], time1, time2)
				linear_velocities.append(linear_velocity)
			
			frame_data[i]["angular_velocities"] = angular_velocities
			frame_data[i]["linear_velocities"] = linear_velocities
	
	# 각속도와 선속도를 그리기 위한 시간 축 데이터
	times = [fd['time'] for fd in frame_data if "angular_velocities" in fd]
	# print(len(times))
	
	# 각속도 그래프
	plt.figure(figsize=(13, 5))
	plt.subplot(2, 1, 1)
	for j in range(len(angle_indices)):
		angular_vels = [fd['angular_velocities'][j] for fd in frame_data if "angular_velocities" in fd]
		plt.plot(times, angular_vels, label=f'Angle {j + 1}')
	plt.xlabel('Time (seconds)')
	plt.ylabel('Angular Velocity (degrees/second)')
	plt.title('Angular Velocities over Time')
	plt.legend()
	plt.xlim(13, 21)
	plt.ylim(-8000, 8000)
	
	# 선속도 그래프
	plt.subplot(2, 1, 2)
	for j in range(len(length_indices)):
		linear_vels = [fd['linear_velocities'][j] for fd in frame_data if "linear_velocities" in fd]
		plt.plot(times, linear_vels, label=f'Length {j + 1}')
	plt.xlabel('Time (seconds)')
	plt.ylabel('Linear Velocity (units/second)')
	plt.title('Linear Velocities over Time')
	plt.legend()
	plt.xlim(13, 21)
	plt.ylim(-6000, 6000)
	
	plt.tight_layout()
	plt.subplots_adjust(top=0.95, hspace=0.4)
	plt.show()
	