import cv2
import numpy as np
from ultralytics import YOLO
import torch
import joblib
from bson_to_data import get_bson_for_frame

# YOLOv8 모델 로드
model_name = "yolov8n-pose"
model = YOLO(f"{model_name}.pt")
model.to("cuda")

# 스켈레톤 및 바운딩 박스 표시 색상과 점 크기 설정
thickness = 2

# 웹캠 초기화
cap = cv2.VideoCapture(0)

def get_color_from_id(track_id):
    np.random.seed(track_id)
    color = np.random.randint(0, 255, 3).tolist()
    return (color[0], color[1], color[2])

label_encoder = joblib.load('./random/label_encoder.joblib')

# 저장된 모델 불러오기
rf_model = joblib.load('./random/random_forest_model.joblib')

results_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 추적 수행
    results = model.track(frame, persist=True, verbose=False)

    # 박스, 트랙 ID 및 키포인트 가져오기
    boxes = results[0].boxes.xywh.cpu().tolist()
    if results[0].boxes.id is None:
        track_ids = []
    else:
        track_ids = results[0].boxes.id.int().cpu().tolist()

    keypoints_xy = results[0].keypoints.xy.cpu().numpy().tolist() if results[0].keypoints is not None else []

    # 프레임에 결과 표시
    for i, box in enumerate(boxes):
        x, y, w, h = map(int, box)
        track_id = track_ids[i] if i < len(track_ids) else None
        color = get_color_from_id(track_id) if track_id is not None else (0, 255, 0)

        # 해당 프레임의 결과 저장
        frame_results = {
            "boxes": boxes,
            "track_ids": track_ids,
            "keypoints_xy": keypoints_xy
        }
        results_data.append(frame_results)
        results_data = results_data[-8:]

        if len(results_data) > 6:
            obj_datas = get_bson_for_frame(results_data, 3, 30)

            for obj_data in obj_datas.values():
                data_merged = obj_data["angles_before"] + obj_data["angles_after"] + obj_data["lengths_before"] + obj_data[
                    "lengths_after"]  # + obj_data["angular_velocities"] + obj_data["linear_velocities"]
                print(data_merged)

                # data_merged로 예측
                data_merged = np.array(data_merged).reshape(1, -1)
                prediction = rf_model.predict(data_merged)
                predicted_label = label_encoder.inverse_transform(prediction)[0]

                # 예측 결과를 프레임에 표시
                cv2.putText(frame, f'Action: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, thickness)
        if track_id is not None:
            cv2.putText(frame, f'ID: {track_id}', (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 키포인트 그리기
        if i < len(keypoints_xy):
            keypoints = keypoints_xy[i]
            for point in keypoints:
                px, py = map(int, point)
                cv2.circle(frame, (px, py), 5, color, -1)

    # 결과 프레임 표시
    cv2.imshow('YOLOv8 Skeleton Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 객체 및 창 해제
cap.release()
cv2.destroyAllWindows()
