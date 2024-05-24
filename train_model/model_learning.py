import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# 데이터 불러오기
data = pd.read_csv('train_data(velocities).csv', header=None)

# 마지막 열을 라벨로 지정
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].astype(str).values

# 라벨 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 각 클래스의 비율을 유지하면서 데이터를 8:2로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 모델 생성
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
rf_model.fit(X_train, y_train)

# 예측
y_pred = rf_model.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# 모델 저장
joblib.dump(rf_model, 'random(velocities)/random_forest_model.joblib')
joblib.dump(label_encoder, 'random(velocities)/label_encoder.joblib')

# 모델 불러오기 (예시)
# loaded_rf_model = joblib.load('random_forest_model.joblib')
# loaded_label_encoder = joblib.load('label_encoder.joblib')
