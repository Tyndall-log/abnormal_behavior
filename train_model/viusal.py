import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import joblib

# 데이터 불러오기
data = pd.read_csv('train_data.csv', header=None)

# 열 이름 지정 (필요에 따라 수정)
# feature_names = [f'feature_{i}' for i in range(data.shape[1] - 1)]
feature_names = [f'angles_before_{i}' for i in range(8)]\
+ [f'angles_after_{i}' for i in range(8)]\
+ [f'lengths_before_{i}' for i in range(8)]\
+ [f'lengths_after_{i}' for i in range(8)]
data.columns = feature_names + ['label']

# 마지막 열을 라벨로 지정
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].astype(str).values

# 라벨 인코딩
label_encoder = joblib.load('./random/label_encoder.joblib')
y_encoded = label_encoder.transform(y)
classes = label_encoder.classes_

# 각 클래스의 비율을 유지하면서 데이터를 8:2로 분리
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 저장된 모델 불러오기
rf_model = joblib.load('./random/random_forest_model.joblib')
catboost_model = joblib.load('./catboost/catboost_model.joblib')

# RandomForest 예측
rf_y_pred = rf_model.predict(X_test)

# CatBoost 예측
catboost_y_pred = catboost_model.predict(X_test)

# 평가
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_report = classification_report(y_test, rf_y_pred, target_names=label_encoder.classes_)

catboost_accuracy = accuracy_score(y_test, catboost_y_pred)
catboost_report = classification_report(y_test, catboost_y_pred, target_names=label_encoder.classes_)

print(f"Random Forest Accuracy: {rf_accuracy}")
print("Random Forest Classification Report:")
print(rf_report)

print(f"CatBoost Accuracy: {catboost_accuracy}")
print("CatBoost Classification Report:")
print(catboost_report)


# 혼동 행렬 시각화
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
	plt.figure(figsize=(10, 7))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
	plt.title(title)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()


rf_cm = confusion_matrix(y_test, rf_y_pred)
catboost_cm = confusion_matrix(y_test, catboost_y_pred)

plot_confusion_matrix(rf_cm, classes, title='Random Forest Confusion Matrix')
plot_confusion_matrix(catboost_cm, classes, title='CatBoost Confusion Matrix')


# ROC 곡선 시각화 (다중 클래스)
def plot_roc_curve(y_test, y_score, classes, title='ROC Curve'):
	y_test_bin = label_binarize(y_test, classes=range(len(classes)))
	plt.figure(figsize=(10, 7))
	for i in range(len(classes)):
		fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(title)
	plt.legend(loc="lower right")
	plt.show()


# Random Forest ROC Curve
rf_y_score = rf_model.predict_proba(X_test)
plot_roc_curve(y_test, rf_y_score, classes, title='Random Forest ROC Curve')

# CatBoost ROC Curve
catboost_y_score = catboost_model.predict_proba(X_test)
plot_roc_curve(y_test, catboost_y_score, classes, title='CatBoost ROC Curve')


# 특징 중요도 시각화
def plot_feature_importance(importance, names, model_type):
	feature_importance = np.array(importance)
	feature_names = np.array(names)
	data = {'feature_names': feature_names, 'feature_importance': feature_importance}
	fi_df = pd.DataFrame(data)
	fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

	plt.figure(figsize=(10, 7))
	sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
	plt.title(f'{model_type} Feature Importance')
	plt.xlabel('Feature Importance')
	plt.ylabel('Feature Names')
	plt.show()


rf_importance = rf_model.feature_importances_
plot_feature_importance(rf_importance, feature_names, 'Random Forest')

catboost_importance = catboost_model.get_feature_importance()
plot_feature_importance(catboost_importance, feature_names, 'CatBoost')
