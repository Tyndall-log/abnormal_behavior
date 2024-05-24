import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
data = pd.read_csv('train_data.csv', header=None)

# 마지막 열을 라벨로 지정
labels = data.iloc[:, -1].astype(str)

# 라벨별 데이터 갯수 세기
label_counts = labels.value_counts()

# 라벨 갯수 시각화 (로그 스케일)
plt.figure(figsize=(12, 8))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
plt.yscale('log')
plt.title('Number of Samples per Label (Log Scale)')
plt.xlabel('Labels')
plt.ylabel('Number of Samples (Log Scale)')
plt.xticks(rotation=45)
plt.show()
