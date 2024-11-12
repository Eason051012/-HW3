import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 生成隨機數據
np.random.seed(42)
X = np.random.randint(0, 1001, 300)  # 300 個 [0, 1000] 區間內的整數
Y = np.where((X > 500) & (X < 800), 1, 0)  # 在 (500, 800) 標記為 1，其餘為 0

# 分割數據
X_train, X_test, Y_train, Y_test = train_test_split(X.reshape(-1, 1), Y, test_size=0.2, random_state=42)

# 建立模型並進行訓練
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y1 = logreg.predict(X_test)

# Support Vector Machine
svm = SVC(probability=True)
svm.fit(X_train, Y_train)
y2 = svm.predict(X_test)

# 繪製圖形
plt.figure(figsize=(15, 6))

# 圖表 1: Logistic Regression 的結果
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='gray', label='True Labels')
plt.scatter(X_test, y1, color='blue', marker='x', label='Logistic Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y and Logistic Regression Prediction')
plt.legend()

# Logistic Regression 決策邊界
x_boundary = np.linspace(0, 1000, 300)
y_boundary = logreg.predict_proba(x_boundary.reshape(-1, 1))[:, 1]
plt.plot(x_boundary, y_boundary, color='blue', linestyle='--', label='Logistic Regression Boundary')
plt.legend()

# 圖表 2: SVM 的結果
plt.subplot(1, 2, 2)
plt.scatter(X, Y, color='gray', label='True Labels')
plt.scatter(X_test, y2, color='green', marker='s', label='SVM')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y and SVM Prediction')
plt.legend()

# SVM 決策邊界
y_boundary = svm.predict_proba(x_boundary.reshape(-1, 1))[:, 1]
plt.plot(x_boundary, y_boundary, color='green', linestyle='--', label='SVM Boundary')
plt.legend()

plt.tight_layout()
plt.show()