import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import log_loss
from utils import load_data
from sklearn.tree import DecisionTreeClassifier
# 创建数据集
dataset1 = load_data("project3_dataset1.txt")
dataset2 = load_data("project3_dataset2.txt")
data = dataset1
X, y = data["data"], data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 过拟合参数配置
ada_overfit = AdaBoostClassifier(n_estimators=400, learning_rate=1, estimator=DecisionTreeClassifier(max_depth=2), random_state=42)
ada_overfit.fit(X_train, y_train)
overfit_train_loss = [log_loss(y_train, y_pred) for y_pred in ada_overfit.staged_predict(X_train)]
overfit_test_loss = [log_loss(y_test, y_pred) for y_pred in ada_overfit.staged_predict(X_test)]

# 欠拟合参数配置
ada_underfit = AdaBoostClassifier(n_estimators=10, learning_rate=0.5, estimator=DecisionTreeClassifier(max_depth=2), random_state=42)
ada_underfit.fit(X_train, y_train)
underfit_train_loss = [log_loss(y_train, y_pred) for y_pred in ada_underfit.staged_predict(X_train)]
underfit_test_loss = [log_loss(y_test, y_pred) for y_pred in ada_underfit.staged_predict_proba(X_test)]

# 绘制过拟合损失曲线
plt.figure(figsize=(12, 6))
plt.plot(overfit_train_loss, label='Train Loss', color='blue')
plt.plot(overfit_test_loss, label='Test Loss', color='orange')
plt.title('Overfitting Loss of Dataset 2')
plt.xlabel('Number of Iterations')
plt.ylabel('Log Loss')
plt.legend()
plt.show()

# 绘制欠拟合损失曲线
plt.figure(figsize=(12, 6))
plt.plot(underfit_train_loss, label='Train Loss', color='blue')
plt.plot(underfit_test_loss, label='Test Loss', color='orange')
plt.title('Underfitting Loss of Dataset 2')
plt.xlabel('Number of Iterations')
plt.ylabel('Log Loss')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss


# Define the base estimators
base_estimators = {
    'Decision Tree': DecisionTreeClassifier(max_depth=2),
    'SVM': SVC(probability=True, kernel='rbf'),
    'Gaussian NB': GaussianNB()
}

# Set the number of estimators
n_estimators = 150

# Plot style
plt.figure(figsize=(12, 8))
plt.title('Loss Curves for Different Base Estimators with AdaBoost on Dataset 1')
plt.xlabel('Number of Iterations')
plt.ylabel('Log Loss')
plt.grid(True)

# Train AdaBoost with different base estimators and plot the loss curves
for name, base_estimator in base_estimators.items():
    ada_boost = AdaBoostClassifier(base_estimator=base_estimator,
                                   n_estimators=n_estimators,
                                   learning_rate=0.01,
                                   random_state=123)
    
    ada_boost.fit(X_train, y_train)
    
    # Get the loss for each iteration
    ada_loss = [log_loss(y_test, y_scores) for y_scores in ada_boost.staged_predict_proba(X_test)]
    
    # Plot the loss curve
    plt.plot(np.arange(1, n_estimators + 1), ada_loss, label=f'{name}')

plt.legend()
plt.show()
