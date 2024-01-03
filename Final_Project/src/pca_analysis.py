import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from utils import load_data
import pandas as pd
import seaborn as sns
dataset1 = load_data("project3_dataset1.txt")
dataset2 = load_data("project3_dataset2.txt")
# 加载数据
data = dataset1

X = data["data"]
y = data["target"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用PCA
pca = PCA(n_components=3)
# pca = PCA()

X_pca = pca.fit_transform(X_scaled)

# 将PCA结果转换为DataFrame，方便绘图
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_df['Target'] = y

# 使用Seaborn绘制散点图矩阵
sns.pairplot(pca_df, hue='Target', vars=[f'PC{i+1}' for i in range(X_pca.shape[1])])
plt.title('PCA Components Scatter Plot Matrix')
plt.show()

# 创建3D图形
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制每个类别的数据点
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(data["target_names"]):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label="target:" + str(target_name))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Dataset 1')
plt.legend()
plt.show()



# 绘制每个主成分的方差贡献率
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Variance')
plt.title('PCA Variance of Dataset 1')
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.show()
