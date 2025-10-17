import numpy as np
import a3_ISs as IS
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def cluster_decision_makers(M, n_clusters=5, random_state=42):
    """
    参数：
        M: 原始三维矩阵 (n_decision_makers, n_samples, n_features)
        n_clusters: 聚类数量
        random_state: 随机种子

    返回：
        clustered_M: 重组后的三维矩阵 (n_clusters, n_samples*, n_features)
        cluster_labels: 每个决策者的聚类标签
        decision_maker_mapping: 每个聚类包含的决策者编号
        cluster_counts: 每个聚类的决策者个数
    """
    # 第一步：将每个决策者的数据展平为特征向量
    n_dm, n_samples, n_features = M.shape
    flattened_data = np.zeros((n_dm, n_samples * n_features))

    for i in range(n_dm):
        flattened_data[i] = M[i].flatten()

    # 第二步：进行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(flattened_data)

    # 第三步：统计每个聚类中的决策者
    unique_labels = np.unique(cluster_labels)
    decision_maker_mapping = {label: [] for label in unique_labels}
    cluster_counts = {label: 0 for label in unique_labels}

    for dm_idx, label in enumerate(cluster_labels):
        decision_maker_mapping[label].append(dm_idx)
        cluster_counts[label] += 1

    # 第四步：为每个聚类创建独立的三维矩阵
    clustered_matrices = {}
    for label in decision_maker_mapping:
        dm_indices = decision_maker_mapping[label]
        # 创建 (n_dm_in_cluster, n_samples, n_features) 矩阵
        cluster_matrix = np.array([M[i] for i in dm_indices])
        clustered_matrices[label] = cluster_matrix

    return clustered_matrices, cluster_labels, decision_maker_mapping, cluster_counts


def combine_clustered_matrices(clustered_M, mapping):
    """将聚类后的矩阵重新组合成一个三维矩阵"""
    cluster_labels = sorted(clustered_M.keys())
    combined_matrices = []

    for label in cluster_labels:
        combined_matrices.append(clustered_M[label])

    # 沿着第一个轴（决策者维度）拼接
    final_matrix = np.concatenate(combined_matrices, axis=0)
    return final_matrix

# 示例使用
# 获得是初始数据
'''
M = Is.M1； M = Is.M2； M = Is.M3；  M = Is.M4
'''
M = IS.M1
print("=== 第一步: K-means聚类 ===")
clustered_M, labels, mapping, counts = cluster_decision_makers(M, n_clusters=5)

A = np.zeros((5, M.shape[1], M.shape[2]))
i = 0
print("决策者映射关系:")
for cluster, dms in mapping.items():
    print(f"聚类{cluster}: 包含决策者{dms} -> 矩阵形状: {clustered_M[cluster].shape}")
    center = np.mean(clustered_M[cluster], axis=0)
    A[i] = center
    i = i+1

print(f"小组代表矩阵:", A)

# 打印每个聚类的决策者个数
C = np.zeros(5)
i = 0
m = np.zeros(5)
print("\n=== 各聚类决策者个数统计 ===")
for cluster, count in counts.items():
    C[i] = count
    m[i] = 1
    print(f"聚类{cluster}: {C[i]}个决策者")
    i = i+1


print("\n=== 第二步: 子组权重 ===")
# 初始化小组权重
W = np.zeros(5)
sum = 0
for k in range(5):
    sum = sum + C[k] * m[k]
for k in range(5):
    W[k] = C[k] * m[k] / sum
print("子组权重为：", W)

print("\n=== 第三步: 大组 ===")
Lm = 1
G = np.zeros((M.shape[1], M.shape[2]))
for k in range(5):
    G = G + A[k] * W[k]
print(f"代表矩阵:", G)
print(f"大组质量:", Lm)