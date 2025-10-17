import numpy as np
import a3_ISs as IS
import a1_DM_simulation as DS
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

    for dm_idx, label in enumerate(cluster_labels):
        decision_maker_mapping[label].append(dm_idx)

    # 第四步：为每个聚类创建独立的三维矩阵
    clustered_matrices = {}
    for label in decision_maker_mapping:
        dm_indices = decision_maker_mapping[label]
        # 创建 (n_dm_in_cluster, n_samples, n_features) 矩阵
        cluster_matrix = np.array([M[i] for i in dm_indices])
        clustered_matrices[label] = cluster_matrix

    return clustered_matrices, cluster_labels, decision_maker_mapping

def generat_granular_ball(data):
    """
    第二步：生成子组
    参数：
        data: 输入数据 (n_cluster, n_samples, n_features)
    返回：
        out: 包含粒球各项属性
    """
    # 计算小组中心
    center = np.mean(data, axis=0)
    # 计算以方案下属性为中心的粒球的半径
    r = np.mean(abs(data-center), axis=0)
    # 计算以方案下属性为中心的粒球的三个可调多粒度准则
    cov = np.zeros((data.shape[1], data.shape[2]))
    spe = np.zeros((data.shape[1], data.shape[2]))
    h = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            GB = 0
            for k in range(data.shape[0]):
                if abs(data[k][i][j]-center[i][j]) <= r[i][j]:
                    GB = GB + 1
            cov[i][j] = GB / data.shape[0]
            spe[i][j] = 1 - r[i][j]
    d = np.abs(data - center)
    sum = np.sum(d, axis=0, keepdims=True)
    if (sum == 0).all():
        H = 0
    else:
        p = d / sum
        h = -np.sum(p * np.log(p), axis=0)
        # 熵准则的归一化
        scaler = MinMaxScaler()
        H = scaler.fit_transform(h)
        # 计算综合准则
    Q = cov * spe * (1-H)
    # 选择最优作为生成小组代表粒球的半径
    max = 0
    a = 0
    b = 0
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if Q[i][j] > max:
                a = i
                b = j
                max = Q[i][j]
    R = r[a][b]
    # 生成小组信息
    # 初始化结果矩阵和索引列表
    valid_k_indices = []
    filtered_arrays = []
    for k in range(data.shape[0]):
        if abs(data[k, a, b] - center[a, b]) <= R:
            valid_k_indices.append(k)
            filtered_arrays.append(data[k])
    # 将结果转为三维矩阵
    filtered_data = np.stack(filtered_arrays)
    # 获得球中心
    C = np.mean(filtered_data, axis=0)
    # 计算球的质量
    M = max
    return C, valid_k_indices, M

def generat_Lgranular_ball(data,W):
    """
    第四步：生成大组
    参数：
        data: 输入数据 (n_cluster, n_samples, n_features)
    返回：
        out: 包含粒球各项属性
    """
    # 计算大组中心
    center = np.zeros((data.shape[1], data.shape[2]))
    for h in range(data.shape[0]):
        center += W[h] * data[h]
    # 计算以方案下属性为中心的粒球的半径
    r = np.mean(abs(data-center), axis=0)
    # 计算以方案下属性为中心的粒球的三个可调多粒度准则
    cov = np.zeros((data.shape[1], data.shape[2]))
    spe = np.zeros((data.shape[1], data.shape[2]))
    h = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            GB = 0
            for k in range(data.shape[0]):
                if abs(data[k][i][j]-center[i][j]) <= r[i][j]:
                    GB = GB + 1
            cov[i][j] = GB / data.shape[0]
            spe[i][j] = 1 - r[i][j]
    d = np.abs(data - center)
    sum = np.sum(d, axis=0, keepdims=True)
    p = d / sum
    h = -np.sum(p * np.log(p), axis=0)
    # 熵准则的归一化
    min_vals = np.min(h, axis=0)
    max_vals = np.max(h, axis=0)
    range_vals = max_vals - min_vals
    H = (h - min_vals) / range_vals
    # 计算综合准则
    Q = cov * spe * (1-H)
    # 选择最优作为生成小组代表粒球的半径
    max = 0
    a = 0
    b = 0
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if Q[i][j] > max:
                a = i
                b = j
                max = Q[i][j]
    # 计算球的质量
    M = max
    return center, M

# 示例使用
# 获得是初始数据
'''
M = Is.M1； M = Is.M2； M = Is.M3；  M = Is.M4
消融实验: M = DS.M1； M = DS.M2； M = DS.M3；  M = DS.M4
'''
M = IS.M4
print("=== 第一步: K-means聚类 ===")
clustered_M, labels, mapping = cluster_decision_makers(M, n_clusters=5)
print("决策者映射关系:")
for cluster, dms in mapping.items():
    print(f"聚类{cluster}: 包含决策者{dms} -> 矩阵形状: {clustered_M[cluster].shape}")

print("\n=== 第二步: 生成粒球（子组） ===")
# 初始化小组代表矩阵,小组成员个数,小组质量矩阵和小组密度矩阵
A = np.zeros((5, M.shape[1], M.shape[2]))
C = np.zeros(5)
m = np.zeros(5)
i=0
for cluster, dms in mapping.items():
    print(f"聚类{cluster}生成粒球：")
    A[i], L, m[i] = generat_granular_ball(clustered_M[cluster])
    C[i] = len(L)
    print(f"成员个数:", C[i])
    print(f"粒球质量:", m[i])
    print(f"代表矩阵:", A[i])
    i = i + 1

print("\n=== 第三步: 子组权重 ===")
# 初始化小组权重
W = np.zeros(5)
sum = 0
for k in range(5):
    sum = sum + C[k] * m[k]
for k in range(5):
    W[k] =  C[k] * m[k] / sum
print("子组权重为：", W)

print("\n=== 第四步: 生成粒球（大组） ===")
G, Lm = generat_Lgranular_ball(A, W)
print(f"代表矩阵:", G)
print(f"大组质量:", Lm)


