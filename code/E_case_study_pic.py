import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import a3_ISs as Is
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("QtAgg")
# 设置字体和字号
config = {
    "font.family":'serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
}
matplotlib.rcParams.update(config)

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
    r = np.mean(abs(data - center), axis=0)
    # 计算以方案下属性为中心的粒球的三个可调多粒度准则
    cov = np.zeros((data.shape[1], data.shape[2]))
    spe = np.zeros((data.shape[1], data.shape[2]))
    h = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            GB = 0
            for k in range(data.shape[0]):
                if abs(data[k][i][j] - center[i][j]) <= r[i][j]:
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
    Q = cov * spe * (1 - H)
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


def generat_Lgranular_ball(data, W):
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
    r = np.mean(abs(data - center), axis=0)
    # 计算以方案下属性为中心的粒球的三个可调多粒度准则
    cov = np.zeros((data.shape[1], data.shape[2]))
    spe = np.zeros((data.shape[1], data.shape[2]))
    h = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            GB = 0
            for k in range(data.shape[0]):
                if abs(data[k][i][j] - center[i][j]) <= r[i][j]:
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
    Q = cov * spe * (1 - H)
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


sns.set(style="white", context="talk")


def _clean_axis(ax):
    """去掉坐标轴网格和面板填充"""
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('white');
    ax.yaxis.pane.set_edgecolor('white');
    ax.zaxis.pane.set_edgecolor('white')
    ax.xaxis.pane.fill = False;
    ax.yaxis.pane.fill = False;
    ax.zaxis.pane.fill = False
    # 隐藏刻度
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def resolve_overlaps(centers, radii, min_radius=1e-4, max_iters=100):
    """迭代收缩半径，使球之间不重叠（仅用于可视化）"""
    centers = centers.copy()
    radii = radii.copy()
    n = len(radii)
    for _ in range(max_iters):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < (radii[i] + radii[j]) and d > 1e-9:
                    if radii[i] >= radii[j]:
                        new_r = max((d - radii[j]) * 0.95, min_radius)
                        if new_r < radii[i] - 1e-12:
                            radii[i] = new_r
                            changed = True
                    else:
                        new_r = max((d - radii[i]) * 0.95, min_radius)
                        if new_r < radii[j] - 1e-12:
                            radii[j] = new_r
                            changed = True
                elif d <= 1e-9 and radii[i] > 0 and radii[j] > 0:
                    if radii[i] >= radii[j]:
                        radii[i] = max(radii[i] * 0.6, min_radius);
                        changed = True
                    else:
                        radii[j] = max(radii[j] * 0.6, min_radius);
                        changed = True
        if not changed:
            break
    return radii


def visualize_clustering_only(M, labels, mapping):
    """
    可视化第一步：K-means聚类结果
    """
    n_dm = M.shape[0]
    flattened = M.reshape(n_dm, -1)
    pca = PCA(n_components=3, random_state=42)
    reduced_pts = pca.fit_transform(flattened)  # (n_dm,3)

    clusters = sorted(mapping.keys())
    n_clusters = len(clusters)
    custom_colors = ['indianred', 'orange', 'steelblue', 'violet', 'seagreen']
    palette = custom_colors[:n_clusters]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _clean_axis(ax)

    # 绘制所有决策者，按聚类着色
    for idx, cluster_label in enumerate(clusters):
        members = mapping[cluster_label]
        if len(members) == 0: continue
        pts = reduced_pts[members]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   color=palette[idx], s=50, alpha=0.8, edgecolors='k', linewidth=0.5,
                   label=f"Cluster {cluster_label+1}")
    plt.tight_layout()
    plt.show()


def visualize_granular_balls_only(M, labels, mapping, clustered_centers, included_global_list, sphere_alpha=0.28):
    """
    可视化第二步：粒球生成结果
    球内外点样式统一
    """
    n_dm = M.shape[0]
    flattened = M.reshape(n_dm, -1)
    pca = PCA(n_components=3, random_state=42)
    reduced_pts = pca.fit_transform(flattened)  # (n_dm,3)

    centers_flat = np.array([c.flatten() for c in clustered_centers])
    centers_3d = pca.transform(centers_flat)  # (n_clusters,3)

    clusters = sorted(mapping.keys())
    n_clusters = len(clusters)

    # 计算每个粒球在 PCA 空间的半径
    radii_pca = np.zeros(n_clusters)
    min_vis_radius = 0.2  # 最小可视化半径
    max_vis_radius = 0.8  # 最大可视化半径
    for idx, cluster_label in enumerate(clusters):
        included = included_global_list[idx]
        if len(included) > 0:
            dists = np.linalg.norm(reduced_pts[included] - centers_3d[idx], axis=1)
            radius = dists.max() * 1.1  # 稍微放大一点
        else:
            members = mapping[cluster_label]
            if len(members) > 0:
                dists = np.linalg.norm(reduced_pts[members] - centers_3d[idx], axis=1)
                radius = dists.max() * 1.05
            else:
                radius = 0.3  # 默认值
        # 限制在最小和最大可视化半径范围内
        radii_pca[idx] = np.clip(radius, min_vis_radius, max_vis_radius)

    # 调整球体避免重叠
    viz_radii = resolve_overlaps(centers_3d, radii_pca.copy(), min_radius=min_vis_radius)

    custom_colors = ['indianred', 'orange', 'steelblue', 'violet', 'seagreen']
    palette = custom_colors[:n_clusters]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _clean_axis(ax)

    # 绘制所有决策者，球内外点样式统一
    for idx, cluster_label in enumerate(clusters):
        members = mapping[cluster_label]
        if len(members) == 0: continue
        pts = reduced_pts[members]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   color=palette[idx], s=50, alpha=0.7, edgecolors='k', linewidth=0.5)

    # 绘制粒球
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    for idx in range(n_clusters):
        center = centers_3d[idx]
        r = viz_radii[idx]
        x = center[0] + r * np.cos(u) * np.sin(v)
        y = center[1] + r * np.sin(u) * np.sin(v)
        z = center[2] + r * np.cos(v)
        ax.plot_surface(x, y, z, color=palette[idx], alpha=sphere_alpha, linewidth=0, antialiased=False, shade=True)
        # 粒球中心
        ax.scatter(center[0], center[1], center[2],
                   color=palette[idx], marker='*', s=300, edgecolors='k', linewidth=1.2)
    plt.tight_layout()
    plt.show()

def visualize_pipeline_separate(M, labels, mapping, clustered_centers, included_global_list, G, sphere_alpha=0.28):
    """
    分别展示三个过程的可视化结果
    """
    print("正在生成第一步可视化：K-means聚类结果...")
    visualize_clustering_only(M, labels, mapping)

    print("正在生成第二步可视化：粒球生成结果...")
    visualize_granular_balls_only(M, labels, mapping, clustered_centers, included_global_list, sphere_alpha)


# 示例使用
# 获得是初始数据
'''
M = Is.M1； M = Is.M2； M = Is.M3；  M = Is.M4
'''
M = Is.M3
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
i = 0
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
    W[k] = C[k] * m[k] / sum
print("子组权重为：", W)

print("\n=== 第四步: 生成粒球（大组） ===")
G, Lm = generat_Lgranular_ball(A, W)
print(f"代表矩阵:", G)
print(f"大组质量:", Lm)

# 构造簇的代表中心与全局被包含索引列表
clusters_sorted = sorted(mapping.keys())
clustered_centers = []
included_global_list = []

for cl in clusters_sorted:
    data = clustered_M[cl]  # (n_dm_in_cluster, n_samples, n_features)
    center_mat, local_valid_idxs, mass = generat_granular_ball(data)
    clustered_centers.append(center_mat)
    # 将簇内的局部索引转为全局索引
    global_indices = [mapping[cl][loc] for loc in local_valid_idxs]
    included_global_list.append(global_indices)

# 调用新的分别可视化函数
visualize_pipeline_separate(M, labels, mapping, clustered_centers, included_global_list, G, sphere_alpha=0.28)