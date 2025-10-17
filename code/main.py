import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint
import matplotlib.pyplot as plt
import b1_Cluster_GB as CGB
import c1_Consensus as CCD


class QuantumDecisionOptimizer:
    def __init__(self, G, w, initial_theta=None):
        """
        初始化量子决策优化器

        参数:
            G: 三维张量 G[k,i,j] 表示第k个决策者在方案i属性j的评分
            w: 权重向量
            initial_theta: 初始相位角矩阵 (可选)
        """
        self.G = np.asarray(G, dtype=np.float64)
        self.w = np.asarray(w, dtype=np.float64)
        self.l, self.m, self.n = self.G.shape  # 决策者数, 方案数, 属性数

        # 初始化参数
        self.initial_theta = np.random.uniform(-1, 1, (self.l, self.l)) * 0.1  # 小随机值
        self.initial_theta = np.triu(self.initial_theta)  # 只保留上三角

        # 预计算条件概率P(a_i|d_k)
        self.P_a_given_d = self._compute_conditional_probs()

    def _compute_conditional_probs(self):
        """计算条件概率 P(a_i|d_k) = G^k_ij的和 / 所有G^k_ij的和"""
        sum_per_decision = np.sum(self.G, axis=(1, 2), keepdims=True)
        return np.sum(self.G, axis=2) / np.squeeze(sum_per_decision, axis=2)

    def _compute_P_a(self, theta):
        """计算方案概率 P(a_i) 的量子公式"""
        # 将扁平化的theta重新构造成矩阵
        theta_mat = np.zeros((self.l, self.l))
        triu_indices = np.triu_indices(self.l, k=1)
        theta_mat[triu_indices] = theta
        theta_mat += theta_mat.T  # 使矩阵对称

        # 线性项
        linear_term = np.sum(self.w[:, np.newaxis] * self.P_a_given_d, axis=0)

        # 量子干涉项
        interference = np.zeros(self.m)
        for k in range(self.l):
            for h in range(k + 1, self.l):
                sqrt_prod = np.sqrt(self.w[k] * self.P_a_given_d[k] * self.w[h] * self.P_a_given_d[h])
                interference += 2 * sqrt_prod * np.cos(theta_mat[k, h])

        return linear_term + interference

    def objective(self, theta):
        """目标函数：最大化方案间概率差异总和"""
        P_a = self._compute_P_a(theta)

        # 计算所有方案对的绝对差异
        diff_matrix = np.abs(P_a[:, None] - P_a)
        np.fill_diagonal(diff_matrix, 0)  # 排除i=j的情况

        return -np.sum(diff_matrix)  # 最小化负值相当于最大化正值

    def solve(self, maxiter=1000):
        """求解优化问题"""
        # 初始值只需要上三角部分（不包括对角线）
        x0 = self.initial_theta[np.triu_indices(self.l, k=1)]

        # 边界条件：相位角范围 [-1,1] 弧度 (~[-57°,57°])
        bounds = [(-1.0, 1.0) for _ in range(len(x0))]

        # 优化求解
        result = minimize(
            self.objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': maxiter}
        )

        # 解析结果
        theta_flat = result.x
        result.theta = np.zeros((self.l, self.l))
        idx = np.triu_indices(self.l, k=1)
        result.theta[idx] = theta_flat
        result.theta += result.theta.T  # 对称矩阵

        result.P_a = self._compute_P_a(theta_flat)
        result.total_difference = -result.fun

        return result

    def visualize(self, result):
        """可视化优化结果"""
        plt.figure(figsize=(12, 5))

        # 1. 方案概率
        plt.subplot(121)
        plt.bar(range(self.m), result.P_a)
        plt.title('Solution Probabilities $P(a_i)$')
        plt.xlabel('Solution Index')
        plt.ylabel('Probability')

        # 2. 相位角矩阵
        plt.subplot(122)
        plt.imshow(result.theta, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Phase Angle (radians)')
        plt.title('Phase Angle Matrix $\\theta_{kh}$')
        plt.tight_layout()
        plt.show()

# 读取数据
def excel_to_3d_array(file_path, N, sheet_name=0, header=None):
    """
    从Excel文件读取数据并转换为三维数组（每隔N行为一个二维子数组）

    参数:
        file_path (str): Excel文件路径
        N (int): 每个二维子数组的行数
        sheet_name (str/int): 工作表名或索引，默认为第一个表
        header (int): 表头行，默认为None（无表头）

    返回:
        np.ndarray: 三维数组，形状为(num_slices, N, num_columns)
    """
    # 1. 读取Excel数据
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header)
        raw_data = df.values
    except Exception as e:
        raise ValueError(f"读取Excel文件失败: {e}")

    # 2. 检查数据有效性
    if len(raw_data.shape) != 2:
        raise ValueError("输入数据必须是二维表格")

    total_rows, num_columns = raw_data.shape
    if total_rows < N:
        raise ValueError(f"总行数({total_rows})小于N({N})，无法分割")

    # 3. 计算可完整分割的子数组数量
    num_slices = total_rows // N
    if total_rows % N != 0:
        print(f"警告: 总行数 {total_rows} 不是 {N} 的整数倍，最后 {total_rows % N} 行将被丢弃")

    # 4. 重塑为三维数组
    valid_rows = num_slices * N
    reshaped_3d = raw_data[:valid_rows].reshape(num_slices, N, num_columns)

    return reshaped_3d

# 获得群体矩阵
def aggregate_G(C_set, w):
    """加权聚合矩阵 G = sum_h w^h * C^h，支持 C 为 (N,m,n)"""
    return np.einsum('h,hij->ij', w, C_set)

# 结果排名
def rank_array(arr):
    # 将数组转换为 NumPy 数组以便处理
    arr = np.array(arr)

    # 获取排序后的索引（降序排列）
    sorted_indices = np.argsort(-arr)

    # 创建一个与原数组大小相同的数组来存储排名
    ranks = np.zeros_like(arr, dtype=int)

    # 初始化排名
    rank = 1
    ranks[sorted_indices[0]] = rank

    # 遍历排序后的索引，分配排名
    for i in range(1, len(sorted_indices)):
        if arr[sorted_indices[i]] == arr[sorted_indices[i - 1]]:
            # 如果当前值与前一个值相同，则排名相同
            ranks[sorted_indices[i]] = rank
        else:
            # 否则，更新排名
            rank += 1
            ranks[sorted_indices[i]] = rank

    return ranks

# 使用示例
if __name__ == "__main__":
    np.random.seed(42)

    # 数据准备
    file_path = "fdata3.xlsx"  # file_path: fdata1,fdata2,fdata3
    m = 10  # 每个二维数组的行数m：9,10,10
    C = excel_to_3d_array(file_path, m)
    C0 = CGB.A
    w = CGB.W
    M = CGB.m.copy()
    LM = CGB.Lm
    alpha = 0.3 # 不同数据集需要修改参数
    G = aggregate_G(C, w)
    Con = np.zeros((C.shape[0], C.shape[1], C.shape[2]))
    for h in range(C.shape[0]):
        for i in range(C.shape[1]):
            for j in range(C.shape[2]):
                Con[h][i][j] = (1 - abs(G[i][j] - C[h][i][j])) * np.exp(-alpha * np.abs(LM - M[h]))


    # 创建优化器
    optimizer = QuantumDecisionOptimizer(Con, w)

    # 求解优化问题
    result = optimizer.solve()

    # 打印结果
    print("=== 优化结果 ===")
    print(f"成功: {result.success}")
    print(f"消息: {result.message}")
    print(f"方案概率: {np.round(result.P_a, 4)}")
    print(f"总差异值: {result.total_difference:.4f}")

    arr = np.round(result.P_a, 4)
    ranks = rank_array(arr)
    print("原始数组:", arr)
    print("排名:", ranks)

    # AD
    AD = CCD.AD
    print("AD：", AD)
    # AC
    AC = 0
    for h in range(5):
        sum = 0
        for i in range(C.shape[1]):
            for j in range(C.shape[2]):
                sum = sum + abs(C0[h][i][j]-C[h][i][j])
        AC = AC + w[h] * (sum / C.shape[1] / C.shape[2])
    print("AC：", AC)
    #CID
    CID = CCD.CID
    print("CID：", CID)





