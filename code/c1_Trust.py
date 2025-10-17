import numpy as np
import b1_Cluster_GB as CGB
from scipy.optimize import minimize
import time

start_time = time.time()  # 记录开始时间
# ====== 工具函数 ======

def aggregate_G_prime(C, w):
    """根据权重 w 计算聚合矩阵 G'"""
    return np.einsum('h,hij->ij', w, C)


def consensus_of_h(G, C_h, LM, M_h, alpha=0.6):
    """计算单个专家 h 的共识度"""
    mean_diff = np.mean(np.abs(G - C_h))
    density_diff = np.abs(LM - M_h)
    return (1 - mean_diff) * np.exp(-alpha * density_diff)


def pick_s_k_l(t, Con, w, h):
    """选择 s, k, l 三个专家"""
    indices = np.arange(len(w))
    mask = indices != h
    s = indices[mask][np.argmax(w[mask])]
    k = indices[mask][np.argmax(Con[mask])]
    t_row_masked = t[h, :][mask]  # 对 t 的第 h 行应用 mask
    l_index_in_masked = np.argmax(t_row_masked)  # 在掩码后的行中找最大值索引
    l = indices[mask][l_index_in_masked]  # 映射回原始索引
    return s, k, l


def build_C_star(C, w, s, k, l, ws_, wk_, wl_, theta1, theta2, theta3):
    """构造候选矩阵 C*，并确保 sqrt 安全"""
    G = aggregate_G_prime(C, w)
    Cs, Ck, Cl = C[s], C[k], C[l]
    term0 = ws_ * Cs + wk_ * Ck + wl_ * Cl
    term1 = 2.0 * np.sqrt(np.maximum(ws_ * Cs * wk_ * Ck, 0.0)) * np.cos(theta1)
    term2 = 2.0 * np.sqrt(np.maximum(ws_ * Cs * wl_ * Cl, 0.0)) * np.cos(theta2)
    term3 = 2.0 * np.sqrt(np.maximum(wk_ * Ck * wl_ * Cl, 0.0)) * np.cos(theta3)
    return (0.7 * G + 0.3 * (term0 + term1 + term2 + term3))


# ====== 核心优化 ======

def optimize_in(C_h, C_min_h, C_max_h, h, C_global, w, LM, M):
    """
    内层优化：最小化与原 C_h 的差异，同时满足边界和“共识度 ≥ 0.9”约束
    """
    m, n = C_h.shape

    def objective_in(x):
        x_reshaped = x.reshape((m, n))
        return np.sum(np.abs(x_reshaped - C_h))

    # 修正 bounds
    bounds_in = []
    for i in range(m):
        for j in range(n):
            lower = float(C_min_h[i, j])
            upper = float(C_max_h[i, j])
            if np.isclose(lower, upper):
                lower -= 1e-8
                upper += 1e-8
            bounds_in.append((lower, upper))

    # 共识度约束：当前专家 h 的共识度 >= 之前的
    def consensus_constraint(x):
        C_temp = C_global.copy()
        C_temp[h] = x.reshape((m, n))
        G_temp = aggregate_G_prime(C_temp, w)
        G = aggregate_G_prime(C, w)
        return consensus_of_h(G_temp, C_temp[h], LM, M[h]) - consensus_of_h(G, C[h], LM, M[h]) - 0.05

    constraints_in = [{'type': 'ineq', 'fun': consensus_constraint}]

    # 初始值
    x0 = np.nan_to_num(C_h.flatten(), nan=0.0)

    result = minimize(objective_in, x0, method='SLSQP',
                      bounds=bounds_in, constraints=constraints_in)

    if result.success:
        return result.x.reshape((m, n))
    else:
        return C_h


def optimize_for_T(C, w, LM, M, t):
    """固定一个随机生成的 T 矩阵，运行一次优化"""
    N, m, n = C.shape

    # 三个随机相位角
    theta_sk = np.random.uniform(-np.pi, np.pi)
    theta_sl = np.random.uniform(-np.pi, np.pi)
    theta_kl = np.random.uniform(-np.pi, np.pi)

    # 初始聚合矩阵和共识度
    G = aggregate_G_prime(C, w)
    Con = np.array([consensus_of_h(G, C[h], LM, M[h]) for h in range(N)])

    # C_star 初始化
    C_star = np.zeros((N, m, n))

    # 遍历每个专家 h
    for h in range(N):
        s, k, l = pick_s_k_l(t, Con, w, h)

        # 归一化权重
        denominator = w[s] + w[k] + w[l]
        ws_, wk_, wl_ = w[s] / denominator, w[k] / denominator, w[l] / denominator

        # 构造候选矩阵
        C_candidate = build_C_star(C, w, s, k, l, ws_, wk_, wl_, theta_sk, theta_sl, theta_kl)
        # 如果候选矩阵更优，则替换
        G_star = aggregate_G_prime(C, w)
        C_star[h] = C_candidate

    # 内层每个专家的边界
    C_min = np.minimum(C, C_star)
    C_max = np.maximum(C, C_star)

    # 最终优化矩阵 C'
    C_prime = np.array([
        optimize_in(C[h], C_min[h], C_max[h], h, C.copy(), w, LM, M)
        for h in range(N)
    ])

    # 新的总共识度
    G_prime = aggregate_G_prime(C_prime, w)
    Con_prime = np.array([consensus_of_h(G_prime, C_prime[h], LM, M[h]) for h in range(N)])
    total_obj = np.sum(Con_prime)

    return total_obj, C_prime, Con_prime


# ====== 外层多次随机 T 优化 ======

def optimize_multi_random_T(C, w, LM, M, num_trials=10):
    """外层循环：多次随机生成 T，取总共识度最高的结果"""
    best_obj = -np.inf
    best_C_prime = None
    best_t = None
    best_Con_prime = None

    for trial in range(num_trials):
        # 随机生成 T
        t = np.random.rand(C.shape[0], C.shape[0])

        # 固定 T 优化一次
        total_obj, C_prime, Con_prime = optimize_for_T(C, w, LM, M, t)
        C = C_prime.copy()

        # 更新最优解
        if total_obj > best_obj:
            best_obj = total_obj
            best_C_prime = C_prime
            best_t = t
            best_Con_prime = Con_prime

    print(f"最优总共识度: {best_obj}")
    print(f"各决策者共识度: {best_Con_prime}")
    return best_C_prime, best_t


# ====== 主程序入口 ======
if __name__ == "__main__":
    # 从 b1_Cluster_GB 中导入数据
    C = CGB.A.copy()     # 专家矩阵集合
    w = CGB.W.copy()     # 权重
    M = CGB.m.copy()     # 每个专家的密度
    LM = CGB.Lm          # 全局密度

    print("开始多次随机 T 优化...")
    C_prime_best, T_best = optimize_multi_random_T(C, w, LM, M, num_trials=10)
    print("优化完成！")
    print("最佳 T 矩阵：")
    T_best = np.array(T_best, dtype=float)
    np.fill_diagonal(T_best, 1)
    np.set_printoptions(precision=3, suppress=True)
    print(T_best)

    end_time = time.time()  # 记录结束时间
    print(f"Execution time: {end_time - start_time:.6f} seconds")
