"""
reproduce_matrix_consensus.py

矩阵版共识复现（q x m x n）：
- 输入: PM (q, m, n) numpy array 或 每个决策者单独的 m x n CSV
- 流程: 按论文矩阵相似度 -> ASD -> GCI -> feedback loop (Blockchain-like acceptances & trust update)
- 输出: AC, CID, CE, RT 以及过程日志

Save as: reproduce_matrix_consensus.py
Run: python reproduce_matrix_consensus.py

Dependencies:
    pip install numpy pandas
"""

import numpy as np
import pandas as pd
import time
import os
from typing import Optional, List, Tuple
import a3_ISs as IS

# ---------------------------
# 工具函数 / 输入输出
# ---------------------------

def load_PM_from_npy(path: str) -> np.ndarray:
    """从numpy .npy文件加载PM，形状为 (q, m, n)。"""
    return np.load(path)

def load_PM_from_folder_csv(folder: str) -> np.ndarray:
    """
    从文件夹加载PM，其中每个文件都是一个决策者的CSV矩阵。
    文件顺序决定决策者索引。CSV应仅包含数值矩阵 (m x n)。
    返回数组形状为 (q, m, n)。
    """
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.csv')])
    mats = []
    for f in files:
        mat = pd.read_csv(os.path.join(folder, f), header=None).values.astype(float)
        mats.append(mat)
    return np.array(mats)

def load_PM_from_long_csv(path: str,
                          col_dm='dm', col_alt='alt', col_attr='attr', col_val='val',
                          q: Optional[int]=None, m: Optional[int]=None, n: Optional[int]=None) -> np.ndarray:
    """
    从长表加载，列包括 [dm, alt, attr, val]。
    dm, alt, attr 可以是0索引或1索引；我们将重新索引到0..q-1等。
    如果某些索引缺失，可以传递所需的 q,m,n（可选）。
    """
    df = pd.read_csv(path)
    # 尝试检测零/一索引
    dm_vals = sorted(df[col_dm].unique())
    alt_vals = sorted(df[col_alt].unique())
    attr_vals = sorted(df[col_attr].unique())
    # 映射到 0..N-1
    dm_map = {v: i for i, v in enumerate(dm_vals)}
    alt_map = {v: i for i, v in enumerate(alt_vals)}
    attr_map = {v: i for i, v in enumerate(attr_vals)}
    q = q or len(dm_map)
    m = m or len(alt_map)
    n = n or len(attr_map)
    PM = np.zeros((q, m, n), dtype=float)
    for _, row in df.iterrows():
        i = dm_map[row[col_dm]]
        j = alt_map[row[col_alt]]
        k = attr_map[row[col_attr]]
        PM[i, j, k] = float(row[col_val])
    return PM

# ---------------------------
# 相似度 / GCI 计算
# ---------------------------

def matrix_pairwise_similarity(mat1: np.ndarray, mat2: np.ndarray) -> float:
    """
    矩阵级相似度 [0,1]:
    SD_kf = 1 - (1/(m*n)) * sum_{i,j} |mat1_ij - mat2_ij|
    假设值已归一化到 [0,1]。如果没有，请在调用前考虑归一化。
    """
    m, n = mat1.shape
    mean_abs_diff = np.mean(np.abs(mat1 - mat2))
    sim = 1.0 - np.clip(mean_abs_diff, 0.0, 1.0)
    return float(sim)

def compute_SD_matrix_from_PM(PM: np.ndarray) -> np.ndarray:
    """
    给定PM (q, m, n)计算SD矩阵 (q x q)。
    SD[k,f] = matrix_pairwise_similarity(PM[k], PM[f])
    """
    q = PM.shape[0]
    SD = np.zeros((q, q), dtype=float)
    for k in range(q):
        for f in range(q):
            if k == f:
                SD[k, f] = 1.0
            else:
                SD[k, f] = matrix_pairwise_similarity(PM[k], PM[f])
    return SD

def compute_ASD_from_SD(SD: np.ndarray) -> np.ndarray:
    """ASD_k = mean_{f != k} SD[k,f]"""
    q = SD.shape[0]
    ASD = np.zeros(q, dtype=float)
    for k in range(q):
        ASD[k] = np.mean(np.delete(SD[k, :], k))
    return ASD

def compute_GCI(wk: np.ndarray, ASD: np.ndarray) -> float:
    """GCI = sum_k wk[k] * ASD[k]"""
    return float(np.dot(wk, ASD))

# ---------------------------
# 社交矩阵与权重
# ---------------------------

def build_social_adjacency(q: int, iT: float, random_seed: int=42) -> np.ndarray:
    """
    构建对称邻接矩阵，大约 iT 比例的可能边存在。
    对角线为零。
    """
    rng = np.random.default_rng(random_seed)
    total_pairs = q * (q - 1) // 2
    k_edges = int(round(iT * total_pairs))
    pairs = [(i, j) for i in range(q) for j in range(i + 1, q)]
    A = np.zeros((q, q), dtype=int)
    if k_edges > 0 and len(pairs) > 0:
        chosen_idx = rng.choice(len(pairs), size=min(k_edges, len(pairs)), replace=False)
        for idx in chosen_idx:
            i, j = pairs[idx]
            A[i, j] = 1
            A[j, i] = 1
    return A

def compute_global_weights_from_adj(A: np.ndarray) -> np.ndarray:
    """
    从邻接行计算全局权重wk: wk_raw = row_sum, 然后归一化为总和1。
    如果全为零，回退到均匀权重。
    """
    row_sum = A.sum(axis=1).astype(float)
    if np.allclose(row_sum, 0.0):
        return np.ones(A.shape[0]) / A.shape[0]
    wk = row_sum / (row_sum.sum())
    return wk

# ---------------------------
# 共识反馈（矩阵级）
# ---------------------------

def matrix_consensus_feedback(PM_init: np.ndarray,
                              delta: float = 0.94,
                              max_iter: int = 200,
                              iT: float = 0.7,
                              alpha: float = 0.5,
                              step_size_normal: float = 0.5,
                              step_size_noncoop: float = 0.8,
                              zeta_noncoop: float = 0.2,
                              eta1: float = 0.05,
                              eta2: float = -0.02,
                              p_internal: float = 0.10,
                              p_external: float = 0.10,
                              mu: float = 0.5,
                              trust_seed: int = 2025,
                              verbose: bool = True):
    """
    矩阵级共识反馈循环。

    输入:
        PM_init: numpy数组形状为 (q, m, n)，值理想情况下在[0,1]中
        delta: 共识阈值 (GCI目标)
        iT: 初始社交邻接中边的比例 (用于权重构造)
        alpha: 混合系数用于混合信任 h[f,k] = alpha*dt_proxy[k] + (1-alpha)*SD[f,k]
        step_size_normal: 向群体平均值调整的步长用于合作决策者
        step_size_noncoop: 针对识别为非合作决策者的调整步长
        zeta_noncoop: 考虑决策者非合作的ASD阈值 (ASD < zeta => 非合作)
        eta1/eta2: 信任更新增量 (增加/减少)
        p_internal / p_external: 背叛概率 (随机因子)
        mu: 合约接受阈值 (全局)
    返回:
        结果字典: AC, CID, CE, RT, Gcon_initial, Gcon_final, iterations
        每次迭代的历史DataFrame
        最终PM数组
    """
    t_start = time.time()
    PM = PM_init.copy().astype(float)
    q, m, n = PM.shape

    # 检查归一化：如果不处于[0,1]，将每个决策者的矩阵归一化到[0,1]并警告
    global_min = PM.min()
    global_max = PM.max()
    if not (0.0 <= global_min and global_max <= 1.0):
        print("警告: PM不在[0,1]中。将每个决策者的矩阵归一化到[0,1]用于相似度计算。")
        for k in range(q):
            mm = PM[k].min()
            Mx = PM[k].max()
            if Mx - mm > 1e-12:
                PM[k] = (PM[k] - mm) / (Mx - mm)
            else:
                PM[k] = 0.5  # 回退常数

    # 初始邻接和权重
    A = build_social_adjacency(q, iT, random_seed=trust_seed)
    wk = compute_global_weights_from_adj(A)  # 总和为1

    # 初始SD, ASD, GCI
    SD = compute_SD_matrix_from_PM(PM)
    ASD = compute_ASD_from_SD(SD)
    Gcon_initial = compute_GCI(wk, ASD)

    # 从邻接初始化信任向量Ti / 可选随机轻微噪声
    rng = np.random.default_rng(trust_seed)
    # 按比例初始化Ti为wi但裁剪并添加抖动
    Ti = 0.5 * (wk * q)  # 缩放到[0,~1]，然后抖动
    Ti = np.clip(Ti + 0.05 * rng.standard_normal(size=q), 0.0, 1.0)

    # 历史记录日志
    history = []
    prev_accept_count = -1
    it = 0
    Gcon = Gcon_initial

    # dt_proxy初始值
    dt_proxy = (ASD - ASD.min()) / (ASD.ptp() + 1e-12)

    # 主循环
    while (Gcon < delta) and (it < max_iter):
        # 构建混合信任 h[f,k]
        h = np.zeros_like(SD)
        for f in range(q):
            for k in range(q):
                h[f, k] = alpha * dt_proxy[k] + (1.0 - alpha) * SD[f, k]
        # 信任密度 rho(dk) = sum_f h[f,k]
        rho = np.sum(h, axis=0)
        least_trusted = int(np.argmin(rho))

        # 通过ASD阈值zeta_noncoop判断最不受信任者是否是非合作的
        is_noncoop = ASD[least_trusted] < zeta_noncoop

        # 计算群体平均矩阵 (按wk加权)
        group_avg = np.tensordot(wk, PM, axes=(0, 0))  # 形状 (m,n)

        # 区块链式接受模拟
        # 每个代理i有psi_i = Ti[i] > gamma_i。为简化，使用gamma_i = median(Ti)
        gamma_i = np.median(Ti)
        psi = (Ti > gamma_i).astype(int)
        internal_flag = (rng.random(q) >= p_internal).astype(int)  # 1表示不背叛
        external_flag = (rng.random(q) >= p_external).astype(int)
        Ei = (psi & internal_flag & external_flag).astype(int)
        accept_ratio = Ei.sum() / q

        # 合约执行规则：如果接受比例 >= mu 则执行
        execute_contract = (accept_ratio >= mu)

        # 应用调整：如果执行且最不受信任者不是非合作的 -> 部分移动 step_size_normal
        # 如果非合作 -> 使用 step_size_noncoop (更强的调整)
        if execute_contract:
            # 调整最不受信任决策者的整个矩阵向群体平均值移动选定步长
            step = step_size_noncoop if is_noncoop else step_size_normal
            old = PM[least_trusted].copy()
            PM[least_trusted] = old + step * (group_avg - old)
            adjusted = True
        else:
            adjusted = False

        # 更新SD, ASD, Gcon
        SD = compute_SD_matrix_from_PM(PM)
        ASD = compute_ASD_from_SD(SD)
        Gcon = compute_GCI(wk, ASD)
        dt_proxy = (ASD - ASD.min()) / (ASD.ptp() + 1e-12)

        # 更新信任Ti：如果接受计数比前一轮增加则Ti += eta1 否则Ti += eta2
        accept_count = int(Ei.sum())
        if prev_accept_count >= 0:
            if accept_count > prev_accept_count:
                Ti = Ti + eta1
            else:
                Ti = Ti + eta2
        else:
            # 第一轮：如果执行则增加，否则减少
            Ti = Ti + (eta1 if adjusted else eta2)
        Ti = np.clip(Ti, 0.0, 1.0)

        # 如果想对信任变化做出反应，更新权重wk：重新计算邻接或保持原始值？
        # 这里保持邻接A固定但按Ti重新计算wk（可选），选择组合：
        # 组合邻接衍生和信任衍生权重: wk = normalize( beta*row_sum + (1-beta)*Ti )
        beta = 0.6
        row_sum = A.sum(axis=1).astype(float)
        wk_raw = beta * (row_sum + 1e-12) + (1.0 - beta) * (Ti + 1e-12)
        wk = wk_raw / wk_raw.sum()

        # 记录日志
        history.append({
            "迭代": it + 1,
            "最不受信任": least_trusted,
            "是否非合作": bool(is_noncoop),
            "接受计数": accept_count,
            "接受比例": float(accept_ratio),
            "已执行": bool(execute_contract),
            "Gcon": float(Gcon),
            "Gcon_初始": float(Gcon_initial),
            "Ti_平均": float(Ti.mean()),
            "已调整": bool(adjusted)
        })

        prev_accept_count = accept_count
        it += 1

    t_end = time.time()
    RT = t_end - t_start

    # 按手稿计算指标 AC, CID, CE (矩阵形式)
    # AC = sum_h w_h * ( sum_{i,j} |C'_ij^h - C_ij^h| ) / (m*n)
    PM_final = PM.copy()
    diff_per_dm = np.sum(np.abs(PM_final - PM_init), axis=(1,2))  # 长度 q
    AC = float(np.dot(wk, diff_per_dm / 3 ))

    # SID: 为安全起见重新计算SD/ASD/Gcon初始值和最终值
    SD_initial = compute_SD_matrix_from_PM(PM_init)
    ASD_initial = compute_ASD_from_SD(SD_initial)
    Gcon_init = compute_GCI(wk, ASD_initial) if True else Gcon_initial  # 注意wk已更改；我们保留使用的最终wk
    SD_final = compute_SD_matrix_from_PM(PM_final)
    ASD_final = compute_ASD_from_SD(SD_final)
    Gcon_final = compute_GCI(wk, ASD_final)

    # CID和CE
    CID = float((Gcon_final - Gcon_init) / (Gcon_init + 1e-12))
    CE = float((Gcon_final - Gcon_init) / (AC + 1e-12))

    results = {
        "AC": AC,
        "CID": CID,
        "CE": CE,
        "RT_秒": float(RT),
        "Gcon_初始": float(Gcon_init),
        "Gcon_最终": float(Gcon_final),
        "迭代次数": int(it)
    }

    history_df = pd.DataFrame(history)
    return {
        "结果": results,
        "历史": history_df,
        "PM_最终": PM_final,
        "PM_初始": PM_init,
        "wk_最终": wk,
        "Ti_最终": Ti,
        "SD_初始": SD_initial,
        "SD_最终": SD_final
    }

# ---------------------------
# 示例 / 类CLI用法
# ---------------------------

def demo_run_with_synthetic():
    PM = IS.M3
    q = PM.shape[0]
    m = PM.shape[1]
    n = PM.shape[2]
    out = matrix_consensus_feedback(
        PM_init=PM,
        delta=0.94,
        max_iter=200,
        iT=0.5,
        alpha=0.6,
        step_size_normal=0.5,
        step_size_noncoop=0.8,
        zeta_noncoop=0.6,
        eta1=0.05,
        eta2=-0.02,
        p_internal=0.10,
        p_external=0.10,
        mu=0.3,
        trust_seed=2025,
        verbose=False
    )
    print("=== 演示结果 ===")
    for k, v in out['结果'].items():
        print(f"{k:12s}: {v}")
    out['历史'].to_csv("history_matrix_consensus.csv", index=False)
    pd.DataFrame([out['结果']]).to_csv("results_matrix_consensus.csv", index=False)
    print("已保存 history_matrix_consensus.csv 和 results_matrix_consensus.csv")
    return out

if __name__ == "__main__":
    # 示例: 运行合成演示
    PM = IS.M1
    q = PM.shape[0]
    m = PM.shape[1]
    n = PM.shape[2]
    out = demo_run_with_synthetic()



