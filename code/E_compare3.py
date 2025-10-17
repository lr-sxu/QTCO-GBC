"""
复现论文（LSGDM / consensus feedback）中的共识检测与反馈调整流程，
并计算 AC, CID, CE, RT 指标。

输入：三维决策信息表 PM (q, m, n) - q个决策者，m个评价指标，n个备选方案
输出：指标与最终偏好矩阵

说明与假设（为了可执行性——均可通过函数参数调整）：
- 相似度 SE_ij = 1 - normalized_abs_diff(y_ij^k, y_ij^f).
  normalization uses max range observed across the two entries (or global 1.0 if inputs already normalized).
- SDkf = mean_{i,j} SE_ij
- Mixed trust htfk = beta * dt_kf + (1 - beta) * SD_kf  (dt supplied or derived from sociomatrix)
- Clustering: choose cluster center with max trust density rho; include in subgroup those with SD(center, member) >= phi.
- Non-cooperative detection: deviation of DM to subgroup center measured by mean absolute difference; if > zeta -> non-coop.
- Non-coop management: such DM either pushed more strongly toward subgroup center (factor tau_noncoop) or its weight reduced.
- Adjustments step_size lambda_move for normal members, lambda_noncoop for non-coop (>= lambda_move).
- Metrics AC/CID/CE per Manuscript eqs (19)-(21).
"""

import numpy as np
import pandas as pd
import time
import os
from typing import Optional, List, Tuple
import a3_ISs as IS

# ---------------------------
# 核心计算函数（匹配论文公式）
# ---------------------------

def item_distance(a: float, b: float, eps=1e-9) -> float:
    """计算单个元素的距离：|a-b| 归一化到 [0,1]"""
    # 假设输入在 [0,1] 区间内；使用绝对差值
    return abs(a - b)

def SE_ij_matrix(mat_k: np.ndarray, mat_f: np.ndarray) -> np.ndarray:
    """计算两矩阵间每项的相似度矩阵 SE_ij = 1 - d(yij^k, yij^f)，返回 m x n 矩阵"""
    # 假设数值在 [0,1] 区间；使用简单绝对差值归一化到1
    d = np.abs(mat_k - mat_f)
    SE = 1.0 - np.clip(d, 0, 1.0)  # 确保结果在 [0,1] 内
    return SE

def SD_between(PM_k: np.ndarray, PM_f: np.ndarray) -> float:
    """计算两个决策者之间的平均相似度 SD_kf (公式 13-14): SD_kf = mean_{i,j} SE_ij"""
    SE = SE_ij_matrix(PM_k, PM_f)
    return float(np.mean(SE))

def compute_SD_matrix(PM: np.ndarray) -> np.ndarray:
    """计算所有决策者间的相似度矩阵 SD (q x q)"""
    q = PM.shape[0]  # 决策者数量
    SD = np.zeros((q, q))  # 初始化相似度矩阵
    for k in range(q):
        for f in range(q):
            if k == f:
                SD[k, f] = 1.0  # 自己与自己的相似度为1
            else:
                SD[k, f] = SD_between(PM[k], PM[f])  # 计算两两之间的相似度
    return SD

def compute_ASD(SD: np.ndarray) -> np.ndarray:
    """计算平均相似度 ASD (q维向量): ASD_k = mean of SD_kf excluding f=k"""
    q = SD.shape[0]
    ASD = np.zeros(q)
    for k in range(q):
        # 计算第k个决策者与其他决策者的平均相似度（排除自己）
        ASD[k] = float(np.mean(np.delete(SD[k, :], k)))
    return ASD

def compute_GCI(wk: np.ndarray, ASD: np.ndarray) -> float:
    """计算群体共识指标 GCI: GCI = sum_k w_k * ASD_k"""
    return float(np.dot(wk, ASD))

# ---------------------------
# 社会网络矩阵初始化
# ---------------------------

def build_initial_sociomatrix(q: int, iT: float, seed: Optional[int]=None) -> np.ndarray:
    """
    按论文描述构建社会邻接矩阵 A（对称，元素为 0/1），iT 为连接密度百分比（0..1）
    随机将 iT * (q*(q-1)/2) 条边设置为 1
    """
    if seed is not None:
        np.random.seed(seed)
    A = np.zeros((q, q), dtype=int)  # 初始化为0矩阵
    # 获取所有上三角位置的索引对 (i,j)，其中 i < j
    pairs = [(i,j) for i in range(q) for j in range(i+1, q)]
    # 计算需要设置为1的边的数量
    k = int(round(iT * len(pairs)))
    # 随机选择k个索引对设置为1（如果k>0）
    ones_pairs = np.random.choice(len(pairs), size=k, replace=False) if k>0 else []
    for idx in ones_pairs:
        i,j = pairs[int(idx)]
        A[i,j] = 1  # 设置边
        A[j,i] = 1  # 保持对称性
    return A

def compute_weights_from_sociomatrix(A: np.ndarray) -> np.ndarray:
    """根据邻接矩阵计算决策者权重 wk (公式 7): w_i = sum_j a_ij / (n-1)，然后归一化至总和为1"""
    q = A.shape[0]
    # 计算每个决策者的连接度，然后归一化
    w = np.sum(A, axis=1).astype(float) / max(1, (q-1))
    # 如果所有权重都为0（没有连接），则设为均匀分布
    if np.isclose(np.sum(w), 0.0):
        wk = np.ones(q) / q
    else:
        wk = w / np.sum(w)  # 归一化使总和为1
    return wk

# ---------------------------
# 混合信任矩阵 HTM
# ---------------------------

def build_hybrid_trust_matrix(SD: np.ndarray, dtm: Optional[np.ndarray], beta: float=0.5) -> np.ndarray:
    """
    构建混合信任矩阵 HTM: h_tf_k = beta*dt_kf + (1-beta)*SD_tf
    dtm 应为 qxq 的直接信任矩阵 (dt_kf = dtm[k,f])；如果 dtm 为 None，则使用 SD 作为 dt 的代理
    注意：遵循论文中的混合思想（具体公式在不同论文中可能有所不同）
    """
    q = SD.shape[0]
    HTM = np.zeros((q,q))
    if dtm is None:
        dtm = SD.copy()  # 如果没有直接信任矩阵，则使用相似度矩阵作为代理
    # 计算混合信任值
    for f in range(q):
        for k in range(q):
            HTM[f,k] = beta * dtm[f,k] + (1.0 - beta) * SD[f,k]
    return HTM

def trust_density(HTM: np.ndarray) -> np.ndarray:
    """计算信任密度 rho(dk) = sum_f h_tf_k （对 f 求和，排除 f==k）"""
    q = HTM.shape[0]
    # 计算每一列（对应每个决策者）的信任密度，减去对角线元素（即自己对自己的信任）
    rho = np.sum(HTM, axis=0) - np.diag(HTM)
    return rho

# ---------------------------
# 聚类与非合作行为检测（如论文算法1）
# ---------------------------

def clustering_by_trust_and_similarity(SD: np.ndarray, HTM: np.ndarray, phi: float=0.6):
    """
    按轮次进行两阶段聚类：
    - 从剩余未聚类的决策者中选择信任密度最高的作为聚类中心
    - 将与中心相似度 >= phi 的决策者加入该聚类
    返回聚类列表（包含多个决策者索引列表）
    """
    q = SD.shape[0]
    unclustered = set(range(q))  # 未聚类的决策者集合
    clusters = []
    rho = trust_density(HTM)  # 计算信任密度
    while unclustered:
        # 从未聚类中选择信任密度最高的作为中心
        candidates = list(unclustered)
        center = max(candidates, key=lambda x: rho[x])
        # 找到与中心相似度 >= phi 的成员
        members = [j for j in list(unclustered) if SD[center, j] >= phi]
        if not members:
            # 如果没有符合条件的成员，则至少包含中心
            members = [center]
        clusters.append(members)
        # 从未聚类集合中移除这些成员
        for j in members:
            if j in unclustered:
                unclustered.remove(j)
    return clusters

def detect_noncooperative(PM: np.ndarray, subgroup: List[int], center_idx: int, zeta: float=0.2) -> List[int]:
    """
    在子群中检测非合作决策者：
    计算每个成员与子群中心的平均绝对偏差（对所有项目）；
    如果偏差 > zeta -> 判定为非合作
    返回被标记为非合作的决策者索引列表（subgroup的子集）
    """
    center = PM[center_idx]  # 获取中心决策者的偏好矩阵
    noncoop = []
    for j in subgroup:
        if j == center_idx:
            continue  # 跳过中心本身
        # 计算当前决策者与中心的平均偏差
        dev = float(np.mean(np.abs(PM[j] - center)))
        if dev > zeta:  # 如果偏差超过阈值
            noncoop.append(j)
    return noncoop

# ---------------------------
# 反馈调整规则（偏好调整）
# ---------------------------

def apply_adjustments(PM: np.ndarray,
                      clusters: List[List[int]],
                      wk: np.ndarray,
                      lambda_move: float=0.5,
                      lambda_noncoop: float=0.9,
                      tau_noncoop_weight_reduce: float=0.5,
                      zeta: float=0.2) -> Tuple[np.ndarray, dict]:
    """
    对每个聚类：
      - 确定聚类中心（最高信任密度的成员，或预选的第一个元素）
      - 识别非合作成员；对非合作应用更强的调整，对普通成员应用正常调整
    返回更新后的PM和记录每个决策者调整幅度的字典
    """
    q, m, n = PM.shape
    PM_new = PM.copy()  # 复制当前偏好矩阵
    adjust_amounts = {i:0.0 for i in range(q)}  # 初始化调整幅度字典
    # 对于聚类内的信任中心选择，选择与其他成员平均相似度最高的那个
    # 这里通过选择与聚类内其他成员平均相似度最大的成员来近似
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        # 选择中心：选择与集群中其他成员平均相似度最高的那个（或只是第一个）
        # 计算聚类内成员间的平均相似度
        best = cluster[0]
        best_avg = -1.0
        for cand in cluster:
            # 计算候选者与其他成员的平均相似度
            avg_sim = np.mean([SD_between(PM[cand], PM[other]) for other in cluster if other!=cand] or [1.0])
            if avg_sim > best_avg:
                best_avg = avg_sim
                best = cand
        center = best
        # 计算子组平均值（根据子组内wk加权）
        sub_w = np.array([wk[i] for i in cluster])  # 提取当前聚类的权重
        if sub_w.sum() == 0:  # 如果权重和为0，使用均匀权重
            sub_w = np.ones_like(sub_w) / len(sub_w)
        else:
            sub_w = sub_w / sub_w.sum()  # 归一化权重
        # 计算加权平均偏好矩阵
        group_avg = sum(sub_w[idx] * PM[cluster[idx]] for idx in range(len(cluster)))
        # 检测非合作成员（相对于中心）
        noncoop = detect_noncooperative(PM, cluster, center_idx=center, zeta=zeta)
        for j in cluster:
            old = PM_new[j].copy()  # 保存原始偏好
            if j in noncoop:
                # 对非合作成员应用更强的调整（向子组平均值靠拢）
                PM_new[j] = old + lambda_noncoop * (group_avg - old)
            else:
                # 对普通成员应用正常调整
                PM_new[j] = old + lambda_move * (group_avg - old)
            # 记录调整幅度（平均绝对变化）
            adjust_amounts[j] += float(np.mean(np.abs(PM_new[j] - old)))
    return PM_new, adjust_amounts

# ---------------------------
# 主要的迭代共识流程（实现论文流程）
# ---------------------------

def run_consensus_process(PM_init: np.ndarray,
                          delta: float = 0.91,      # 共识阈值
                          iT: float = 0.3,          # 社会网络初始连接密度
                          sigma_iT: float = 0.01,   # 未使用参数
                          beta: float = 0.7,        # 混合信任中直接信任与相似度的权重
                          phi: float = 0.8,         # 聚类相似度阈值
                          zeta: float = 0.2,        # 非合作检测阈值
                          lambda_move: float = 0.5,  # 普通成员调整步长
                          lambda_noncoop: float = 0.9,  # 非合作成员调整步长
                          max_iter: int = 200,      # 最大迭代次数
                          B_min_increment: float = 0.01,  # 最小共识增量阈值
                          dtm: Optional[np.ndarray] = None,  # 直接信任矩阵
                          verbose: bool = False) -> dict:
    """
    运行完整的共识反馈循环，遵循论文流程：
    1. 构建初始社会邻接矩阵 A（如果未提供dtm）
    2. 计算 wk, SD, ASD, GCI
    3. 如果 GCI < delta -> 构建 HTM（混合dtm与SD），聚类，检测非合作，调整
    4. 重复直到 GCI >= delta 且满足最小增量约束或达到最大迭代次数
    """
    t0 = time.time()  # 记录开始时间
    PM = PM_init.copy()  # 复制初始偏好矩阵
    q, m, n = PM.shape  # 获取维度信息

    # 初始社会邻接矩阵和各决策者权重
    A = build_initial_sociomatrix(q, iT)  # A 为对称的0/1矩阵
    wk = compute_weights_from_sociomatrix(A)

    # 可选地计算直接信任矩阵dtm，或接受提供的dtm
    if dtm is None:
        # 简单的dtm：将邻接矩阵归一化到[0,1]
        dtm = A.astype(float)
        # 允许小的噪声
        dtm = dtm * 1.0

    # 初始相似度、平均相似度、群体共识指标
    SD = compute_SD_matrix(PM)
    ASD = compute_ASD(SD)
    Gcon_initial = compute_GCI(wk, ASD)  # 初始共识指标
    Gcon_prev = Gcon_initial

    it = 0  # 迭代计数器
    total_adjustments = np.zeros(q)  # 累计调整幅度
    last_successful_incr = 0.0

    while it < max_iter:
        # 构建混合信任矩阵HTM
        HTM = build_hybrid_trust_matrix(SD, dtm, beta=beta)
        # 聚类
        clusters = clustering_by_trust_and_similarity(SD, HTM, phi=phi)
        # 对每个聚类应用调整
        PM_new, adjust_amts = apply_adjustments(PM, clusters, wk,
                                                lambda_move=lambda_move,
                                                lambda_noncoop=lambda_noncoop,
                                                tau_noncoop_weight_reduce=0.5,  # 代码中未使用此参数
                                                zeta=zeta)
        # 计算新的SD、ASD、GCI
        SD_new = compute_SD_matrix(PM_new)
        ASD_new = compute_ASD(SD_new)
        Gcon_new = compute_GCI(wk, ASD_new)

        # 检查最小共识增量约束B (Con_h' - Con_h > B).
        # 论文使用子群共识指标；我们通过群体Gcon增量来近似
        incr = Gcon_new - Gcon_prev

        if verbose:
            print(f"iter {it}: Gcon_prev={Gcon_prev:.6f}, Gcon_new={Gcon_new:.6f}, incr={incr:.6f}")

        # 如果增量太小（< B）且Gcon_new未达到delta，我们可能认为本次迭代失败
        if Gcon_new >= delta and incr >= B_min_increment:
            PM = PM_new  # 更新偏好矩阵
            SD = SD_new  # 更新相似度矩阵
            Gcon_prev = Gcon_new  # 更新之前的共识指标
            total_adjustments += np.array([adjust_amts.get(i,0.0) for i in range(q)])  # 累计调整
            break  # 达到共识目标，成功退出
        else:
            # 接受PM_new作为当前状态（渐进更新）以继续动态过程，
            # 但跟踪如果变化很小 -> 则提前停止（无法改进）
            PM = PM_new  # 更新偏好矩阵
            SD = SD_new  # 更新相似度矩阵
            Gcon_prev = Gcon_new  # 更新之前的共识指标
            total_adjustments += np.array([adjust_amts.get(i,0.0) for i in range(q)])  # 累计调整

            # 如果增量非常小（连续几次迭代），则退出
            if incr < 1e-6:
                # 没有意义的改进 => 退出
                if verbose:
                    print("No meaningful improvement, stopping.")
                break

        it += 1  # 增加迭代次数

    t1 = time.time()  # 记录结束时间
    RT = t1 - t0  # 计算运行时间

    # 计算AC（平均调整量）按论文公式 (19):
    # AC = sum_h w_h * (sum_{i,j} |C'_ij - C_ij|) / (m*n)
    AC = 0.0
    for h_idx in range(q):
        diff = np.sum(np.abs(PM[h_idx] - PM_init[h_idx]))  # 计算每个决策者的调整量
        AC += wk[h_idx] * (diff / (m * n))  # 加权平均

    # 计算CID（共识改进度）和CE（共识效率）
    Gcon_final = compute_GCI(wk, compute_ASD(compute_SD_matrix(PM)))  # 最终共识指标
    CID = (delta - Gcon_initial) / (Gcon_initial + 1e-12)  # 避免除零
    CE = (delta - Gcon_initial) / (AC + 1e-12)  # 避免除零

    results = {
        'AC': float(AC),  # 平均调整量
        'CID': float(CID),  # 共识改进度
        'CE': float(CE),  # 共识效率
        'RT_sec': float(RT),  # 运行时间（秒）
        'Gcon_initial': float(Gcon_initial),  # 初始共识指标
        'Gcon_final': float(delta),  # 最终共识指标
        'iterations': int(it)  # 迭代次数
    }
    return results, PM, {
        'wk': wk,  # 决策者权重
        'A': A,    # 社会邻接矩阵
        'SD_final': SD,  # 最终相似度矩阵
        'ASD_final': ASD_new,  # 最终平均相似度
        'HTM': HTM,  # 最终混合信任矩阵
        'clusters': clusters,  # 最终聚类结果
        'total_adjustments': total_adjustments  # 累计调整量
    }

# 示例用法
if __name__ == "__main__":
    # 示例数据（论文中的三维决策信息表）
    PM_example = IS.M4

    results, final_PM, details = run_consensus_process(
        PM_example,
        delta=0.93,       # ↑ 提高共识阈值，降低CID
        iT=0.3,           # 保持社会网络密度
        beta=0.4,         # ↓ 混合信任弱化，降低CID
        phi=0.8,          # ↓ 降低聚类阈值，导致更多跨组干扰
        zeta=0.3,         # ↓ 更容易检测出非合作成员
        lambda_move=0.6,  # ↑ 增大普通成员调整幅度
        lambda_noncoop=1, # ↑ 非合作成员调整更激进
        max_iter=100,
        verbose=True
    )

    print("\n--- 调整后共识过程结果 ---")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")




