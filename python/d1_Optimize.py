# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import b1_Cluster_GB as CGB
# 消融实验
import E_abtion_GB as EGB
from scipy.optimize import minimize
import time
start_time = time.time()  # 记录开始时间
# ================= 工具函数 =================

def aggregate_G(C_set, w):
    """加权聚合矩阵 G = sum_h w^h * C^h，支持 C 为 (N,m,n)"""
    return np.einsum('h,hij->ij', w, C_set)


def consensus_of_h(G, C_h, LM, M_h, alpha=0.4):
    """单专家 h 的共识度： (1 - mean|G - C_h|) * exp(-alpha * |LM - M_h|)"""
    mean_diff = np.mean(np.abs(G - C_h))
    density_penalty = np.exp(-alpha * np.abs(LM - M_h))
    return (1.0 - mean_diff) * density_penalty


def consensus_all(C_set, w, LM, M, alpha=0.4):
    """返回每个 h 的共识度向量 Con(h) 及加权和 Gon"""
    G = aggregate_G(C_set, w)
    Con = np.array([consensus_of_h(G, C_set[h], LM, M[h], alpha) for h in range(C_set.shape[0])])
    Gon = float(np.dot(w, Con))
    return Con, Gon


def pick_s_k_l(t, Con, w, h):
    """
    根据约束：
      s = argmax_{s!=h} w^s
      k = argmax_{k!=h} Con^k
      l = argmax_{l} t_{hl}（若 l==h 则取次大）
    """
    N = len(w)
    idx = np.arange(N)
    mask = (idx != h)

    if not np.any(mask):
        raise ValueError("No valid indices found for s or k.")

    s = idx[mask][np.argmax(w[mask])]
    k = idx[mask][np.argmax(Con[mask])]

    l_mask = mask.copy()
    if not np.any(l_mask):
        raise ValueError("No valid indices found for l.")
    l = idx[l_mask][np.argmax(t[h][l_mask])]
    return s, k, l


def build_C_star_h(C, w, s, k, l, ws_, wk_, wl_, th_sk, th_sl, th_kl):
    """
    构造单个 h 的候选 C^{h*}：
    包括三位专家的加权平均 + 三个量子干涉项（cos θ）
    """
    G = aggregate_G(C, w)
    Cs, Ck, Cl = C[s], C[k], C[l]
    term0 = ws_ * Cs + wk_ * Ck + wl_ * Cl
    eps = 1e-10  # 防止 sqrt(负数) 或除零
    term1 = 2.0 * np.sqrt(np.maximum(ws_ * Cs * wk_ * Ck, eps)) * np.cos(th_sk)
    term2 = 2.0 * np.sqrt(np.maximum(ws_ * Cs * wl_ * Cl, eps)) * np.cos(th_sl)
    term3 = 2.0 * np.sqrt(np.maximum(wk_ * Ck * wl_ * Cl, eps)) * np.cos(th_kl)

    return (0.7 * G + 0.3 * (term0 + term1 + term2 + term3))


def build_C_star_all(theta_vec, C, w, LM, M, t, alpha=0.4):
    """
    构造所有 h 的候选 C_star
    """
    gon_threshold=0.95
    N, m, n = C.shape
    if len(theta_vec) != N * 3:
        raise ValueError(f"theta_vec length {len(theta_vec)} does not match expected {N * 3}")
    theta = np.array(theta_vec).reshape(N, 3)

    Con_base, _ = consensus_all(C, w, LM, M, alpha)
    C_star = C.copy()
    G = aggregate_G(C_star, w)
    for h in range(N):
        try:
            s, k, l = pick_s_k_l(t, Con_base, w, h)
        except ValueError as e:
            print(f"Warning in pick_s_k_l for h={h}: {e}. Skipping C* update.")
            continue

        denom = w[s] + w[k] + w[l]
        if denom == 0:
            continue
        ws_, wk_, wl_ = w[s] / denom, w[k] / denom, w[l] / denom
        th_sk, th_sl, th_kl = theta[h]
        C_candidate = build_C_star_h(C, w, s, k, l, ws_, wk_, wl_, th_sk, th_sl, th_kl)
        if h != k and consensus_of_h(G, C_star[h], LM, M[h], alpha=0.4) < gon_threshold:
            C_star[h] = C_candidate
    return C_star


# ================= 外层优化：目标 & 约束 =================

def fun_gon(theta_vec, C, w, LM, M, t, alpha, gon_threshold):
    """约束：Gon >= gon_threshold"""
    try:
        C_star = build_C_star_all(theta_vec, C, w, LM, M, t, alpha)
        _, Gon_star = consensus_all(C_star, w, LM, M, alpha)
        return Gon_star - gon_threshold
    except Exception:
        return -1e10


def outer_objective_max(theta_vec, C_current, C_original, w, LM, M, t, tau, nu, alpha=0.4):
    """
    外层目标函数（最大化），C_current 会迭代更新
    C_original 永远是初始的 C（目标函数参考基准）
    """
    try:
        C_star = build_C_star_all(theta_vec, C_current, w, LM, M, t, alpha)
    except Exception:
        return 1e10

    # 目标值始终和原始 C 比较
    Delta = np.abs(C_star - C_original)
    N = C_star.shape[0]
    total = 0.0
    for h in range(N):
        Delta_h = Delta[h]
        diff = Delta - Delta_h
        diff[h] = 0.0
        diff1 = Delta_h - Delta
        diff1[h] = 0.0
        penalty = np.maximum(diff, 0.0).sum(axis=0)
        penalty1 = np.maximum(diff1, 0.0).sum(axis=0)
        term_h = Delta_h - (tau[h] / (N - 1.0)) * penalty - (nu[h] / (N - 1.0)) * penalty1
        total += term_h.sum()
    return -total  # 转最小化


def outer_constraints(C, w, LM, M, t, alpha=0.4, gon_threshold=0.95):
    """生成约束列表"""
    return [{
        'type': 'ineq',
        'fun': lambda theta: fun_gon(theta, C, w, LM, M, t, alpha, gon_threshold)
    }]


def optimize_layer(C, w, LM, M, t, tau, nu, alpha=0.4, gon_threshold=0.95):
    """
    外层优化主函数
    每次迭代后 C 会更新为最新的 C_star
    但目标函数始终与原始 C 比较
    """
    N = C.shape[0]
    bounds = [(-np.pi, np.pi)] * (N * 3)
    np.random.seed(42)
    x0 = np.random.uniform(-np.pi, np.pi, N * 3)

    # 可变的 C（每轮迭代更新）
    C_current = C.copy()
    # 固定的原始 C（目标值参考）
    C_original = C.copy()

    constraints = outer_constraints(C_current, w, LM, M, t, alpha, gon_threshold)

    def callback(theta):
        """每次迭代更新 C_current"""
        nonlocal C_current
        C_current = build_C_star_all(theta, C_current, w, LM, M, t, alpha)

    res = minimize(
        fun=outer_objective_max,
        x0=x0,
        args=(C_current, C_original, w, LM, M, t, tau, nu, alpha),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        callback=callback,  # 每轮迭代后更新 C
        options={'maxiter': 1000, 'ftol': 1e-5, 'disp': True}
    )
    return res


# ================= 主程序 =================

if __name__ == "__main__":
    C = CGB.A.copy()
    w = CGB.W.copy()
    M = CGB.m.copy()
    LM = CGB.Lm
    C0 = CGB.A.copy()

    '''
    C = CGB.A.copy()
    w = CGB.W.copy()
    M = CGB.m.copy()
    LM = CGB.Lm
    C0 = CGB.A.copy()
    
    #消融
    C = EGB.A.copy()
    w = EGB.W.copy()
    M = EGB.m.copy()
    LM = EGB.Lm
    C0 = EGB.A.copy()
    '''

    N = C.shape[0]

    '''
    T = np.array([[1.0, 0.389, 0.822, 0.083, 0.593],
                  [0.171, 1.0, 0.013, 0.591, 0.575],
                  [0.77, 0.733, 1.0, 0.92, 0.529],
                  [0.062, 0.578, 0.93, 1.0, 0.304],
                  [0.986, 0.034, 0.196, 0.885, 1.0]], dtype=np.float64)
   '''
    T = np.array([[1.0, 0.389, 0.822, 0.083, 0.593],
                  [0.171, 1.0, 0.013, 0.591, 0.575],
                  [0.77, 0.733, 1.0, 0.92, 0.529],
                  [0.062, 0.578, 0.93, 1.0, 0.304],
                  [0.986, 0.034, 0.196, 0.885, 1.0]], dtype=np.float64)
    tau = np.full(N, 0.2, dtype=float)
    nu = np.full(N, 0.1, dtype=float)
    gon_threshold=0.95
    alpha=0.4
    Con0, Gon0 = consensus_all(C, w, LM, M, alpha=alpha)

    print("开始优化（使用量子干涉式 C*）...")
    result = optimize_layer(C, w, LM, M, T, tau, nu, alpha=alpha, gon_threshold=gon_threshold)

    print("\n" + "=" * 50)
    print("优化过程结束。")
    print("=" * 50)
    print("是否成功:", result.success)
    print("终止原因:", result.message)
    print("最后的目标函数值（负的真实值）:", result.fun)
    print("对应的真实目标值:", -result.fun if np.isfinite(result.fun) else "N/A")

    if hasattr(result, 'x') and result.x is not None:
        theta_final = result.x.reshape(N, 3)
        print("使用的 θ (N×3):\n", theta_final)
        try:
            C_star_final = build_C_star_all(result.x, C, w, LM, M, T, alpha=alpha)
            Con_star_vals_final, Gon_star_final = consensus_all(C_star_final, w, LM, M, alpha=alpha)
            Con_base_final, Gon_base_final = consensus_all(C, w, LM, M, alpha=alpha)

            print("基线共识度 Con(h):", np.round(Con_base_final, 4))
            print("最终 C* 下共识度 Con*(h):", np.round(Con_star_vals_final, 4))
            print("最终加权共识 Gon*(C*):", round(Gon_star_final, 4))
            print("基线加权共识 Gon(C):", round(Gon_base_final, 4))

        except Exception as e:
            print(f"计算最终结果时出错: {e}")
    else:
        print("优化过程未能产生参数估计值。")

    # AC
    AC = 0
    for h in range(5):
        sum = 0
        for i in range(C.shape[1]):
            for j in range(C.shape[2]):
                sum = sum + abs(C_star_final[h][i][j] - C0[h][i][j])
        AC = AC + w[h] * (sum / C.shape[1] / C.shape[2])
    print("AC：", AC)


    CID = (Gon_star_final-Gon_base_final) / Gon_base_final
    print("CID", CID)

    # CE
    CE = (Gon_star_final - Gon_base_final) / AC
    print("CE", CE)

    '''
    存入exel表格
    reshaped_array = np.concatenate(C_star_final, axis=0)
    df = pd.DataFrame(reshaped_array)
    df.to_excel("fdata1.xlsx", index=False, header=False)
    '''

    reshaped_array = np.concatenate(C_star_final, axis=0)
    df = pd.DataFrame(reshaped_array)
    df.to_excel("fdata.xlsx", index=False, header=False)

    end_time = time.time()  # 记录结束时间
    print(f"Execution time: {end_time - start_time:.6f} seconds")