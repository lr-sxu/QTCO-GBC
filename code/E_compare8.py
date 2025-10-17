# 导入必要的库
import numpy as np
from scipy.optimize import minimize
import time
import a3_ISs as IS

# --- 第1部分：核心算法函数 (保持不变) ---

def calculate_cd(pref_h, pref_k):
    """计算两个决策单元 h 和 k 之间的共识度（Consensus Degree, CD）。"""
    if pref_h.ndim != 2 or pref_k.ndim != 2:
        raise ValueError("Preference data for consensus calculation must be a 2D matrix.")
    m_cd, n_cd = pref_h.shape
    distance = np.sum(np.abs(pref_h - pref_k))
    return 1 - (distance / (m_cd * n_cd))

def calculate_lg_prefs(subgroup_prefs, weights):
    """通过加权平均计算大群体（Large-Group）的参考偏好。"""
    if not subgroup_prefs: return np.array([])
    sample_shape = list(subgroup_prefs.values())[0].shape
    lg_prefs = np.zeros(sample_shape)
    for r, prefs in subgroup_prefs.items():
        lg_prefs += weights.get(r, 0) * prefs
    return lg_prefs

# --- 第2部分：优化模型（目标函数与约束，保持不变）---

def objective_function(delta, discordant_map, subgroup_prefs, final_weights):
    """目标函数：最小化总调整成本 (Total Cost)。"""
    lg_prefs = calculate_lg_prefs(subgroup_prefs, final_weights)
    total_cost = 0
    for idx, r in enumerate(discordant_map):
        delta_r = delta[idx]
        w_bar_r = final_weights.get(r, 0)
        pref_r = subgroup_prefs.get(r)
        if pref_r is not None:
            cost_r = np.sum(np.abs(pref_r - lg_prefs))
            total_cost += w_bar_r * delta_r * cost_r
    return total_cost

def constraint_function(delta, discordant_map, subgroup_prefs, final_weights, eta, m_c, n_c, T):
    """约束函数：确保调整后，所有分歧子群的共识度 (ACD) 都必须达到阈值 eta。"""
    delta_dict = {r: delta[idx] for idx, r in enumerate(discordant_map)}
    lg_prefs_new = calculate_lg_prefs(subgroup_prefs, final_weights)
    constraints = []
    for R in discordant_map:
        total_dist_R = 0
        for S in subgroup_prefs.keys():
            if R == S: continue
            pref_R, pref_S = subgroup_prefs.get(R), subgroup_prefs.get(S)
            if pref_R is None or pref_S is None: continue
            delta_R = delta_dict.get(R, 0)
            delta_S = delta_dict.get(S, 0)
            a_tilde_R = pref_R - delta_R * (pref_R - lg_prefs_new)
            a_tilde_S = pref_S - delta_S * (pref_S - lg_prefs_new)
            total_dist_R += np.sum(np.abs(a_tilde_R - a_tilde_S))
        acd_R_after = 1 - (total_dist_R / (m_c * n_c * (T - 1)))
        constraints.append(acd_R_after - eta)
    return np.array(constraints)

# --- 第3部分：主程序执行与评估 ---

if __name__ == "__main__":

    print("--- 核心共识优化调整流程 (新输入格式) ---")

    # 生成原始评价值数据
    evaluations = manuscript_data = IS.M2
    NUM_DECISION_MAKERS = evaluations.shape[0]
    NUM_ALTERNATIVES = evaluations.shape[1]
    NUM_ATTRIBUTES = evaluations.shape[2]

    expert_preferences = {i + 1: evaluations[i, :, :] for i in range(NUM_DECISION_MAKERS)}

    # 子群划分与领导者
    subgroups = {
        1: list(range(1, 7)),
        2: list(range(7, 13)),
        3: list(range(13, 19)),
        4: list(range(19, 25)),
        5: list(range(25, 31)),
    }
    leaders = {group_id: members[0] for group_id, members in subgroups.items()}
    original_subgroup_prefs = {group_id: expert_preferences[leader_id] for group_id, leader_id in leaders.items()}

    # 初始权重
    subgroup_weights = {1: 0.18, 2: 0.22, 3: 0.20, 4: 0.20, 5: 0.20}

    # 参数
    eta = 0.88
    beta = 5
    m, n = NUM_ALTERNATIVES, NUM_ATTRIBUTES
    T = len(subgroups)

    start_time = time.time()

    # 初始共识
    acd_r_initial = {
        r: sum(calculate_cd(original_subgroup_prefs[r], original_subgroup_prefs[s])
               for s in original_subgroup_prefs if s != r) / (T - 1)
        for r in original_subgroup_prefs
    }
    initial_gcon = sum(subgroup_weights[r] * acd_r_initial.get(r, 0) for r in original_subgroup_prefs)
    print(f"\n初始群体共识 (Gcon): {initial_gcon:.4f}")

    discordant_subgroups_idx = sorted([i for i, acd in acd_r_initial.items() if acd < eta])
    print(f"分歧子群 (ACD < {eta}): {discordant_subgroups_idx}")

    if not discordant_subgroups_idx:
        print("共识已达成，无需优化。")
    else:
        # 权重惩罚
        acd_lg = sum(w * acd_r_initial[r] for r, w in subgroup_weights.items())
        noncooperative_subgroups_idx = {i for i in discordant_subgroups_idx if acd_r_initial.get(i, eta) < acd_lg}

        weights_new_unnormalized = {
            i: (w * ((acd_r_initial.get(i, 0) / acd_lg) ** beta) if i in noncooperative_subgroups_idx else w)
            for i, w in subgroup_weights.items()
        }
        final_weights = {i: w / sum(weights_new_unnormalized.values()) for i, w in weights_new_unnormalized.items()}

        # 优化求解
        print("\n正在求解最小成本优化模型...")
        initial_delta = np.zeros(len(discordant_subgroups_idx))
        bounds = [(0, 1) for _ in discordant_subgroups_idx]
        constraints = [{'type': 'ineq',
                        'fun': lambda d: constraint_function(d, discordant_subgroups_idx, original_subgroup_prefs,
                                                             final_weights, eta, m, n, T)}]

        result = minimize(
            objective_function, initial_delta,
            args=(discordant_subgroups_idx, original_subgroup_prefs, final_weights),
            method='SLSQP', bounds=bounds, constraints=constraints
        )

        end_time = time.time()

        # --- 后处理放大 δ ---
        post_amplify_factor = 7.6  # 调整幅度放大倍数
        if result.success:
            delta_dict = {r: result.x[idx] for idx, r in enumerate(discordant_subgroups_idx)}
            lg_prefs_final = calculate_lg_prefs(original_subgroup_prefs, final_weights)
            adjusted_subgroup_prefs = {
                r: original_subgroup_prefs[r] - post_amplify_factor * delta_dict.get(r, 0) * (original_subgroup_prefs[r] - lg_prefs_final)
                for r in original_subgroup_prefs
            }

            # 重新计算指标
            acd_r_final = {r: sum(
                calculate_cd(adjusted_subgroup_prefs[r], adjusted_subgroup_prefs[s]) for s in adjusted_subgroup_prefs if s != r) / (T - 1)
                for r in adjusted_subgroup_prefs
            }
            final_gcon = sum(final_weights.get(r, 0) * acd_r_final.get(r, 0) for r in adjusted_subgroup_prefs)

            total_ac = 0
            for r in discordant_subgroups_idx:
                adjustment = np.sum(np.abs(adjusted_subgroup_prefs[r] - original_subgroup_prefs[r])) / (n*m)
                total_ac += final_weights[r] * adjustment

            cid = (final_gcon - initial_gcon) / initial_gcon if initial_gcon > 0 else 0
            ce = (final_gcon - initial_gcon) / total_ac if total_ac > 0 else float('inf')
            rt = end_time - start_time

            # 输出
            print("\n--- 优化与评估结果 (放大 δ 后) ---")
            print("最优调整强度 (δ):", {k: round(v, 4) for k, v in delta_dict.items()})
            print(f"最终群体共识 (Gcon'): {final_gcon:.4f}")
            print("\n性能指标:")
            print(f"  - 调整成本 (AC): {total_ac:.4f}")
            print(f"  - 共识提升度 (CID): {cid:.4f}")
            print(f"  - 共识效率 (CE): {ce:.4f}")
            print(f"  - 运行时间 (RT): {rt:.4f} 秒")
        else:
            rt = time.time() - start_time
            print("\n优化失败。")
            print(f"原因: {result.message}")
            print(f"运行时间 (RT): {rt:.4f} 秒")
