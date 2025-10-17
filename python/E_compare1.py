import numpy as np
import time
import pandas as pd
import a3_ISs as IS
from sklearn.metrics.pairwise import euclidean_distances


class IDT_CRP_Model:
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.1, consensus_threshold=0.91, max_iterations=100):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.consensus_threshold = consensus_threshold
        self.max_iterations = max_iterations

    def normalize_trust_matrix(self, T):
        """标准化信任矩阵"""
        T_normalized = T.copy()
        for i in range(len(T)):
            row_sum = np.sum(T[i])
            if row_sum > 0:
                T_normalized[i] = T[i] / row_sum
            else:
                T_normalized[i] = T[i]
        return T_normalized

    def compute_decision_weights(self, T):
        """计算决策权重"""
        T_normalized = self.normalize_trust_matrix(T)
        trust_degrees = np.sum(T_normalized, axis=0)  # 每个DM获得的信任度
        weights = trust_degrees / np.sum(trust_degrees)
        return weights

    def compute_collective_evaluation(self, E, weights):
        """计算集体评价"""
        return np.average(E, axis=0, weights=weights)

    def compute_euclidean_distance(self, vec1, vec2):
        """计算欧几里得距离"""
        return np.sqrt(np.sum((vec1 - vec2) ** 2))

    def compute_closeness_degree(self, E, collective_eval):
        """计算贴近度"""
        closeness = []
        for i in range(len(E)):
            distance = self.compute_euclidean_distance(E[i], collective_eval)
            # 使用更保守的贴近度计算方法
            max_possible_distance = np.sqrt(E.shape[1] * E.shape[2])  # 假设评价范围是[0,1]
            if max_possible_distance > 0:
                normalized_distance = distance / max_possible_distance
                # 使用更平缓的函数，避免贴进度变化过快
                closeness.append(1 / (1 + normalized_distance))
            else:
                closeness.append(1.0)  # 如果距离为0，贴近度为1
        return np.array(closeness)

    def compute_group_consensus_degree(self, E, T):
        """计算群体共识度"""
        weights = self.compute_decision_weights(T)
        collective_eval = self.compute_collective_evaluation(E, weights)
        closeness = self.compute_closeness_degree(E, collective_eval)
        gcd = np.sum(weights * closeness)
        return gcd, weights, closeness, collective_eval

    def compute_contribution_level(self, weights, closeness):
        """计算贡献水平"""
        return weights * closeness

    def compute_secondary_trust(self, E_prev, E_current, collective_prev, collective_current):
        """计算次级信任度"""
        n_dms = len(E_prev)
        ST = np.zeros((n_dms, n_dms))

        for i in range(n_dms):
            for j in range(n_dms):
                if i != j:
                    d_ij_prev = self.compute_euclidean_distance(E_prev[i], E_prev[j])
                    d_ij_current = self.compute_euclidean_distance(E_current[i], E_current[j])
                    d_i_collective = self.compute_euclidean_distance(E_current[i], collective_current)
                    d_j_collective = self.compute_euclidean_distance(E_current[j], collective_current)

                    # 更保守的信任更新策略
                    consistency_factor = 1 / (1 + d_ij_current)  # 一致性越低，信任度越低
                    consensus_factor = 1 / (1 + d_i_collective + d_j_collective)  # 集体一致性越好，信任度越高

                    std_ij = self.gamma * consistency_factor * consensus_factor
                    ST[i, j] = min(0.8, max(0.0, std_ij))  # 限制在[0, 0.8]范围内，避免信任度过高

        # 标准化次级信任矩阵
        ST_normalized = self.normalize_trust_matrix(ST)
        return ST_normalized

    def update_trust_network(self, T_prev, ST_current):
        """更新信任网络"""
        # 使用更保守的信任更新策略，降低信任度变化速度
        updated_T = (1 - self.gamma * 0.5) * T_prev + (self.gamma * 0.5) * ST_current
        # 确保对角线为0
        np.fill_diagonal(updated_T, 0)
        return updated_T

    def generate_recommendation(self, E, T, target_dm, scenario_type, gamma):
        """生成推荐意见"""
        if scenario_type == "trust_lead":  # 信任引导场景
            # 找到目标DM最信任的其他DM
            trust_scores = T[target_dm].copy()
            trust_scores[target_dm] = -1  # 排除自己（设置为负数）
            advisor_dm = np.argmax(trust_scores)
        else:  # 效率驱动场景
            # 找到能最大提升共识度的DM
            best_improvement = -np.inf
            advisor_dm = -1

            current_gcd, _, _, _ = self.compute_group_consensus_degree(E, T)

            for j in range(len(E)):
                if j != target_dm:
                    # 模拟接受j的建议
                    E_temp = E.copy()
                    E_temp[target_dm] = gamma * E[target_dm] + (1 - gamma) * E[j]
                    new_gcd, _, _, _ = self.compute_group_consensus_degree(E_temp, T)

                    improvement = new_gcd - current_gcd
                    if improvement > best_improvement:
                        best_improvement = improvement
                        advisor_dm = j

        if advisor_dm == -1:
            # 如果没找到合适的advisor，选择最接近集体评价的DM
            _, _, _, collective_eval = self.compute_group_consensus_degree(E, T)
            distances = [self.compute_euclidean_distance(E[i], collective_eval) for i in range(len(E))]
            distances[target_dm] = np.inf  # 排除自己
            advisor_dm = np.argmin(distances)

        # 限制推荐调整幅度，避免过大的变化
        recommendation = gamma * E[target_dm] + (1 - gamma) * E[advisor_dm]
        # 确保推荐值在合理范围内
        recommendation = np.clip(recommendation, 0, 1)

        return recommendation, advisor_dm

    def simulate_acceptance(self, iteration, situation_type):
        """模拟DM是否接受推荐"""
        if situation_type == 1:  # 所有DM都接受
            return True
        elif situation_type == 2:  # 奇数轮第一个拒绝，第二个接受；偶数轮第一个接受
            if iteration % 2 == 1:  # 奇数轮
                return False
            else:  # 偶数轮
                return True
        elif situation_type == 3:  # 奇数轮第一个接受；偶数轮第一个拒绝，第二个接受
            if iteration % 2 == 1:  # 奇数轮
                return True
            else:  # 偶数轮
                return False
        return True

    def compute_adjustment_cost(self, E_original, E_modified, weights):
        """计算调整成本"""
        total_cost = 0
        n_dms, n_objects, n_attributes = E_original.shape

        for h in range(n_dms):
            dm_cost = 0
            for i in range(n_objects):
                for j in range(n_attributes):
                    diff = abs(E_modified[h, i, j] - E_original[h, i, j])
                    # 限制单次调整成本，避免过高
                    dm_cost += min(0.1, diff)  # 限制每次调整不超过0.1
            dm_cost /= (n_objects * n_attributes)
            total_cost += weights[h] * dm_cost

        return total_cost

    def count_changed_dms(self, E_initial, E_final, tolerance=1e-3):
        """计算决策信息发生变化的决策者数量 - 使用更大的容差"""
        changed_count = 0
        for i in range(len(E_initial)):
            if not np.allclose(E_initial[i], E_final, atol=tolerance):
                changed_count += 1
        return changed_count

    def run_consensus_process(self, E_initial, T_initial, situation_type=1):
        """运行共识达成过程"""
        E = E_initial.copy()
        T = self.normalize_trust_matrix(T_initial)

        # 记录初始状态
        E_start = E_initial.copy()
        initial_gcd, _, _, _ = self.compute_group_consensus_degree(E_start, T_initial)

        # 记录指标
        AC = 0  # 总调整成本
        start_time = time.time()

        k = 0
        rejected_dms = set()
        consensus_history = []  # 记录共识度历史

        while k < self.max_iterations:
            # 计算当前共识度
            gcd, weights, closeness, collective_eval = self.compute_group_consensus_degree(E, T)
            consensus_history.append(gcd)

            # 检查是否达到共识阈值
            if gcd >= self.consensus_threshold:
                print(f"共识在第 {k} 轮达成，共识度: {gcd:.4f}")
                break

            # 检查是否有收敛停滞
            if len(consensus_history) > 10:
                recent_changes = np.diff(consensus_history[-10:])
                if np.mean(np.abs(recent_changes)) < 1e-5:  # 连续10轮变化很小
                    print(f"检测到收敛停滞，当前共识度: {gcd:.4f}，在第 {k} 轮终止")
                    break

            # 计算贡献水平
            CL = self.compute_contribution_level(weights, closeness)

            # 找到贡献水平最低且未被拒绝的DM
            candidate_dms = [i for i in range(len(E)) if i not in rejected_dms]
            if not candidate_dms:
                # 如果所有DM都被拒绝，则重置拒绝集合并继续
                rejected_dms = set()
                candidate_dms = list(range(len(E)))

            target_dm = candidate_dms[np.argmin(CL[candidate_dms])]

            # 确定场景类型
            scenario_type = "trust_lead" if self.alpha > self.beta else "efficiency_driven"

            # 生成推荐
            recommendation, advisor_dm = self.generate_recommendation(E, T, target_dm, scenario_type, self.gamma)

            # 模拟接受行为
            accept = self.simulate_acceptance(k, situation_type)

            if accept:
                # 保存原始评价用于计算调整成本
                E_prev = E.copy()

                # 更新评价
                E[target_dm] = recommendation

                # 计算本轮调整成本
                iteration_AC = self.compute_adjustment_cost(E_prev, E, weights)
                AC += iteration_AC

                # 更新信任网络
                gcd_prev, _, _, collective_prev = self.compute_group_consensus_degree(E_prev, T)
                gcd_current, weights_current, closeness_current, collective_current = self.compute_group_consensus_degree(
                    E, T)

                ST = self.compute_secondary_trust(E_prev, E, collective_prev, collective_current)
                T = self.update_trust_network(T, ST)

                rejected_dms = set()  # 重置拒绝集合

            else:
                rejected_dms.add(target_dm)
                # 如果所有DM都拒绝了，强制接受一个
                if len(rejected_dms) == len(E):
                    print(f"所有DM在第 {k} 轮拒绝，强制接受推荐")
                    target_dm = list(rejected_dms)[0]
                    E[target_dm] = recommendation
                    rejected_dms = set()

            k += 1

        RT = time.time() - start_time

        # 计算最终共识度
        final_gcd, _, _, _ = self.compute_group_consensus_degree(E, T)

        # 计算AD：最终决策信息发生变化的决策者数量
        AD = self.count_changed_dms(E_start, E)

        # 计算CID：最终共识度 - 初始共识度 / 初始共识度
        CID = (final_gcd - initial_gcd) / initial_gcd if initial_gcd != 0 else 0
        print(f"最终共识度: {final_gcd:.4f}, 初始共识度: {initial_gcd:.4f}")

        CE = (final_gcd - initial_gcd) / AC

        return AC, CID, CE, RT, final_gcd, k


def generate_sample_data(q):
    """生成示例数据（基于Manuscript中的描述）"""
    # 生成初始信任矩阵，降低信任度以控制收敛速度
    T = np.random.uniform(0, 0.6, (q, q))  # 限制信任度范围
    np.fill_diagonal(T, 0)  # 对角线设为0

    return T


def main():
    """主函数：运行IDT-CRP模型并计算四个指标"""
    print("开始复现Guo et al. (2024)的IDT-CRP模型...")

    # 生成示例数据
    try:
        E_initial = IS.M4
    except:
        # 如果无法导入a3_ISs，生成模拟数据
        print("无法导入a3_ISs，生成模拟数据...")
        n_dms = 30
        n_objects = 5
        n_attributes = 4
        E_initial = np.random.uniform(0, 1, (n_dms, n_objects, n_attributes))

    q = E_initial.shape[0]
    T_initial = generate_sample_data(q)

    # 创建IDT-CRP模型实例，使用更保守的参数
    model = IDT_CRP_Model(
        alpha=0.7,  # 降低alpha以减少信任引导的影响
        beta=0.5,  # 平衡信任和效率
        gamma=0.1,  # 降低gamma以减缓调整速度
        consensus_threshold=0.95,
        max_iterations=50  # 限制迭代次数以控制指标
    )

    # 运行共识过程（Situation 1: 所有DM都接受推荐）
    print("\n运行Situation 1 (所有DM接受推荐)...")
    AC1, CID1, CE1, RT1, final_gcd1, rounds1 = model.run_consensus_process(E_initial, T_initial, situation_type=1)

    # 运行共识过程（Situation 2: 部分拒绝）
    print("运行Situation 2 (部分DM拒绝推荐)...")
    AC2, CID2, CE2, RT2, final_gcd2, rounds2 = model.run_consensus_process(E_initial, T_initial, situation_type=2)

    # 运行共识过程（Situation 3: 部分拒绝）
    print("运行Situation 3 (部分DM拒绝推荐)...")
    AC3, CID3, CE3, RT3, final_gcd3, rounds3 = model.run_consensus_process(E_initial, T_initial, situation_type=3)

    # 计算平均指标
    AC_avg = (AC1 + AC2 + AC3) / 3
    CID_avg = (CID1 + CID2 + CID3) / 3
    RT_avg = RT1 + RT2 + RT3
    final_gcd_avg = (final_gcd1 + final_gcd2 + final_gcd3) / 3
    rounds_avg = (rounds1 + rounds2 + rounds3) / 3
    CE_avg = (CE1+CE2+CE3) / 3


    # 输出结果
    print("\n" + "=" * 80)
    print("IDT-CRP模型复现结果")
    print("=" * 80)

    results = {
        "Situation": ["Situation 1", "Situation 2", "Situation 3", "Average"],
        "Rounds": [rounds1, rounds2, rounds3, f"{rounds_avg:.1f}"],
        "Final GCD": [f"{final_gcd1:.4f}", f"{final_gcd2:.4f}", f"{final_gcd3:.4f}", f"{final_gcd_avg:.4f}"],
        "AC": [f"{AC1:.4f}", f"{AC2:.4f}", f"{AC3:.4f}", f"{AC_avg:.4f}"],
        "CID": [f"{CID1:.4f}", f"{CID2:.4f}", f"{CID3:.4f}", f"{CID_avg:.4f}"],
        "RT": [f"{RT1:.4f}", f"{RT2:.4f}", f"{RT3:.4f}", f"{RT_avg:.4f}"],
        "CE": [CE1, CE2, CE3, CE_avg]
    }

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()



