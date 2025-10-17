import numpy as np
import time
import a3_ISs as IS

class SNGDM_3D:
    def __init__(self, X0, W0, A0, r_star, Sm_star, R_star, epsilon, theta, alpha, beta, rho, T,
                 adjustment_rate=0.05, self_stubbornness=0.95, noise_scale=0.02):
        """
        修改后的模型（目标：降低最终群体共识度）
        新增/调整参数说明：
          adjustment_rate: eta, 每次观点更新的步长（更小 -> 更新更保守）
          self_stubbornness: sigma, 决策者保留自身观点权重（越大越固执）
          noise_scale: 每次更新时加入的噪声幅度（保留分歧）
        """
        self.k, self.m, self.n = X0.shape
        self.X_initial = np.copy(X0)
        self.X_history = [X0]
        self.W_history = [W0]
        self.A_history = [A0]
        self.r_star = r_star
        self.Sm_star = Sm_star
        self.R_star = R_star
        self.epsilon = epsilon
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.T = T
        self.gamma = 0.6

        # 调整以降低共识度
        self.eta = adjustment_rate
        self.sigma = np.clip(self_stubbornness, 0.0, 0.99)
        self.noise_scale = noise_scale

        self.R_history = [self.epsilon]

    def _compute_similarity_matrix(self, Xt):
        Sm = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                Sm[i, j] = 1 - np.mean(np.abs(Xt[i, :, :] - Xt[j, :, :]))
        return Sm

    def _compute_opinion_reliability(self, t):
        Xt = self.X_history[t]
        Xt_prev = self.X_history[t - 1]
        Rt_prev = self.R_history[t - 1]

        S_t = 1 - np.mean(np.abs(Xt - Xt_prev), axis=(1, 2))

        H = np.where(Rt_prev >= self.r_star)[0]
        WS_t = np.zeros(self.k)
        if len(H) > 0:
            for i in range(self.k):
                weights_H = np.ones(len(H)) / len(H)
                weighted_dist_sq = 0.0
                for idx, h_idx in enumerate(H):
                    dist_matrix = Xt[i, :, :] - Xt_prev[h_idx, :, :]
                    weighted_dist_sq += weights_H[idx] * np.mean(dist_matrix ** 2)
                # 防止负值或大于1
                WS_t[i] = max(0.0, 1 - np.sqrt(weighted_dist_sq))

        Rt = self.theta * S_t + (1 - self.theta) * WS_t
        # 额外约束到 [0,1]
        Rt = np.clip(Rt, 0.0, 1.0)
        return Rt

    def _social_network_evolution(self, t, Sm_t, Rt):
        """更保守的网络演化：提高阈值判断，减少连边"""
        At_prev = self.A_history[t - 1]
        At = np.zeros_like(At_prev)
        for i in range(self.k):
            for j in range(self.k):
                if i == j: continue

                sm_ij = Sm_t[i, j]
                r_j = Rt[j]

                # 更严格的阈值判定（要求更高的相似度与可靠性才能连边）
                if sm_ij >= self.Sm_star and r_j >= self.R_star:
                    At[i, j] = 1.0
                else:
                    # 大多数情况保持无连接，或极小概率为弱连边
                    At[i, j] = 0.0 if np.random.rand() > 0.02 else self.gamma * 0.5
        return At

    def _trust_propagation_and_aggregation(self, t, At, Sm_t, Rt):
        indegree = np.sum(At, axis=0)
        outdegree = np.sum(At, axis=1)
        # 当 k=1 会导致除零，保护性处理
        denom = (2 * (self.k - 1)) if self.k > 1 else 1
        CD_t = (indegree + outdegree) / denom

        comprehensive_trust = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                if i == j: continue
                comprehensive_trust[i, j] = self.alpha * Sm_t[i, j] + self.beta * Rt[j] + \
                                            (1 - self.alpha - self.beta) * CD_t[j]

        T_tilde = comprehensive_trust * At

        # 归一化并引入高自环权重 sigma（提高固执性 -> 降低共识）
        Wt = np.zeros_like(T_tilde)
        for i in range(self.k):
            row = T_tilde[i, :].copy()
            row[i] = 0.0
            row_sum = row.sum()
            if row_sum <= 1e-12:
                Wt[i, :] = 0.0
                Wt[i, i] = 1.0
            else:
                # 他人部分缩放到 (1 - sigma)
                scaled_off_diag = (1.0 - self.sigma) * (row / row_sum)
                Wt[i, :] = scaled_off_diag
                Wt[i, i] = self.sigma

        return Wt

    def _opinion_evolution(self, t, Wt, Sm_t, Rt):
        Xt_prev = self.X_history[t - 1]
        Xt_proposed = np.zeros_like(Xt_prev)

        N2_set = np.where(Rt >= self.R_star)[0]

        for i in range(self.k):
            # 注意：这里 N1 条件严格（要求相似度足够高）
            N1_set = np.where(Sm_t[i, :] >= (1 - self.Sm_star))[0]

            weighted_sum_N1 = np.zeros((self.m, self.n))
            weight_sum_N1 = 0.0
            if len(N1_set) > 0:
                weights_N1 = Wt[i, N1_set]
                for idx, j1 in enumerate(N1_set):
                    weighted_sum_N1 += weights_N1[idx] * Xt_prev[j1, :, :]
                weight_sum_N1 = np.sum(weights_N1)

            weighted_sum_N2 = np.zeros((self.m, self.n))
            weight_sum_N2 = 0.0
            if len(N2_set) > 0:
                weights_N2 = Wt[i, N2_set]
                for idx, j2 in enumerate(N2_set):
                    weighted_sum_N2 += weights_N2[idx] * Xt_prev[j2, :, :]
                weight_sum_N2 = np.sum(weights_N2)

            term1 = weighted_sum_N1 / weight_sum_N1 if weight_sum_N1 > 1e-8 else np.zeros((self.m, self.n))
            term2 = weighted_sum_N2 / weight_sum_N2 if weight_sum_N2 > 1e-8 else np.zeros((self.m, self.n))

            proposed = self.rho * term1 + (1 - self.rho) * term2
            Xt_proposed[i, :, :] = proposed

        # 插值更新（更保守的 eta）
        Xt = np.zeros_like(Xt_prev)
        for i in range(self.k):
            # 基于 eta 插值 + 小幅噪声（防止完全一致）
            base_update = (1.0 - self.eta) * Xt_prev[i, :, :] + self.eta * Xt_proposed[i, :, :]
            noise = np.random.normal(loc=0.0, scale=self.noise_scale, size=(self.m, self.n))
            Xt[i, :, :] = np.clip(base_update + noise, 0.0, 1.0)  # 假设观点在 [0,1] 区间
        return Xt

    def _calculate_group_consensus(self, opinions_tensor):
        avg_std_dev = np.mean(np.std(opinions_tensor, axis=0))
        # 为防止 avg_std_dev 超过1 导致负值，做裁剪
        avg_std_dev = np.clip(avg_std_dev, 0.0, 1.0)
        return 1 - avg_std_dev

    def _calculate_metrics(self, Gcon_initial, Gcon_final, RT):
        AC = np.mean(np.abs(self.X_history[-1] - self.X_initial))
        CID = (Gcon_final - Gcon_initial) / (Gcon_initial + 1e-6)
        CE = (Gcon_final - Gcon_initial) / (AC)
        # 不做额外放缩，这里保留真实关系
        return {"AC": AC, "CID": CID, "CE": CE, "RT": RT}

    def run(self):
        start_time = time.time()
        Gcon_initial = self._calculate_group_consensus(self.X_initial)
        print(f"Initial Group Consensus (Gcon): {Gcon_initial:.4f}")

        for t in range(1, self.T + 1):
            Xt_prev = self.X_history[t - 1]
            Rt_prev = self.R_history[t - 1]

            Sm_t_minus_1 = self._compute_similarity_matrix(Xt_prev)
            At = self._social_network_evolution(t, Sm_t_minus_1, Rt_prev)
            self.A_history.append(At)

            Wt = self._trust_propagation_and_aggregation(t, At, Sm_t_minus_1, Rt_prev)
            self.W_history.append(Wt)

            Xt = self._opinion_evolution(t, Wt, Sm_t_minus_1, Rt_prev)
            self.X_history.append(Xt)

            Rt = self._compute_opinion_reliability(t)
            self.R_history.append(Rt)

            current_gcon = self._calculate_group_consensus(Xt)
            print(f"Iteration {t}: Group Consensus = {current_gcon:.4f}")

            # 不提前轻易停止：即使高共识也继续迭代至 T，以便观察长期演化（如需可改变）
            # if current_gcon > 0.98:
            #     print(f"Consensus reached at iteration {t}.")
            #     break

        end_time = time.time()
        RT = end_time - start_time

        Gcon_final = self._calculate_group_consensus(self.X_history[-1])
        print(f"\nFinal Group Consensus (Gcon'): {Gcon_final:.4f}")

        metrics = self._calculate_metrics(Gcon_initial, Gcon_final, RT)
        return self.X_history[-1], metrics


# ----------------------------
# 主程序示例（参数已调整以降低最终共识度）
# ----------------------------
if __name__ == '__main__':
    X0 = IS.M4  # 你的三维决策信息表 (k, m, n)
    k, m, n = X0.shape

    A0 = np.random.randint(0, 2, size=(k, k))
    np.fill_diagonal(A0, 0)
    W0 = np.random.rand(k, k)
    # 如果希望 W0 行和为1，则进行归一化（但模型里并不强制使用初始 W0）
    W0 = W0 / (W0.sum(axis=1, keepdims=True) + 1e-12)

    epsilon = np.random.rand(k)

    # 调整这些参数以进一步降低群体共识度
    theta = 0.2
    alpha = 0.05
    beta = 0.05
    rho = 0.02

    # 更严格的阈值（更难建立连接）
    r_star = 0.65
    R_star = 0.65
    Sm_star = 0.6

    # 迭代次数：保持一定步数，但因固执与噪声，难以收敛
    T = 10

    # 关键：高固执 + 小步长 + 噪声 -> 最终群体共识更低
    adjustment_rate = 0.13    # 很小的 eta -> 每步变化有限
    self_stubbornness = 0.75   # 高自环权重 -> 更难被他人影响
    noise_scale = 0.01         # 每步加入少量随机扰动，保持差异

    model = SNGDM_3D(
        X0, W0, A0,
        r_star, Sm_star, R_star,
        epsilon, theta, alpha, beta, rho, T,
        adjustment_rate=adjustment_rate,
        self_stubbornness=self_stubbornness,
        noise_scale=noise_scale
    )

    final_opinions, calculated_metrics = model.run()

    print("\n--- Performance Metrics ---")
    print(f"Adjustment Cost (AC): {calculated_metrics['AC']:.6f}")
    print(f"Consensus Improvement Degree (CID): {calculated_metrics['CID']:.6f}")
    print(f"Consensus Efficiency (CE): {calculated_metrics['CE']:.6f}")
    print(f"Running Time (RT): {calculated_metrics['RT']:.6f} seconds")
