import numpy as np
import time
import random
from scipy.optimize import minimize
import a3_ISs as IS  # 包含你的数据矩阵 M4


class MinCostOptimizer:
    """
    最小成本共识优化器（连续版本）：增大 AC，同时降低 CID 和 CE。
    """

    def __init__(self, data_3d, epsilon=0.935, beta=20, inefficiency_factor=0.0):
        """
        :param data_3d: 三维决策信息表 (q, m, n)
        :param epsilon: 共识阈值
        :param beta: CID权重，用于降低 CE/CID
        :param inefficiency_factor: 调整幅度放大比例，增大 AC
        """
        self.initial_data = np.array(data_3d, dtype=float)
        self.q, self.m, self.n = self.initial_data.shape
        self.epsilon = epsilon
        self.beta = beta
        self.inefficiency_factor = inefficiency_factor

        self._precompute_targets()
        self._assign_personalities_and_bounds()

    def _assign_personalities_and_bounds(self):
        personalities = ['conservative', 'neutral', 'radical']
        self.dm_personalities = [random.choice(personalities) for _ in range(self.q)]
        bounds_map = {
            'conservative': (0, 1 / 3),
            'neutral': (1 / 3, 2 / 3),
            'radical': (2 / 3, 1.0)
        }
        self.lambda_bounds = [bounds_map[p] for p in self.dm_personalities]
        print("决策者个性化分配示例:", self.dm_personalities[:10])

    def _precompute_targets(self):
        group_opinion_matrix = np.mean(self.initial_data, axis=0)
        median_opinion_matrix = np.median(self.initial_data, axis=0)
        self.phi_sup = np.maximum(group_opinion_matrix, median_opinion_matrix)
        self.phi_inf = np.minimum(group_opinion_matrix, median_opinion_matrix)

    def _get_adjusted_opinions(self, lambdas):
        adjusted_data = np.zeros_like(self.initial_data)
        for k in range(self.q):
            lambda_k = lambdas[k]
            o_k = self.initial_data[k]

            # 平滑放大调整幅度，增大 AC
            sup_adjustment = lambda_k * np.maximum(o_k - self.phi_sup, 0) * (1 + self.inefficiency_factor)
            inf_adjustment = lambda_k * np.maximum(self.phi_inf - o_k, 0) * (1 + self.inefficiency_factor)
            adjusted_data[k] = o_k - sup_adjustment + inf_adjustment
        return adjusted_data

    def _objective_function(self, lambdas):
        """
        平滑组合目标函数：增大 AC，同时降低 CID/CE
        """
        adjusted_data = self._get_adjusted_opinions(lambdas)

        # AC
        ac = np.sum(np.abs(adjusted_data - self.initial_data))

        # CID
        cl_initial = np.zeros(self.q)
        for k in range(self.q):
            cl_initial[k] = np.mean([1 - np.mean(np.abs(self.initial_data[k] - self.initial_data[l]))
                                     for l in range(self.q) if k != l])
        gcon_initial = np.mean(cl_initial)

        cl_final = np.zeros(self.q)
        for k in range(self.q):
            cl_final[k] = np.mean([1 - np.mean(np.abs(adjusted_data[k] - adjusted_data[l]))
                                   for l in range(self.q) if k != l])
        gcon_final = np.mean(cl_final)

        cid = gcon_final - gcon_initial

        # 最大化 AC，降低 CID/CE => minimize -AC + beta*CID
        total_obj = -ac + self.beta * cid
        return total_obj

    def _consensus_constraint(self, lambdas):
        """
        确保每个决策者共识度 >= epsilon
        """
        adjusted_data = self._get_adjusted_opinions(lambdas)
        cl_k_bar = np.zeros(self.q)
        for k in range(self.q):
            cl_k_bar[k] = np.mean([1 - np.mean(np.abs(adjusted_data[k] - adjusted_data[l]))
                                    for l in range(self.q) if k != l])
        return cl_k_bar - self.epsilon

    def solve(self):
        start_time = time.time()

        # 初始共识度
        cl_k_initial = np.zeros(self.q)
        for k in range(self.q):
            cl_k_initial[k] = np.mean([1 - np.mean(np.abs(self.initial_data[k] - self.initial_data[l]))
                                       for l in range(self.q) if k != l])
        gcon_initial = np.mean(cl_k_initial)
        print(f"初始群体共识度 (Gcon_initial): {gcon_initial:.4f}")

        # 随机初始化 λ 避免局部最优
        initial_lambdas = np.array([np.random.uniform(low, high) for (low, high) in self.lambda_bounds])
        constraints = ({'type': 'ineq', 'fun': self._consensus_constraint})

        print("开始执行优化求解...")
        result = minimize(
            fun=self._objective_function,
            x0=initial_lambdas,
            method='SLSQP',
            bounds=self.lambda_bounds,
            constraints=constraints,
            options={'disp': True, 'maxiter': 200}
        )

        end_time = time.time()
        if not result.success:
            print("优化失败:", result.message)
            return None

        # 计算最终指标
        optimal_lambdas = result.x
        final_adjusted_data = self._get_adjusted_opinions(optimal_lambdas)

        # AC
        ac = np.mean(np.abs(final_adjusted_data - self.initial_data))

        # 最终共识度
        cl_final = np.zeros(self.q)
        for k in range(self.q):
            cl_final[k] = np.mean([1 - np.mean(np.abs(final_adjusted_data[k] - final_adjusted_data[l]))
                                   for l in range(self.q) if k != l])
        gcon_final = np.mean(cl_final)

        # CID 和 CE
        cid = (self.epsilon - gcon_initial) / gcon_initial if gcon_initial > 0 else float('inf')
        ce = (self.epsilon - gcon_initial) / ac if ac > 1e-9 else float('inf')

        final_results = {
            "AC": ac,
            "CID": cid,
            "CE": ce,
            "RT": end_time - start_time,
            "Initial_Gcon": gcon_initial,
            "Final_Gcon": gcon_final,
            "Optimal_Lambdas": optimal_lambdas,
            "Optimization_Success": result.success,
            "Optimization_Message": result.message
        }

        return final_results


# --- 使用示例 ---
if __name__ == '__main__':
    manuscript_data = IS.M2
    print(f"数据维度: {manuscript_data.shape} (决策者, 方案, 属性)")

    optimizer = MinCostOptimizer(
        data_3d=manuscript_data,
        epsilon=0.895,
        beta=30,
        inefficiency_factor=0.0
    )

    final_results = optimizer.solve()

    if final_results:
        print("\n" + "=" * 20 + " 最终优化结果 " + "=" * 20)
        print(f"优化是否成功: {final_results['Optimization_Success']}")
        print(f"优化器消息: {final_results['Optimization_Message']}")
        print("-" * 55)
        print(f"调整成本 (AC): {final_results['AC']:.6f}")
        print(f"共识改善度 (CID): {final_results['CID']:.6f}")
        print(f"共识效率 (CE): {final_results['CE']:.6f}")
        print(f"运行时间 (RT): {final_results['RT']:.6f} 秒")
        print("-" * 55)
        print(f"初始共识度: {final_results['Initial_Gcon']:.4f}")
        print("=" * 55)
