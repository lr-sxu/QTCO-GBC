import numpy as np
import time
import random
import a3_ISs as IS

class DWConsensusReproducer:
    """
    复现3.pdf中的简化版Deffuant-Weisbuch (DW)共识模型。
    该模型使用Manuscript.pdf中的数据结构和评估指标。
    """

    def __init__(self, data_3d, mu=0.3, sigma=0.5, delta=0.95, max_iterations=100):
        """
        初始化共识模型
        :param data_3d: 三维决策信息表 (q, m, n)，q=决策者数量, m=方案数量, n=属性数量
        :param mu: 意见收敛参数 (0, 0.5]，代表调整速率
        :param sigma: 意见更新阈值，决策者间意见距离小于此值才会更新
        :param delta: 群体共识阈值 (GCI)，达到此值则共识成功
        :param max_iterations: 最大迭代次数
        """
        if not (0 < mu <= 0.5):
            raise ValueError("mu (收敛参数) 应该在 (0, 0.5] 区间内")

        self.initial_data = np.array(data_3d, dtype=float)
        self.q, self.m, self.n = self.initial_data.shape

        self.mu = mu
        self.sigma = sigma
        self.delta = delta
        self.max_iterations = max_iterations

    def _calculate_matrix_distance(self, matrix_a, matrix_b):
        """
        计算两个决策矩阵之间的平均绝对差距离。
        """
        return np.mean(np.abs(matrix_a - matrix_b))

    def calculate_gci(self, current_data):
        """
        计算当前数据的群体共识指数 (GCI)，作为群共识度 (Gcon)。
        参考 3.pdf 中的公式 (13)。
        """
        # 计算集体意见矩阵 (所有决策者矩阵的平均值)
        collective_matrix = np.mean(current_data, axis=0)

        # 计算每个决策者与集体意见的相似度
        similarities = []
        for k in range(self.q):
            dist = self._calculate_matrix_distance(current_data[k], collective_matrix)
            similarity = 1 - dist
            similarities.append(similarity)

        # GCI是所有相似度的平均值
        return np.mean(similarities)

    def run_consensus_process(self):
        """
        执行完整的共识达成过程，并计算四个核心指标。
        """
        start_time = time.time()

        current_data = self.initial_data.copy()

        # --- 指标初始化 ---
        # 1. 计算初始共识度
        gci_initial = self.calculate_gci(current_data)
        print(f"初始群体共识指数 (GCI_initial): {gci_initial:.4f}")

        # 2. 迭代过程
        iteration_count = 0
        for i in range(1, self.max_iterations + 1):
            iteration_count = i

            # 随机选择一个“说服者”
            persuader_idx = random.randint(0, self.q - 1)
            persuader_matrix = current_data[persuader_idx].copy()

            # 其余的作为“监听者”并更新意见
            for listener_idx in range(self.q):
                if listener_idx == persuader_idx:
                    continue

                listener_matrix = current_data[listener_idx]

                # 计算监听者和说服者的意见距离
                distance = self._calculate_matrix_distance(listener_matrix, persuader_matrix)

                # 如果距离在阈值 sigma 内，则更新意见
                if distance <= self.sigma:
                    current_data[listener_idx] = listener_matrix + self.mu * (persuader_matrix - listener_matrix)

            # 检查是否达成共识
            current_gci = self.calculate_gci(current_data)
            print(f"迭代轮数: {i}, 当前GCI: {current_gci:.4f}")

            if current_gci >= self.delta:
                print(f"在第 {i} 轮迭代达成共识！")
                break

        if iteration_count == self.max_iterations and self.calculate_gci(current_data) < self.delta:
            print(f"已达到最大迭代次数 {self.max_iterations}，共识过程终止。")

        end_time = time.time()

        # --- 最终指标计算 ---
        gci_final = self.calculate_gci(current_data)

        # 1. RT (Running Time)
        rt = end_time - start_time

        # 2. AC (Adjustment Cost)
        total_adjustment = 0
        for k in range(self.q):
            # 每个决策者从初始状态到最终状态的总调整量
            total_adjustment += self._calculate_matrix_distance(self.initial_data[k], current_data[k])
        ac = total_adjustment / self.q

        # 3. CID (Consensus Improvement Degree)
        if gci_initial > 0:
            cid = (gci_final - gci_initial) / gci_initial
        else:
            cid = float('inf') if gci_final > 0 else 0

        # 4. CE (Consensus Efficiency)
        if ac > 1e-9:  # 避免除以零
            ce = (gci_final - gci_initial) / ac
        else:
            ce = float('inf') if (gci_final - gci_initial) > 0 else 0

        results = {
            "AC": ac,
            "CID": cid,
            "CE": ce,
            "RT": rt,
            "Iterations": iteration_count,
            "Initial_GCI": gci_initial,
            "Final_GCI": gci_final
        }

        return results


# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 准备你的数据 (从 Manuscript.pdf 的实验中获取)
    # 这里我们使用随机生成的数据作为示例。
    # 请务必将这部分替换为你的真实数据加载代码。

    # 假设有 30 个决策者 (q), 10 个方案 (m), 6 个属性 (n)
    # (此维度参照了Manuscript.pdf案例研究中的描述: "30 related datasets, which are regarded as the evaluation opinions of DAs")

    manuscript_data = IS.M4

    print(f"数据维度: {manuscript_data.shape} (决策者, 方案, 属性)")

    # 2. 设置参数并运行模型
    # 这些参数可以根据 3.pdf 的敏感性分析部分进行调整
    # mu: 意见收敛参数, 3.pdf 中没有直接给出，DW模型通常取0.1-0.5
    # sigma: 意见更新阈值, 3.pdf 中敏感区域为 0.2 < sigma < 0.6
    # delta: 共识阈值, 3.pdf 中设为0.9，Manuscript.pdf 中设为0.95
    reproducer = DWConsensusReproducer(
        data_3d=manuscript_data,
        mu=0.5,
        sigma=0.08,
        delta=0.955,
        max_iterations=200
    )

    final_results = reproducer.run_consensus_process()

    # 3. 打印最终计算的四个指标
    print("\n" + "=" * 20 + " 最终实验结果 " + "=" * 20)
    print(f"调整成本 (AC): {final_results['AC']:.6f}")
    print(f"共识改善度 (CID): {final_results['CID']:.6f}")
    print(f"共识效率 (CE): {final_results['CE']:.6f}")
    print(f"运行时间 (RT): {final_results['RT']:.6f} 秒")
    print("-" * 55)
    print(f"总迭代次数: {final_results['Iterations']}")
    print(f"初始共识度: {final_results['Initial_GCI']:.4f}")
    print(f"最终共识度: {final_results['Final_GCI']:.4f}")
    print("=" * 55)