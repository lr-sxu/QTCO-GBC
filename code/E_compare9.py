import numpy as np

from scipy.optimize import linprog

import time
import a3_ISs as IS

# 假设 a3_ISs.py 文件存在，并且其中有名为 M2 的三维Numpy数组

# import a3_ISs as IS


def solve_mccm_ap_scipy(o, c, w, d, a, xi, psi):
    """

    使用 SciPy.optimize.linprog 求解线性化的“带有利他偏好的最小成本共识模型 (MCCM-AP)”。

    此模型源于 Liang et al. (2023) 的研究。

    """

    m = len(o)

    max_o, min_o = np.max(o), np.min(o)

    if max_o - min_o < 1e-6:  # 增加一个容差以避免浮点数问题

        return {"status": "Optimal", "total_cost": 0, "consensus_opinion": max_o, "revised_opinions": o}

    # --- 1. 定义变量向量 x ---

    var_count = 7 * m + 1

    # --- 2. 构建目标函数 c ---

    c_obj = np.zeros(var_count)

    c_obj[m + 1:2 * m + 1] = c
    c_obj[2 * m + 1:3 * m + 1] = c

    # --- 3. 构建等式约束 A_eq, b_eq ---

    A_eq = np.zeros((1 + 3 * m, var_count))

    b_eq = np.zeros(1 + 3 * m)

    A_eq[0, 0:m] = w

    A_eq[0, m] = -1

    for i in range(m):
        A_eq[1 + i, i] = 1;
        A_eq[1 + i, m + 1 + i] = 1;
        A_eq[1 + i, 2 * m + 1 + i] = -1

        b_eq[1 + i] = o[i]

        A_eq[1 + m + i, m] = 1;
        A_eq[1 + m + i, 3 * m + 1 + i] = 1;
        A_eq[1 + m + i, 4 * m + 1 + i] = -1

        b_eq[1 + m + i] = o[i]

        A_eq[1 + 2 * m + i, i] = 1;
        A_eq[1 + 2 * m + i, m] = -1;
        A_eq[1 + 2 * m + i, 5 * m + 1 + i] = -1;
        A_eq[1 + 2 * m + i, 6 * m + 1 + i] = 1

        b_eq[1 + 2 * m + i] = 0

    # --- 4. 构建不等式约束 A_ub, b_ub ---

    A_ub = np.zeros((2 * m, var_count))

    b_ub = np.zeros(2 * m)

    b = np.zeros((m, m))

    for i in range(m):

        denominator = sum(1 - abs(o[i] - o[k]) / (max_o - min_o) for k in range(m) if k != i)

        if denominator > 1e-9:

            for j in range(m):

                if i != j: b[i, j] = (1 - abs(o[i] - o[j]) / (max_o - min_o)) / denominator

        else:

            for j in range(m):

                if i != j: b[i, j] = 1 / (m - 1) if m > 1 else 0

    for i in range(m):

        A_ub[i, 5 * m + 1 + i] = 1 + psi

        A_ub[i, 6 * m + 1 + i] = 1 + psi

        b_ub[i] = (max_o - min_o) * (1 - psi)

        term_i_self = (1 - a[i]) / d[i] if d[i] > 1e-9 else 1e9  # 避免除以零

        A_ub[m + i, 3 * m + 1 + i] = term_i_self

        A_ub[m + i, 4 * m + 1 + i] = term_i_self

        for j in range(m):

            if i != j:
                term_j_other = a[i] * b[i, j] / d[j] if d[j] > 1e-9 else 1e9  # 避免除以零

                A_ub[m + i, 3 * m + 1 + j] += term_j_other

                A_ub[m + i, 4 * m + 1 + j] += term_j_other

        b_ub[m + i] = 1 - xi

    # --- 5. 求解 ---

    bounds = (0, None)

    sol = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # --- 6. 整理结果 ---

    if sol.success:

        x = sol.x

        return {"status": "Optimal", "total_cost": sol.fun, "consensus_opinion": x[m], "revised_opinions": x[0:m]}

    else:

        # 当找不到解时，返回原始意见

        return {"status": "Infeasible", "total_cost": 0, "consensus_opinion": np.mean(o), "revised_opinions": o}


def run_consensus_on_3d_matrix(initial_opinions_3d, c, w, d, a, xi, psi):
    """将2D共识模型应用于3D矩阵的每个单元格。"""

    num_das, num_objects, num_attributes = initial_opinions_3d.shape

    adjusted_opinions_3d = np.zeros_like(initial_opinions_3d)

    for i in range(num_objects):

        for j in range(num_attributes):
            opinions_1d = initial_opinions_3d[:, i, j]

            result = solve_mccm_ap_scipy(opinions_1d, c, w, d, a, xi, psi)

            adjusted_opinions_3d[:, i, j] = result["revised_opinions"]

    return adjusted_opinions_3d


def calculate_metrics(initial_opinions_3d, adjusted_opinions_3d, rt):
    """根据 QTCO-GBC 手稿中改编的定义计算AC, CID, 和 CE。"""

    num_das, num_objects, num_attributes = initial_opinions_3d.shape

    def get_gcon(opinions_3d, initial_opinions_for_range):

        total_cd = 0

        for i in range(num_objects):

            for j in range(num_attributes):

                opinions_1d = opinions_3d[:, i, j]

                max_o_initial, min_o_initial = np.max(initial_opinions_for_range[:, i, j]), np.min(
                    initial_opinions_for_range[:, i, j])

                # 分母为0的处理

                range_initial = max_o_initial - min_o_initial

                if range_initial < 1e-6:
                    total_cd += num_das

                    continue

                consensus_opinion = np.mean(opinions_1d)

                cell_cd = 0

                for opinion in opinions_1d:
                    deviation = abs(opinion - consensus_opinion)

                    cd = 1 - (2 * deviation) / (range_initial + deviation)

                    cell_cd += cd

                total_cd += cell_cd

        return total_cd / (num_das * num_objects * num_attributes)

    gcon_initial = get_gcon(initial_opinions_3d, initial_opinions_3d)

    gcon_final = get_gcon(adjusted_opinions_3d, initial_opinions_3d)

    ac = np.mean(np.abs(adjusted_opinions_3d - initial_opinions_3d))

    cid = (psi - gcon_initial) / gcon_initial if gcon_initial != 0 else float('inf')

    ce = (psi - gcon_initial) / ac if ac != 0 else float('inf')

    return {"AC": ac, "CID": cid, "CE": ce, "RT": rt, "Initial_Gcon": gcon_initial, "Final_Gcon": gcon_final}


# --- 主执行模块 ---

if __name__ == "__main__":

    # --- 1. 可调参数设置与解释 ---

    print("--- 参数配置 ---")

    c = [0.0000000124999999999999986] * 30

    w = [1 / 30] * 30

    # --- 参数修改处：放宽约束 ---

    d = [1] * 30  # d: 妥协限制 (原为0.5, 增大以放宽)

    a = [0.02] * 30

    xi = 0.6  # xi (ξ): 满意度阈值 (原为0.75, 降低以放宽)

    psi = 0.715  # psi (ψ): 共识度阈值 (原为0.8, 降低以放宽)

    print("参数已调整为更宽松的设置。")

    print(f"  共识度阈值 (psi) = {psi}")

    print(f"  满意度阈值 (xi) = {xi}")

    print(f"  妥协限制 (d) = {d[0]}\n")

    # --- 2. 加载数据 ---

    print("--- 数据加载 ---")

    initial_data = IS.M4

    print(f"数据维度: {initial_data.shape} (决策者, 对象, 属性)\n")

    # --- 3. 运行共识过程 ---

    print("--- 共识过程 ---")

    print("正在对三维矩阵运行共识模型 (使用 SciPy)...")

    start_time = time.time()

    adjusted_data = run_consensus_on_3d_matrix(initial_data, c, w, d, a, xi, psi)

    end_time = time.time()

    running_time = end_time - start_time

    print("共识过程完成。\n")

    # --- 4. 评估性能与结果输出 ---

    print("--- 评估结果 ---")

    metrics = calculate_metrics(initial_data, adjusted_data, running_time)

    # 检查是否进行了调整

    if np.allclose(initial_data, adjusted_data):

        print("警告: 调整后的数据与初始数据相同，模型可能仍未找到可行解。请尝试进一步放宽参数。")

    else:

        print("信息: 数据已成功调整。")

    print(f"调整成本 (AC): {metrics['AC']:.4f}")

    print(f"   - 解释: 每个意见值的平均调整幅度。越低越好。")

    print(f"共识改善度 (CID): {metrics['CID']:.4f}")

    print(f"   - 解释: 群体共识度的相对提升。越高越好。")

    print(f"共识效率 (CE): {metrics['CE']:.4f}")

    print(f"   - 解释: 每单位调整成本所带来的共识改善。越高越好。")

    print(f"运行时间 (RT): {metrics['RT']:.4f} 秒")

    print(f"   - 解释: 计算所需的时间。越低越好。")
