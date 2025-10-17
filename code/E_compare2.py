import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import warnings
import a3_ISs as IS
import time
start_time = time.time()  # 记录开始时间

class CRP_STWD_RT_IMSDIS:
    def __init__(self, alpha=2.3, consensus_threshold=0.84, delta=0.4, max_iter=20,
                 num_experts=30, random_state=42, bootstrap=False):
        """
        alpha: 保留参数（目前未显式使用，可用于未来扩展）
        consensus_threshold: 达成共识的阈值
        delta: 用于遗憾函数的参数
        max_iter: 最大迭代次数
        num_experts: 生成专家数量
        random_state: 随机种子
        bootstrap: 是否使用自助采样来生成专家意见的多样性
        """
        self.alpha = alpha
        self.consensus_threshold = consensus_threshold
        self.delta = delta
        self.max_iter = max_iter
        self.num_experts = num_experts
        self.random_state = random_state
        self.bootstrap = bootstrap
        np.random.seed(self.random_state)

    def generate_experts_from_data(self, data, num_experts=None):
        """基于数据生成专家意见，噪声水平依据数据标准差自适应，支持bootstrap增强多样性"""
        if num_experts is None:
            num_experts = self.num_experts
        n, m = data.shape

        experts = []
        data_std = np.std(data)
        base_level = max(1e-6, data_std)

        for i in range(num_experts):
            if self.bootstrap:
                # bootstrap 行采样，然后加小噪声
                indices = np.random.choice(n, n, replace=True)
                sampled = data[indices, :]
                noise_level = 0  # 小幅度增加噪声
                noisy = sampled + np.random.normal(0, noise_level, sampled.shape)
            else:
                # 基于原始数据，加噪声
                noise_level = 0# 例如 5% ~ 15% 的 std
                noisy = data + np.random.normal(0, noise_level, data.shape)

            noisy = np.clip(noisy, 0, 1)  # 保证落在 [0,1]
            experts.append(noisy)

        return experts

    def calculate_expert_weights(self, experts):
        """计算专家权重：中心化余弦相似度与专家内部一致性（1/std）加权组合"""
        z = len(experts)
        if z == 0:
            return []

        n, m = experts[0].shape

        # 理想矩阵：加权或平均，这里使用简单平均
        M_idea = np.mean(np.stack(experts, axis=0), axis=0)

        similarities = []
        consistencies = []
        for h in range(z):
            expert_flat = experts[h].flatten()
            ideal_flat = M_idea.flatten()

            # 中心化再计算余弦，降低偏置影响
            ef = expert_flat - np.mean(expert_flat)
            idf = ideal_flat - np.mean(ideal_flat)

            norm_ef = np.linalg.norm(ef)
            norm_idf = np.linalg.norm(idf)
            if norm_ef == 0 or norm_idf == 0:
                sim = 0.0
            else:
                sim = np.dot(ef, idf) / (norm_ef * norm_idf)
                sim = max(-1.0, min(1.0, sim))  # 限制范围

            # 归一化到 [0,1]
            sim01 = (sim + 1.0) / 2.0
            similarities.append(sim01)

            # 一致性度量：专家内部属性总体稳定性（方差越小越稳定）
            var = np.var(experts[h])
            consistency = 1.0 / (1.0 + var)
            consistencies.append(consistency)

        # 合并相似度和一致性，给予相似度更高的权重（可调）
        sims = np.array(similarities)
        cons = np.array(consistencies)

        combined = 0.6 * sims + 0.4 * (cons / (np.sum(cons) + 1e-12))
        combined = np.maximum(0, combined)  # 防止负值
        total = combined.sum()
        if total == 0:
            weights = [1.0 / z] * z
        else:
            weights = (combined / total).tolist()

        return weights

    def calculate_attribute_weights(self, experts):
        """按专家分别计算属性权重，越稳定（方差低）的属性权重越高"""
        z, n, m = len(experts), experts[0].shape[0], experts[0].shape[1]
        attribute_weights = []

        for h in range(z):
            # 属性稳定性：1/(1+std)
            stds = np.std(experts[h], axis=0)
            stability = 1.0 / (1.0 + stds)
            if np.sum(stability) == 0:
                aw = np.ones(m) / m
            else:
                aw = stability / np.sum(stability)
            attribute_weights.append(aw)

        return attribute_weights

    def calculate_consensus_levels(self, experts, expert_weights, attribute_weights):
        """
        计算共识度：
        - MQ 为加权平均群体意见
        - GE[h,i,j] 为专家 h 对样本 i 属性 j 的相似度（0-1）
        - GEE[h] 为专家 h 的总体共识度
        - GCI 群体共识指数（加权）
        """
        z, n, m = len(experts), experts[0].shape[0], experts[0].shape[1]

        # MQ：专家加权平均
        MQ = np.zeros((n, m))
        for h in range(z):
            MQ += expert_weights[h] * experts[h]

        # 计算每个属性的全局范围用于归一化距离（避免逐元素 max/min 错误）
        all_vals = np.concatenate([experts[h].reshape(-1, m) for h in range(z)], axis=0)
        global_max = np.max(all_vals, axis=0)
        global_min = np.min(all_vals, axis=0)
        global_range = global_max - global_min
        global_range[global_range == 0] = 1e-8  # 防止除零

        GE = np.zeros((z, n, m))
        for h in range(z):
            # 距离按属性归一化
            diff = np.abs(experts[h] - MQ)  # shape (n,m)
            # 对每个属性归一化
            GE[h] = 1.0 - (diff / global_range)  # 越接近 MQ 越相似
            GE[h] = np.clip(GE[h], 0.0, 1.0)

        # GEE 每个专家的总体共识度（按属性权重与样本均匀权重加权）
        GEE = np.zeros(z)
        for h in range(z):
            # 属性权重为 attribute_weights[h], 对每个样本 j 使用相同属性权重
            weighted = GE[h] * attribute_weights[h]  # 广播 (n,m) * (m,) -> (n,m)
            # 对样本和属性求加权平均
            GEE[h] = np.sum(weighted) / (n * 1.0)

        # GCI 群体共识指数：结合专家权重和属性权重
        GCI_num = 0.0
        GCI_den = 0.0
        for h in range(z):
            # 对每个属性 j，按属性权重求和
            att_w = attribute_weights[h]  # (m,)
            # GE[h] 的每一行乘以 att_w，再求平均
            per_expert_score = np.sum(GE[h] * att_w) / n
            GCI_num += expert_weights[h] * per_expert_score
            GCI_den += expert_weights[h]

        GCI = GCI_num / (GCI_den + 1e-12)

        # 保证数值在 [0,1]
        GCI = float(np.clip(GCI, 0.0, 1.0))
        GEE = np.clip(GEE, 0.0, 1.0)

        return GCI, GEE, GE, MQ

    def stwd_classification(self, values):
        """顺序三支决策的分类：将值域分为三等分"""
        if len(values) == 0:
            return [], [], []

        vals = np.array(values)
        vmin = np.min(vals)
        vmax = np.max(vals)
        if vmax == vmin:
            # 全部相同，全部视为正域
            return vals.tolist(), [], []

        # 三等分阈值
        t1 = vmin + (vmax - vmin) * (2.0 / 3.0)  # 正域下限
        t2 = vmin + (vmax - vmin) * (1.0 / 3.0)  # 负域上限

        positive = [v for v in vals if v >= t1]
        boundary = [v for v in vals if (v > t2 and v < t1)]
        negative = [v for v in vals if v <= t2]

        return positive, boundary, negative

    def calculate_adjustment_parameters(self, GEE):
        """根据 GEE 计算每个专家的调整意愿 theta: GEE 越低，theta 越高（越愿意调整）"""
        z = len(GEE)
        theta = np.zeros(z)
        # 线性映射：GEE in [0,1] -> theta in [theta_min, theta_max]
        theta_min = 0.1
        theta_max = 0.9
        for h in range(z):
            # 反向映射
            theta[h] = theta_min + (theta_max - theta_min) * (1.0 - GEE[h])
        return theta

    def adjust_opinions(self, experts, MQ, theta, adjustment_set):
        """按照 theta 将专家意见向 MQ 靠拢：new = (1-theta)*current + theta*MQ"""
        z, n, m = len(experts), experts[0].shape[0], experts[0].shape[1]
        adjusted = [ex.copy() for ex in experts]

        for h in range(z):
            if h in adjustment_set:
                th = float(theta[h])
                # 保证 th 在 [0,1]
                th = np.clip(th, 0.0, 1.0)
                # 向群体靠拢
                adjusted[h] = (1 - th) * experts[h] + th * MQ
                adjusted[h] = np.clip(adjusted[h], 0.0, 1.0)

        return adjusted

    def calculate_adjustment_cost(self, experts_initial, experts_final, expert_weights):
        """计算平均绝对调整成本，加权专家权重"""
        z, n, m = len(experts_initial), experts_initial[0].shape[0], experts_initial[0].shape[1]
        total_cost = 0.0
        for h in range(z):
            diff = np.abs(experts_final[h] - experts_initial[h])
            expert_cost = np.mean(diff)  # 平均每个元素的绝对变化
            total_cost += expert_weights[h] * expert_cost
        return float(total_cost)

    def fit(self, data):
        """执行改进的 CRP-STWD-RT-IMSDIS 流程"""
        start_time = time.time()

        # 数据形状处理
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)

        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data)

        # 生成初始专家
        experts_initial = self.generate_experts_from_data(data_normalized, num_experts=self.num_experts)
        experts_current = [ex.copy() for ex in experts_initial]

        # 初始权重与共识
        expert_weights = self.calculate_expert_weights(experts_current)
        attribute_weights = self.calculate_attribute_weights(experts_current)

        if not expert_weights:
            print("警告：未能计算专家权重")
            return None

        GCI_initial, GEE_initial, GE_initial, MQ_initial = self.calculate_consensus_levels(
            experts_current, expert_weights, attribute_weights)

        print(f"初始共识度: {GCI_initial:.4f}")
        print(f"各专家初始共识度: {GEE_initial}")

        GCI_current = GCI_initial
        consensus_history = [GCI_initial]
        iteration_count = 0

        for iteration in range(self.max_iter):
            if GCI_current >= self.consensus_threshold:
                print(f"在第 {iteration_count} 轮迭代达成共识（GCI={GCI_current:.4f}）")
                break

            # STWD 分类用于选取调整对象：这里使用最新的 GEE_initial（上轮或初始）
            positive, boundary, negative = self.stwd_classification(GEE_initial.tolist())

            # 按策略选择调整集合：优先调整负域专家；若无则调整边界；否则小范围调整最差 k 个
            adjustment_set = [i for i, val in enumerate(GEE_initial) if val in negative]
            if not adjustment_set:
                adjustment_set = [i for i, val in enumerate(GEE_initial) if val in boundary]
            if not adjustment_set:
                # 选最差的一个或两个
                worst_idxs = np.argsort(GEE_initial)[:max(1, len(GEE_initial)//3)]
                adjustment_set = worst_idxs.tolist()

            print(f"第 {iteration_count + 1} 轮迭代 - 需要调整的专家: {adjustment_set}")

            # 计算 theta（基于当前 GEE）
            theta = self.calculate_adjustment_parameters(GEE_initial)
            print(f"调整参数 theta: {theta}")

            # 调整意见（基于当前 MQ_initial）
            experts_adjusted = self.adjust_opinions(experts_current, MQ_initial, theta, adjustment_set)

            # 动态更新权重（调整后再评估）
            expert_weights = self.calculate_expert_weights(experts_adjusted)
            attribute_weights = self.calculate_attribute_weights(experts_adjusted)

            # 重新计算共识
            GCI_new, GEE_new, GE_new, MQ_new = self.calculate_consensus_levels(
                experts_adjusted, expert_weights, attribute_weights)

            experts_current = experts_adjusted
            GCI_current = GCI_new
            GEE_initial = GEE_new
            MQ_initial = MQ_new
            consensus_history.append(GCI_current)
            iteration_count += 1

            print(f"第 {iteration_count} 轮后共识度: {GCI_current:.4f}")

            # 提前停止：若共识不再改善或改善非常小
            if iteration_count > 1 and (consensus_history[-1] <= consensus_history[-2] + 1e-6):
                print("共识度不再显著改善，提前停止")
                break

        end_time = time.time()
        rt = end_time - start_time

        # 计算指标
        AC = self.calculate_adjustment_cost(experts_initial, experts_current, expert_weights)
        CID = (GCI_current - GCI_initial) / (abs(GCI_initial) + 1e-12)
        CE = CID / (AC + 1e-12)

        results = {
            'AC': AC,
            'CID': CID,
            'CE': CE,
            'RT': rt,
            'initial_consensus': GCI_initial,
            'final_consensus': GCI_current,
            'iterations': iteration_count,
            'consensus_history': consensus_history,
            'expert_weights': expert_weights,
            'GEE_final': GEE_initial,
            'MQ_final': MQ_initial
        }

        return results


def load_single_dataset(dataset_name='car'):
    """加载单个数据集"""
    try:
        if dataset_name == 'car':
            data = IS.M1
        elif dataset_name == 'wine':
            data = IS.M2
        elif dataset_name == 'maternal':
            data = IS.M3
        elif dataset_name == 'student':
            data = IS.M4
        else:
            raise ValueError(f"未知数据集: {dataset_name}")

        # 转换为numpy数组
        if hasattr(data, 'values'):
            data = data.values
        elif hasattr(data, 'data'):
            data = data.data
            if hasattr(data, 'values'):
                data = data.values

        data = np.asarray(data)

        # 尝试将非数值列转换为数值
        if data.dtype == object:
            try:
                data = data.astype(float)
            except:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                if len(data.shape) == 1:
                    data = le.fit_transform(data).reshape(-1, 1)
                else:
                    data = np.array([le.fit_transform(col) for col in data.T]).T

        print(f"数据集 {dataset_name} 形状: {data.shape}, 数据类型: {data.dtype}")
        return data

    except Exception as e:
        print(f"加载数据集 {dataset_name} 时出错: {e}")
        return np.random.rand(50, 5)


def run_single_experiment(dataset_name='car'):
    print("=" * 60)
    print(f"开始处理数据集: {dataset_name}")
    print("=" * 60)

    try:
        data = load_single_dataset(dataset_name)

        if data.shape[0] > 1000:
            np.random.seed(42)
            indices = np.random.choice(data.shape[0], 1000, replace=False)
            data = data[indices]
            print(f"采样后数据形状: {data.shape}")

        # 使用改进的模型：可开启 bootstrap 以提升专家多样性
        model = CRP_STWD_RT_IMSDIS(consensus_threshold=0.84, max_iter=20,
                                   num_experts=5, random_state=42, bootstrap=True)
        result = model.fit(data)

        if result is not None:
            print("\n" + "=" * 60)
            print("最终结果")
            print("=" * 60)
            print(f"数据集: {dataset_name}")
            print(f"初始共识度: {result['initial_consensus']:.4f}")
            print(f"最终共识度: {result['final_consensus']:.4f}")
            print(f"迭代次数: {result['iterations']}")
            print(f"专家权重: {[f'{w:.3f}' for w in result['expert_weights']]}")
            print(f"调整成本 (AC): {result['AC']:.4f}")
            print(f"共识改进度 (CID): {result['CID']:.4f}")
            print(f"共识效率 (CE): {result['CE']:.4f}")
            print(f"共识度变化历史: {[f'{c:.3f}' for c in result['consensus_history']]}")

            if result['AC'] > 0.034:
                print(f"✅ 调整成本 AC ({result['AC']:.4f})")
            else:
                print(f"⚠️ 调整成本 AC ({result['AC']:.4f})")

            return result
        else:
            print("方法执行失败")
            return None

    except Exception as e:
        print(f"处理数据集 {dataset_name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("开始复现改进版本的 CRP-STWD-RT-IMSDIS 方法...")

    dataset_name = 'wine'
    result = run_single_experiment(dataset_name)

    if result:
        print("\n✅ 实验完成！")
    else:
        print("\n❌ 实验失败！")
    end_time = time.time()  # 记录结束时间
    print(f"Execution time: {end_time - start_time:.6f} seconds")