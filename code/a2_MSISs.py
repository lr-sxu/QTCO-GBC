import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import a1_DM_simulation as DMS
from sklearn.cluster import KMeans

# 固定随机种子，保证结果可复现
np.random.seed(0)
torch.manual_seed(0)

# ===================== 模型结构 =====================
class Encoder(nn.Module):
    """
    编码器：将输入样本映射到潜在空间
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入 -> 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)  # 隐藏层 -> 潜在向量
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """
    解码器：将潜在空间向量映射回原空间（重构）
    """

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # 输出限制在 [0,1]，适合归一化数据
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    判别器：判断潜在向量是否来自真实分布
    """

    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出为概率
        )

    def forward(self, z):
        return self.net(z)


# ===================== 单个二维矩阵的处理流程 =====================
def process_single_matrix(X_single, sample,  hidden_dim=64, latent_dim=10, epochs=1000):
    """
    对单个二维矩阵（样本数 × 特征数）进行多尺度 AAE 编码、聚类和权重融合
    """
    input_dim = X_single.shape[1]  # 特征维度

    # 初始化模型
    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, input_dim)
    discriminator = Discriminator(latent_dim, hidden_dim)

    # 初始化优化器
    opt_enc = optim.Adam(encoder.parameters(), lr=1e-3)
    opt_dec = optim.Adam(decoder.parameters(), lr=1e-3)
    opt_dis = optim.Adam(discriminator.parameters(), lr=1e-3)

    # 批量大小（不超过样本总数）
    batch_size = min(16, len(X_single))

    # ========== 训练 AAE ==========
    for _ in range(epochs):
        # 随机采样一个批次
        idx = np.random.choice(len(X_single), batch_size, replace=False)
        batch = X_single[idx]

        # (1) 重构训练
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        z_fake = encoder(batch)  # 编码
        X_recon = decoder(z_fake)  # 解码
        loss_recon = nn.MSELoss()(X_recon, batch)  # 重构损失
        loss_recon.backward()
        opt_enc.step()
        opt_dec.step()

        # (2) 判别器训练
        opt_dis.zero_grad()
        z_real = torch.randn(batch_size, latent_dim)  # 从先验分布采样
        d_real = discriminator(z_real)  # 真实潜在向量判别结果
        d_fake = discriminator(encoder(batch).detach())  # 生成的潜在向量判别结果
        # 对抗损失（判别器希望 d_real→1, d_fake→0）
        loss_dis = -torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))
        loss_dis.backward()
        opt_dis.step()

        # (3) 编码器对抗训练（让生成的潜在向量更像真实分布）
        opt_enc.zero_grad()
        z_fake = encoder(batch)
        d_fake = discriminator(z_fake)
        loss_adv = -torch.mean(torch.log(d_fake + 1e-8))  # 编码器希望 d_fake→1
        loss_adv.backward()
        opt_enc.step()

    # ========== 获取全体样本的潜在向量 ==========
    with torch.no_grad():
        z_all = encoder(X_single).numpy()

    # ========== 多尺度聚类 ==========
    k_mid, k_coarse = math.ceil(sample/2), math.ceil(sample/4)  # 中、粗尺度的聚类簇数
    kmeans_mid = KMeans(n_clusters=k_mid, random_state=0).fit(z_all)
    kmeans_coarse = KMeans(n_clusters=k_coarse, random_state=0).fit(z_all)

    # 将标签转换为聚类中心向量
    z_fine = torch.tensor(z_all, dtype=torch.float32)  # 细尺度：原始潜在向量
    z_mid = torch.tensor(kmeans_mid.cluster_centers_[kmeans_mid.labels_], dtype=torch.float32)
    z_coarse = torch.tensor(kmeans_coarse.cluster_centers_[kmeans_coarse.labels_], dtype=torch.float32)

    # ========== 解码回原空间 ==========
    with torch.no_grad():
        x_fine = decoder(z_fine)  # 细尺度重构
        x_mid = decoder(z_mid)  # 中尺度重构
        x_coarse = decoder(z_coarse)  # 粗尺度重构

    # ========== 学习融合权重 ==========
    init_logits = torch.tensor([0.7, 0.4, 0.0], requires_grad=True)  # 初始权重（logits）
    scale_logits = nn.Parameter(init_logits.clone())
    optimizer_weight = optim.Adam([scale_logits], lr=1e-2)

    # 堆叠多尺度数据 (3, 样本数, 特征数)
    multi_scale_data = torch.stack([x_fine, x_mid, x_coarse], dim=0)

    # 迭代优化权重
    for _ in range(50):
        optimizer_weight.zero_grad()
        weights = torch.softmax(scale_logits, dim=0).view(3, 1, 1)  # 转换为概率并调整形状
        fused = torch.sum(weights * multi_scale_data, dim=0)  # 加权融合
        loss_fuse = nn.MSELoss()(fused, X_single)  # 融合结果与原数据的差异
        loss_fuse.backward()
        optimizer_weight.step()

    # 最终权重和融合结果
    weights = torch.softmax(scale_logits, dim=0).detach().numpy()
    fused_result = fused.detach().numpy()

    return fused_result, weights, x_fine.numpy(), x_mid.numpy(), x_coarse.numpy()


# ===================== 三维矩阵批量处理 =====================
def process_3d_matrix(X_3d, hidden_dim=64, latent_dim=10, epochs=1000):
    """
    处理多个二维矩阵（形状: num_matrices × num_samples × num_features）
    对每个二维矩阵分别进行：
    - AAE 编码与解码
    - 多尺度聚类
    - 权重融合
    """
    X_tensor = torch.tensor(X_3d, dtype=torch.float32)
    sample = X_3d.shape[1]

    # 存储所有矩阵的结果
    all_fused, all_weights, all_fine, all_mid, all_coarse = [], [], [], [], []

    # 逐个矩阵处理
    for i in range(X_tensor.shape[0]):
        fused, weights, fine, mid, coarse = process_single_matrix(
            X_tensor[i], sample, hidden_dim, latent_dim, epochs
        )
        all_fused.append(fused)
        all_weights.append(weights)
        all_fine.append(fine)
        all_mid.append(mid)
        all_coarse.append(coarse)

    # 转换为 numpy 数组
    return (
        np.array(all_fused),  # 融合后的矩阵
        np.array(all_weights),  # 每个矩阵的融合权重
        np.array(all_fine),  # 细尺度结果
        np.array(all_mid),  # 中尺度结果
        np.array(all_coarse),  # 粗尺度结果
    )


def save_3d_matrix_to_excel(M, attributes, filename):
    """
    将三维矩阵存入Excel文件，每个决策者只显示一次标记，不添加空行

    参数:
        M: 三维numpy数组，形状为(n_decision_makers, n_samples, n_attributes)
        attributes: 属性名称列表，长度应为n_attributes
        filename: 输出的Excel文件名
    """
    # 检查输入
    assert len(M.shape) == 3, "输入必须是三维矩阵"
    assert len(attributes) == M.shape[2], "属性数量与矩阵维度不匹配"

    # 准备最终数据列表
    data_rows = []

    # 添加属性名称作为第一行
    header = ["Decision Maker"] + attributes
    data_rows.append(header)

    for dm_idx in range(M.shape[0]):
        # 添加决策者标记行(只显示一次)
        dm_label = f"decision-maker [{dm_idx+1}]"
        first_data_row = [dm_label] + M[dm_idx, 0].tolist()
        data_rows.append(first_data_row)

        # 添加该决策者的剩余数据行(不显示标记)
        for sample_idx in range(1, M.shape[1]):
            data_row = [""] + M[dm_idx, sample_idx].tolist()
            data_rows.append(data_row)

    # 转换为DataFrame
    df = pd.DataFrame(data_rows[1:], columns=data_rows[0])

    # 保存到Excel
    df.to_excel(filename, index=False, engine='openpyxl')


# ===================== 使用示例 =====================
M1 = DMS.M1
M2 = DMS.M2
M3 = DMS.M3
M4 = DMS.M4

'''
all_fused1, all_weights1, all_fine1, all_mid1, all_coarse1 = process_3d_matrix(
    M1, hidden_dim=64, latent_dim=10, epochs=200)
attributes1 = ["Buying", "Maint", "Doors",	"Persons",	"Lug_boot",	"Safety"]
save_3d_matrix_to_excel(all_fine1, attributes1, "scar-fine-scale.xlsx")
save_3d_matrix_to_excel(all_mid1, attributes1, "scar-mid-scale.xlsx")
save_3d_matrix_to_excel(all_coarse1, attributes1, "scar-coarse-scale.xlsx")
save_3d_matrix_to_excel(all_fused1, attributes1, "sdata1.xlsx")

all_fused2, all_weights2, all_fine2, all_mid2, all_coarse2 = process_3d_matrix(
    M2, hidden_dim=64, latent_dim=10, epochs=200)
attributes2 = ["Fixed Acidity", "Volatile Acidity", "Chlorides", "Density",	"Alcohol"]
save_3d_matrix_to_excel(all_fine2, attributes2, "swine-fine-scale.xlsx")
save_3d_matrix_to_excel(all_mid2, attributes2, "swine-mid-scale.xlsx")
save_3d_matrix_to_excel(all_coarse2, attributes2, "swine-coarse-scale.xlsx")
save_3d_matrix_to_excel(all_fused2, attributes2, "sdata2.xlsx")


all_fused3, all_weights3, all_fine3, all_mid3, all_coarse3 = process_3d_matrix(
    M3, hidden_dim=64, latent_dim=10, epochs=200)
attributes3 = ["Age", "SystolicBP", "DiastolicBP",	"BS",	"BodyTemp",	"HeartRate"]
save_3d_matrix_to_excel(all_fine3, attributes3, "shealth-fine-scale.xlsx")
save_3d_matrix_to_excel(all_mid3, attributes3, "shealth-mid-scale.xlsx")
save_3d_matrix_to_excel(all_coarse3, attributes3, "shealth-coarse-scale.xlsx")
save_3d_matrix_to_excel(all_fused3, attributes3, "sdata3.xlsx")

all_fused4, all_weights4, all_fine4, all_mid4, all_coarse4 = process_3d_matrix(
    M4, hidden_dim=64, latent_dim=10, epochs=200)
attributes4 = ["Studytime", "Famrel", "Absences", "G1",	"G2", "G3"]
save_3d_matrix_to_excel(all_fine4, attributes4, "student-fine-scale.xlsx")
save_3d_matrix_to_excel(all_mid4, attributes4, "student-mid-scale.xlsx")
save_3d_matrix_to_excel(all_coarse4, attributes4, "student-coarse-scale.xlsx")
save_3d_matrix_to_excel(all_fused4, attributes4, "sdata4.xlsx")
'''
all_fused3, all_weights3, all_fine3, all_mid3, all_coarse3 = process_3d_matrix(
    M3, hidden_dim=64, latent_dim=10, epochs=200)
attributes3 = ["Age", "SystolicBP", "DiastolicBP",	"BS",	"BodyTemp",	"HeartRate"]
save_3d_matrix_to_excel(all_fine3, attributes3, "shealth-fine-scale.xlsx")
save_3d_matrix_to_excel(all_mid3, attributes3, "shealth-mid-scale.xlsx")
save_3d_matrix_to_excel(all_coarse3, attributes3, "shealth-coarse-scale.xlsx")
save_3d_matrix_to_excel(all_fused3, attributes3, "sdata3.xlsx")