import numpy as np
import b1_Cluster_GB as CGB

def calculate_consensus(C, G, m, Lm, alpha):
    """
    计算子组共识度 Con^h = (1 - 平均差异) * e^(-α|Lm - m^h|)
    返回: 共识度值 (float)
    """

    # 计算平均绝对差异
    mean_diff = np.mean(np.abs(G - C))

    # 计算密度差异项
    density_diff = abs(Lm - m)

    # 计算共识度
    consensus = (1 - mean_diff) * np.exp(-alpha * density_diff)

    return consensus

# 示例使用
# data1: 共识阈值设为0.94
# data2: 共识阈值设为0.95
# data3: 共识阈值设为0.95
# data4: 共识阈值设为0.95

consensus = 0.95
alpha = 0.3

C = CGB.A
G = CGB.G
m = CGB.m
Lm = CGB.Lm
w = CGB.W
# 计算小组共识度
con = np.zeros(5)
Lcon = 0
# AD
AD = 0
for h in range(C.shape[0]):
    con[h] = calculate_consensus(C[h], G, m[h], Lm, alpha)
    print(f"小组{h+1}的共识度为：", con[h])
    if con[h] < consensus:
        AD = AD+1
    Lcon = Lcon + con[h] * w[h]
print(f"大组的共识度为：", Lcon)

# CID
CID = (consensus-Lcon) / Lcon





