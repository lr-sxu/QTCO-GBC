import numpy as np
import pandas as pd

# 获取数据
def excel_to_3d_matrix(filename):
    """
    从Excel文件读取数据并组成三维矩阵

    参数:
        filename: Excel文件名

    返回:
        三维numpy数组，形状为(n_decision_makers, n_samples, n_attributes)
        属性名称列表
    """
    # 读取Excel文件
    df = pd.read_excel(filename, engine='openpyxl')

    # 获取属性名称(排除第一列"Decision Maker")
    attributes = df.columns[1:].tolist()

    # 初始化变量
    decision_makers = []
    current_dm_samples = []
    current_dm_id = None

    # 遍历每一行数据
    for _, row in df.iterrows():
        # 检查是否是新的决策者标记
        if isinstance(row[0], str) and row[0].startswith("decision-maker ["):
            # 保存上一个决策者的数据(如果有)
            if current_dm_id is not None and len(current_dm_samples) > 0:
                decision_makers.append(current_dm_samples)

            # 开始新的决策者
            current_dm_id = int(row[0].split("[")[1].split("]")[0])
            current_dm_samples = [row[1:].values.astype(float)]
        else:
            # 添加样本到当前决策者
            if len(row[1:]) > 0 and not pd.isna(row[1]):
                current_dm_samples.append(row[1:].values.astype(float))

    # 添加最后一个决策者的数据
    if len(current_dm_samples) > 0:
        decision_makers.append(current_dm_samples)

    # 转换为三维numpy数组
    n_dm = len(decision_makers)
    n_samples = max(len(dm) for dm in decision_makers) if n_dm > 0 else 0
    n_attrs = len(attributes)

    # 创建三维数组并用NaN填充缺失值
    M = np.full((n_dm, n_samples, n_attrs), np.nan)
    for i, dm_samples in enumerate(decision_makers):
        M[i, :len(dm_samples)] = dm_samples

    return M

# 获取单尺度数据
M1 = excel_to_3d_matrix("sdata1.xlsx")
M2 = excel_to_3d_matrix("sdata2.xlsx")
M3 = excel_to_3d_matrix("sdata3.xlsx")
M4 = excel_to_3d_matrix("sdata4.xlsx")
