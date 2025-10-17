import numpy as np
import pandas as pd

# 读取文件
def read_excel_from_second_row_pandas(file_path):
    # 读取Excel文件，跳过第一行
    df = pd.read_excel(file_path, header=None, skiprows=1)
    # 转换为矩阵（二维列表）
    matrix = df.values.tolist()
    return matrix

# 效益性归一化
def normalize_matrix1(matrix):
    # 转换为 NumPy 数组
    data = np.array(matrix, dtype=float)
    # 计算每列的最小值和最大值
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    range_vals = max_vals - min_vals
    # 归一化：(X - min) / (max - min)
    normalized_data = (data - min_vals) / range_vals
    return normalized_data.tolist()  # 转换回列

# 数据二的归一化
def normalize_matrix2(data_matrix):
    # 初始化归一化后的矩阵
    normalized_matrix = np.zeros_like(data_matrix, dtype=float)
    # 列索引（假设顺序与表格一致）
    col_fixed_acidity = 0  # 固定酸度
    col_volatile_acidity = 1  # 挥发性酸度
    col_chlorides = 2  # 氯化物
    col_density = 3  # 密度
    col_alcohol = 4  # 酒精

    # (1) 固定酸度：高斯型归一化（越接近6，值越大）
    ideal = 6.0
    scale = 1.0
    normalized_matrix[:, col_fixed_acidity] = np.exp(
        -((data_matrix[:, col_fixed_acidity] - ideal) ** 2 / (2 * scale ** 2)
          ))

    # (2) 挥发性酸度、氯化物、密度：越小越好 → 1 - (x - min)/(max - min)
    for col in [col_volatile_acidity, col_chlorides, col_density]:
        min_val = np.min(data_matrix[:, col])
        max_val = np.max(data_matrix[:, col])
        range_val = max_val - min_val
        range_val = range_val if range_val != 0 else 1  # 避免除零
        normalized_matrix[:, col] = 1 - (data_matrix[:, col] - min_val) / range_val

    # (3) 酒精：越大越好 → (x - min)/(max - min)
    min_val = np.min(data_matrix[:, col_alcohol])
    max_val = np.max(data_matrix[:, col_alcohol])
    range_val = max_val - min_val
    range_val = range_val if range_val != 0 else 1  # 避免除零
    normalized_matrix[:, col_alcohol] = (data_matrix[:, col_alcohol] - min_val) / range_val

    return normalized_matrix

# 数据三的归一化
def normalize_matrix3(data):
    """
        输入:
            data: DataFrame 或 numpy array
                  列顺序为 [Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]
        输出:
            numpy array 归一化得分矩阵 (每列 0~1，值越高风险越低)
        """

    # 定义高斯评分函数
    def gaussian_score(x, mu, sigma):
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # 针对每个指标的参数 (mu, sigma)
    params = {
        "Age": (27.5, 7.5),
        "SystolicBP": (115, 10),
        "DiastolicBP": (75, 7),
        "BS": (5.1, 0.5),
        "BodyTemp": (98.24, 0.54),
        "HeartRate": (80, 7)
    }

    # 转成 numpy
    if isinstance(data, pd.DataFrame):
        matrix = data.values
    else:
        matrix = np.array(data)

    # 按列计算得分
    scores = np.zeros_like(matrix, dtype=float)
    for i, col in enumerate(params.keys()):
        mu, sigma = params[col]
        scores[:, i] = gaussian_score(matrix[:, i], mu, sigma)

    return scores

# 数据四的归一化
def normalize_matrix4(matrix):
    """
    对二维矩阵进行归一化处理：
    - 第三列（索引为2）使用成本归一化
    - 其他列使用效益归一化

    参数:
    matrix: 二维数组或列表的列表

    返回:
    normalized_matrix: 归一化后的二维numpy数组
    """
    # 转换为numpy数组以便处理
    arr = np.array(matrix, dtype=float)

    # 创建结果数组
    normalized_arr = np.zeros_like(arr)

    # 对每一列进行归一化
    for col_idx in range(arr.shape[1]):
        column = arr[:, col_idx]
        if col_idx == 2:  # 第三列（索引为2）使用成本归一化
            min_val = np.min(column)
            max_val = np.max(column)
            # 成本归一化公式：归一化值 = (max - x) / (max - min)
            normalized_arr[:, col_idx] = (max_val - column) / (max_val - min_val)
        else:  # 其他列使用效益归一化
            min_val = np.min(column)
            max_val = np.max(column)
            # 效益归一化公式：归一化值 = (x - min) / (max - min)
            normalized_arr[:, col_idx] = (column - min_val) / (max_val - min_val)

    return normalized_arr

# 模拟生成决策者
def Simulation(original_data):
    # 转换为 NumPy 数组
    original_data = np.array(original_data, dtype=float)
    n, m = original_data.shape
    H = 30  # 30个决策者
    sigma = 0.1  # 扰动标准差
    # 生成30个DM的评估数据
    dm_data = []
    for h in range(H):
       # 生成正态分布扰动矩阵
       perturbation = np.random.normal(0, sigma, size=(n, m))
       # 计算扰动后的数据（乘法扰动）
       perturbed_data = original_data * (1 + perturbation)
       dm_data.append(perturbed_data)
    return dm_data

# 原始数据（假设是一个 n×m 的矩阵）
# 车质量评估数据
file_path1 = 'odata1.xlsx'  # 替换为你的文件路径
data_matrix1 = read_excel_from_second_row_pandas(file_path1)
m1 = Simulation(data_matrix1)
a1 = np.array(m1, dtype=float)
M1 = np.zeros((a1.shape[0], a1.shape[1], a1.shape[2]))
for i in range(M1.shape[0]):
    M1[i] = normalize_matrix1(a1[i])
M1 = np.round(M1, 3)
#print(M1)

# 葡萄酒质量评估数据
file_path2 = 'odata2.xlsx'  # 替换为你的文件路径
data_matrix2 = read_excel_from_second_row_pandas(file_path2)
m2 = Simulation(data_matrix2)
a2 = np.array(m2, dtype=float)
M2 = np.zeros((a2.shape[0], a2.shape[1], a2.shape[2]))
for i in range(M2.shape[0]):
    M2[i] = normalize_matrix2(a2[i])
M2 = np.round(M2, 3)
#print(M2)

# 孕妇健康风险评估数据
file_path3 = 'odata3.xlsx'  # 替换为你的文件路径
data_matrix3 = read_excel_from_second_row_pandas(file_path3)
columns = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
df = pd.DataFrame(data_matrix3, columns=columns)
dfm = normalize_matrix3(df)
m3 = Simulation(dfm)
a3 = np.array(m3, dtype=float)
M3 = np.zeros((a3.shape[0], a3.shape[1], a3.shape[2]))
for i in range(M3.shape[0]):
    M3[i] = normalize_matrix1(a3[i])
# 保留三位小数
M3 = np.round(M3, 3)
#print(M3)

# 学生成绩表现数据
file_path4 = 'odata4.xlsx'  # 替换为你的文件路径
data_matrix4 = read_excel_from_second_row_pandas(file_path4)
m4 = Simulation(data_matrix4)
a4 = np.array(m4, dtype=float)
M4 = np.zeros((a4.shape[0], a4.shape[1], a4.shape[2]))
for i in range(M4.shape[0]):
    M4[i] = normalize_matrix4(a4[i])
M4 = np.round(M4, 3)
print(M4)
