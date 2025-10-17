import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("QtAgg")

# 设置字体和字号
config = {
    "font.family":'serif',
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
}
matplotlib.rcParams.update(config)

# 横轴数据（\varsigma）
sigma = [0.93, 0.94, 0.95, 0.96]

# 数据集1
success1 = ['yes', 'yes', 'no', 'no']
RT1 = [0.866, 1.597, 36.762, 46.973]

# 数据集2
success2 = ['yes', 'yes', 'yes', 'no']
RT2 = [1.106, 1.314, 1.167, 37.688]

# 数据集3
success3 = ['yes', 'yes', 'yes', 'yes']
RT3 = [0.382, 0.234, 0.371, 1.932]

# 数据集4
success4 = ['yes', 'yes', 'yes', 'no']
RT4 = [7.449, 3.249, 2.126, 34.561]

# 定义绘图函数
def plot_group(ax, sigma, RT, success, title):
    for s, rt, suc in zip(sigma, RT, success):
        if suc == 'yes':
            ax.scatter(s, rt, color='LightCoral', marker='o', s=120,
                       label='Success' if 'Success' not in ax.get_legend_handles_labels()[1] else "")
        else:
            ax.scatter(s, rt, color='DarkGreen', marker='x', s=120,
                       label='Failure' if 'Failure' not in ax.get_legend_handles_labels()[1] else "")

    ax.plot(sigma, RT, linestyle='-', color='Tan', alpha=0.8, linewidth=2)

    ax.set_xlabel(r'$\varsigma$', fontsize=26)  # 使用公式显示sigma
    ax.set_ylabel('RT', fontsize=26)
    ax.set_title(title, fontsize=26)

    ax.tick_params(axis='both', labelsize=26)
    ax.grid(True, color='gray', alpha=0.3, linestyle='--')
    ax.legend(fontsize=26)

# 创建 2x2 子图
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

plot_group(axs[0, 0], sigma, RT1, success1, 'Data1')
plot_group(axs[0, 1], sigma, RT2, success2, 'Data2')
plot_group(axs[1, 0], sigma, RT3, success3, 'Data3')
plot_group(axs[1, 1], sigma, RT4, success4, 'Data4')

# 调整子图间距
plt.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.07, hspace=0.3, wspace=0.25)

# 保存高分辨率 PDF
plt.savefig("min2.pdf", format='pdf', dpi=300, bbox_inches='tight')

plt.show()
