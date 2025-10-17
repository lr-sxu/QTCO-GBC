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

# 数据
B = [0.01, 0.02, 0.03, 0.04, 0.05]

success1 = ['yes', 'no', 'no', 'no', 'no']
RT1 = [2.159, 11.106, 15.927, 22.194, 20.9545]

success2 = ['yes', 'no', 'no', 'no', 'no']
RT2 = [4.23, 7.477, 8.533, 15.912, 13.848]

success3 = ['yes', 'no', 'no', 'no', 'no']
RT3 = [6.153, 18.163, 28.027, 35.579, 31.139]

success4 = ['yes', 'yes', 'no', 'no', 'no']
RT4 = [1.628, 10.701, 27.917, 30.664, 35.8]

# 定义颜色和标记
def plot_group(ax, B, RT, success, title):
    for b, rt, suc in zip(B, RT, success):
        if suc == 'yes':
            ax.scatter(b, rt, color='LightCoral', marker='o', s=120,
                       label='Success' if 'Success' not in ax.get_legend_handles_labels()[1] else "")
        else:
            ax.scatter(b, rt, color='DarkGreen', marker='x', s=120,
                       label='Failure' if 'Failure' not in ax.get_legend_handles_labels()[1] else "")

    ax.plot(B, RT, linestyle='-', color='Tan', alpha=0.8, linewidth=2)

    ax.set_xlabel(r'$B$', fontsize=26)
    ax.set_ylabel('RT', fontsize=26)
    ax.set_title(title, fontsize=26)

    ax.tick_params(axis='both', labelsize=26)
    ax.grid(True)
    ax.legend(fontsize=26)

# 创建图形
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

plot_group(axs[0, 0], B, RT1, success1, 'Data1')
plot_group(axs[0, 1], B, RT2, success2, 'Data2')
plot_group(axs[1, 0], B, RT3, success3, 'Data3')
plot_group(axs[1, 1], B, RT4, success4, 'Data4')

# 调整子图间距，避免上下行重叠
plt.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.07, hspace=0.25, wspace=0.25)

# 保存为高分辨率 PDF
plt.savefig("min1.pdf", format='pdf', dpi=300, bbox_inches='tight')

plt.show()
