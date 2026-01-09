import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# TABLE2的数据
qubits = [2, 3, 4, 5]
traditional_nn = [0.9762, 0.9010, 0.6810, 0.5688]
least_squares = [0.9347, 0.7947, 0.7130, 0.3234]
mle = [0.4525, 0.3197, 0.2426, 0.1947]
pinn = [0.9835, 0.9124, 0.8872, 0.5814]

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制线条
line1 = ax.plot(qubits, traditional_nn, marker='o', markersize=8, linewidth=2.5, 
                label='Traditional NN', color='#2E86AB', linestyle='-')
line2 = ax.plot(qubits, least_squares, marker='s', markersize=8, linewidth=2.5, 
                label='Least Squares', color='#A23B72', linestyle='--')
line3 = ax.plot(qubits, mle, marker='^', markersize=8, linewidth=2.5, 
                label='MLE', color='#F18F01', linestyle='-.')
line4 = ax.plot(qubits, pinn, marker='D', markersize=10, linewidth=3, 
                label='PINN', color='#C73E1D', linestyle='-')

# 设置坐标轴
ax.set_xlabel('Number of Qubits', fontsize=14, fontweight='bold')
ax.set_ylabel('Fidelity', fontsize=14, fontweight='bold')
ax.set_xticks(qubits)
ax.set_xticklabels([f'{q}' for q in qubits], fontsize=12)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_yticklabels([f'{y:.1f}' for y in np.arange(0, 1.1, 0.1)], fontsize=12)
ax.set_ylim([0, 1.05])

# 添加网格
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# 添加图例
legend = ax.legend(loc='upper right', fontsize=12, frameon=True, 
                   fancybox=True, shadow=True, framealpha=0.9)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('gray')

# 添加标题
ax.set_title('Scalability Analysis: Fidelity Comparison Across 2-5 Qubits', 
             fontsize=16, fontweight='bold', pad=20)

# 添加注释突出显示4量子比特的PINN优势
ax.annotate('PINN achieves\n30.3% improvement\nat 4 qubits', 
            xy=(4, 0.8872), xytext=(4.3, 0.75),
            arrowprops=dict(arrowstyle='->', color='#C73E1D', lw=2),
            fontsize=11, fontweight='bold', color='#C73E1D',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

# 调整布局
plt.tight_layout()

# 保存图片
output_path = '/Users/fcc/Documents/个人/【个人】-论文写作/5-【qml】/experiment/images/scalability_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"图片已保存到: {output_path}")

# 显示图片
plt.show()



