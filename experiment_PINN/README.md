# PINN for Quantum Tomography Experiments

本项目包含使用物理信息神经网络（Physics-Informed Neural Networks, PINN）进行量子层析成像（Quantum Tomography）的实验代码和结果。

## 项目概述

本项目实现了基于 PINN 的量子态重构方法，并进行了系统的实验验证，包括：
- 多量子比特系统的可扩展性分析
- 消融实验（Ablation Study）
- 高保真度链路应用
- 真实设备验证

## 项目结构

```
experiment_PINN/
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python 依赖包
├── .gitignore                         # Git 忽略文件配置
│
├── ablation_study.py                  # 消融实验主脚本
├── ablation_study_notebook.ipynb      # 消融实验 Jupyter notebook
├── analyze_ablation_results.md       # 消融实验结果分析报告
│
├── generate_scalability_figure.py     # 可扩展性分析图表生成脚本
├── generate_scalability_figure.ipynb  # 可扩展性分析 notebook
├── generate_comprehensive_scalability_figure.ipynb  # 综合可扩展性分析
│
├── multi_qubit_experiment_final.ipynb              # 多量子比特实验（最终版）
├── multi_qubit_scalability_experiment.ipynb        # 多量子比特可扩展性实验
├── multi_qubit_scalability_experiment_lsandmle.ipynb  # 包含 LS 和 MLE 对比的实验
│
├── 3*3_scalability_experiment.ipynb                # 3×3 可扩展性实验
├── 3*3_scalability_experiment_lsandmle.ipynb       # 3×3 可扩展性实验（含对比）
│
├── high_fidelity_link_application.ipynb            # 高保真度链路应用
│
├── real_device_validation.ipynb                    # 真实设备验证实验
│
├── ablation_results/                 # 消融实验结果
│   ├── ablation_results_table.csv
│   ├── ablation_fidelity_comparison.png
│   ├── ablation_fidelity_paper.png
│   ├── ablation_fidelity_paper.pdf
│   ├── ablation_component_contribution_paper.png
│   ├── ablation_component_contribution_paper.pdf
│   ├── ablation_heatmap.png
│   └── ablation_training_curves.png
│
├── results/                          # 实验结果数据
│   └── scalability_analysis_data.csv
│
└── images/                           # 实验图片
    ├── 2qubit_training_results.png
    ├── 3qubit_training_results.png
    ├── 4qubit_training_results.png
    ├── 5qubit_training_results.png
    └── scalability_analysis.png
```

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 主要依赖包

- `numpy>=1.21.0` - 数值计算
- `matplotlib>=3.5.0` - 数据可视化
- `seaborn>=0.11.0` - 统计图表
- `torch>=1.10.0` - 深度学习框架
- `pandas>=1.3.0` - 数据处理
- `scipy>=1.7.0` - 科学计算
- `tqdm>=4.62.0` - 进度条
- `jupyter>=1.0.0` - Jupyter notebook 环境

## 使用方法

### 1. 消融实验

运行消融实验以评估 PINN 架构中各组件的贡献：

```bash
python ablation_study.py
```

或使用 Jupyter notebook：

```bash
jupyter notebook ablation_study_notebook.ipynb
```

消融实验评估的组件包括：
- Residual Connections 的影响
- Attention Mechanism 的影响
- Dynamic Weighting vs Fixed Weighting
- 不同约束权重设置的影响

### 2. 可扩展性分析

生成可扩展性分析图表：

```bash
python generate_scalability_figure.py
```

或使用相应的 Jupyter notebook 进行交互式分析。

### 3. 多量子比特实验

打开相应的 notebook 文件运行多量子比特系统实验：

```bash
jupyter notebook multi_qubit_experiment_final.ipynb
```

## 实验结果

### 消融实验结果摘要

根据 `analyze_ablation_results.md` 中的分析：

| 配置 | 最佳保真度 | 相对Baseline改进 |
|------|-----------|----------------|
| **Full Model** | **0.8158** | **+2.30%** |
| w/o Residual | 0.8115 | +1.77% |
| w/o Attention | 0.7906 | -0.86% |
| Fixed Weight (0.15) | 0.8140 | +2.09% |
| **Baseline** | **0.7974** | **0.00%** |

主要发现：
- **Attention Mechanism** 是贡献最大的组件（+3.19%改进）
- **Residual Connections** 提供了额外的改进（+0.53%）
- **Dynamic Weighting** 相比固定权重有轻微优势（+0.22%）
- 权重 0.15 是最优配置

详细分析请参考 `analyze_ablation_results.md`。

## 文件说明

### 核心脚本

- **ablation_study.py**: 消融实验的主脚本，包含完整的 PINN 架构实现和消融实验逻辑
- **generate_scalability_figure.py**: 生成可扩展性分析图表的脚本

### Jupyter Notebooks

- **multi_qubit_experiment_final.ipynb**: 多量子比特系统的最终实验版本
- **multi_qubit_scalability_experiment*.ipynb**: 不同配置下的可扩展性实验
- **ablation_study_notebook.ipynb**: 消融实验的交互式版本
- **real_device_validation.ipynb**: 真实量子设备的验证实验

## 注意事项

1. **数据文件**: 某些实验可能需要额外的数据文件（如 `.npz` 格式的量子态数据），请确保数据文件路径正确。

2. **计算资源**: 多量子比特实验（特别是 5 量子比特及以上）可能需要较长的计算时间和较大的内存。

3. **随机种子**: 代码中设置了随机种子（`np.random.seed(42)` 和 `torch.manual_seed(42)`）以确保结果的可重复性。

4. **GPU 支持**: 如果系统有 GPU，PyTorch 会自动使用 GPU 加速训练。

5. **重复文件**: 项目中可能存在一些包含 "copy" 或 "copy 2" 的文件名，这些通常是开发过程中的备份文件。建议使用主版本文件（不包含 "copy" 的文件名）。

## 引用

如果使用本代码，请引用相关论文。

## 许可证

本项目仅供研究使用。

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

