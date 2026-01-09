"""
消融实验：系统评估PINN架构中各组件的贡献
包括：
1. Residual Connections的影响
2. Attention Mechanism的影响
3. Dynamic Weighting vs Fixed Weighting
4. 不同约束权重设置的影响
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from scipy.linalg import sqrtm
import json
import os
from datetime import datetime

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 设置绘图风格
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

print("消融实验环境已就绪！")


class MultiQubitQuantumTools:
    """多量子比特系统工具类"""
    
    @staticmethod
    def random_pure_state(n_qubits):
        dim = 2 ** n_qubits
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        return state / np.linalg.norm(state)
    
    @staticmethod
    def random_mixed_state(n_qubits, purity=0.8):
        dim = 2 ** n_qubits
        psi = MultiQubitQuantumTools.random_pure_state(n_qubits)
        rho = np.outer(psi, psi.conj())
        identity = np.eye(dim, dtype=complex)
        rho_mixed = purity * rho + (1 - purity) * identity / dim
        return rho_mixed / np.trace(rho_mixed)
    
    @staticmethod
    def GHZ_state(n_qubits):
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1 / np.sqrt(2)
        state[-1] = 1 / np.sqrt(2)
        return np.outer(state, state.conj())
    
    @staticmethod
    def W_state(n_qubits):
        if n_qubits == 2:
            state = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)
        elif n_qubits == 3:
            state = np.array([0, 1/np.sqrt(3), 1/np.sqrt(3), 0, 1/np.sqrt(3), 0, 0, 0], dtype=complex)
        else:
            state = np.zeros(2**n_qubits, dtype=complex)
            for i in range(n_qubits):
                idx = 2**i
                state[idx] = 1/np.sqrt(n_qubits)
        return np.outer(state, state.conj())
    
    @staticmethod
    def pauli_operators():
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return [I, X, Y, Z]
    
    @staticmethod
    def multiqubit_pauli_matrices(n_qubits):
        paulis = MultiQubitQuantumTools.pauli_operators()
        if n_qubits == 1:
            return paulis
        multi_paulis = []
        for ops in np.ndindex((4,) * n_qubits):
            op = paulis[ops[0]]
            for i in range(1, n_qubits):
                op = np.kron(op, paulis[ops[i]])
            multi_paulis.append(op)
        return multi_paulis
    
    @staticmethod
    def density_to_cholesky(rho):
        """将密度矩阵转换为Cholesky参数"""
        dim = rho.shape[0]
        L = np.linalg.cholesky(rho + 1e-10 * np.eye(dim))
        params = []
        for i in range(dim):
            for j in range(i + 1):
                if i == j:
                    params.append(np.real(L[i, j]))
                else:
                    params.append(np.real(L[i, j]))
                    params.append(np.imag(L[i, j]))
        return np.array(params)
    
    @staticmethod
    def cholesky_to_density_matrix(params, n_qubits):
        """将Cholesky参数转换回密度矩阵"""
        dim = 2 ** n_qubits
        L = np.zeros((dim, dim), dtype=complex)
        idx = 0
        for i in range(dim):
            for j in range(i + 1):
                if idx >= len(params):
                    break
                if i == j:
                    L[i, j] = np.abs(params[idx]) + 1e-9
                    idx += 1
                else:
                    if idx + 1 < len(params):
                        L[i, j] = params[idx] + 1j * params[idx + 1]
                        idx += 2
                    else:
                        break
        rho = L @ L.conj().T
        rho = rho / np.trace(rho)
        return rho
    
    @staticmethod
    def fidelity(rho1, rho2):
        """计算两个密度矩阵之间的保真度"""
        sqrt_rho1 = sqrtm(rho1)
        product = sqrt_rho1 @ rho2 @ sqrt_rho1
        return np.real(np.trace(sqrtm(product)) ** 2)
    
    @staticmethod
    def constraint_violation(rho):
        """计算约束违反度"""
        dim = rho.shape[0]
        # Hermiticity
        herm = np.linalg.norm(rho - rho.conj().T) ** 2
        # Trace
        trace = np.abs(np.trace(rho) - 1.0) ** 2
        # Positivity
        eigenvals = np.linalg.eigvals(rho)
        pos = np.sum(np.maximum(0, -np.real(eigenvals)) ** 2)
        return (herm + trace + pos) / (dim ** 0.5)


class AblationPINN(nn.Module):
    """支持消融实验的PINN模型"""
    
    def __init__(self, n_qubits, input_dim, hidden_dims=[256, 128], 
                 use_residual=True, use_attention=True, 
                 use_dynamic_weighting=True, fixed_weight=0.15):
        super().__init__()
        self.n_qubits = n_qubits
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.use_dynamic_weighting = use_dynamic_weighting
        self.fixed_weight = fixed_weight
        
        dim = 2 ** n_qubits
        self.dim = dim
        # Cholesky参数数量：对角元素dim个（实数）+ 非对角元素dim*(dim-1)/2个（复数，每个2个参数）
        # 总共：dim + dim*(dim-1) = dim*dim
        self.output_dim = dim * dim
        
        # 特征提取层
        self.feature_layers = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1 if hidden_dim >= 256 else 0.05)
            )
            self.feature_layers.append(layer)
            
            # 残差连接
            if use_residual:
                if prev_dim != hidden_dim:
                    self.residual_projs.append(nn.Linear(prev_dim, hidden_dim))
                else:
                    self.residual_projs.append(nn.Identity())
            else:
                self.residual_projs.append(nn.Identity())
            
            prev_dim = hidden_dim
        
        # 注意力机制
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                nn.ReLU(),
                nn.Linear(prev_dim // 2, prev_dim),
                nn.Sigmoid()
            )
        else:
            self.attention = None
        
        # 输出层
        self.density_head = nn.Sequential(
            nn.Linear(prev_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.output_dim)
        )
        
        # 噪声严重度预测（用于动态权重）
        if use_dynamic_weighting:
            self.severity_head = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        # 物理约束权重
        base_weight = fixed_weight
        normalized_weight = base_weight / (dim ** 0.5)
        self.register_buffer('physics_weight_base', torch.tensor(normalized_weight))
        self.qt = MultiQubitQuantumTools()
    
    def get_physics_weight(self, severity=None):
        """获取物理约束权重"""
        base = self.physics_weight_base
        if self.use_dynamic_weighting and severity is not None and torch.is_tensor(severity) and severity.numel() > 0:
            severity_avg = severity.mean().item()
            adaptive_factor = max(0.5, 1.0 - severity_avg * 1.5)
            return base * adaptive_factor
        return base.clone() if torch.is_tensor(base) else base
    
    def forward(self, x, return_severity=False):
        features = x
        
        # 特征提取（带或不带残差连接）
        for i, (layer, residual_proj) in enumerate(zip(self.feature_layers, self.residual_projs)):
            if self.use_residual:
                residual = residual_proj(features)
                features = layer(features) + residual
            else:
                features = layer(features)
        
        # 注意力机制
        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights
        
        # 输出
        density_params = self.density_head(features)
        
        if return_severity and self.use_dynamic_weighting:
            severity_pred = self.severity_head(features).squeeze(-1)
            return density_params, severity_pred
        
        return density_params
    
    def cholesky_to_density_torch(self, alpha):
        """将Cholesky参数转换为密度矩阵（PyTorch版本）"""
        batch_size = alpha.shape[0]
        dim = self.dim
        n_params_used = dim * dim
        L_real = torch.zeros(batch_size, dim, dim, device=alpha.device, dtype=alpha.dtype)
        L_imag = torch.zeros(batch_size, dim, dim, device=alpha.device, dtype=alpha.dtype)
        idx = 0
        for i in range(dim):
            for j in range(i + 1):
                if idx >= n_params_used:
                    break
                if i == j:
                    L_real[:, i, j] = torch.abs(alpha[:, idx]) + 1e-9
                    idx += 1
                else:
                    if idx + 1 < alpha.shape[1]:
                        L_real[:, i, j] = alpha[:, idx]
                        L_imag[:, i, j] = alpha[:, idx + 1]
                        idx += 2
                    else:
                        break
        L_real_T = L_real.transpose(-2, -1)
        L_imag_T = L_imag.transpose(-2, -1)
        rho_real = torch.bmm(L_real, L_real_T) + torch.bmm(L_imag, L_imag_T)
        rho_imag = torch.bmm(L_real, L_imag_T) - torch.bmm(L_imag, L_real_T)
        trace = torch.sum(rho_real[:, torch.arange(dim), torch.arange(dim)], dim=1, keepdim=True)
        trace = trace.unsqueeze(-1)
        rho_real = rho_real / (trace + 1e-9)
        rho_imag = rho_imag / (trace + 1e-9)
        return rho_real, rho_imag
    
    def compute_physics_loss_torch(self, rho_real, rho_imag):
        """计算物理约束损失（PyTorch版本）"""
        dim = self.dim
        rho_real_T = rho_real.transpose(-2, -1)
        rho_imag_T = rho_imag.transpose(-2, -1)
        hermiticity_real = rho_real - rho_real_T
        hermiticity_imag = rho_imag + rho_imag_T
        hermiticity_loss = torch.mean(hermiticity_real ** 2 + hermiticity_imag ** 2)
        trace = torch.sum(rho_real[:, torch.arange(dim), torch.arange(dim)], dim=1)
        trace_violation = torch.mean((trace - 1.0) ** 2)
        diag_elements = rho_real[:, torch.arange(dim), torch.arange(dim)]
        min_diag = torch.min(diag_elements, dim=1)[0]
        positivity_loss = torch.mean(torch.clamp(-min_diag, min=0.0) ** 2)
        total_loss = (hermiticity_loss + trace_violation + positivity_loss) / (dim ** 0.5)
        return total_loss


class QuantumDataset(Dataset):
    """量子态数据集"""
    
    def __init__(self, n_qubits, n_samples, noise_level=0.1):
        self.n_qubits = n_qubits
        self.qt = MultiQubitQuantumTools()
        self.data = []
        
        # 生成多样化的量子态
        for _ in range(n_samples):
            state_type = np.random.choice(['GHZ', 'W', 'random_pure', 'random_mixed'])
            if state_type == 'GHZ':
                rho = self.qt.GHZ_state(n_qubits)
            elif state_type == 'W':
                rho = self.qt.W_state(n_qubits)
            elif state_type == 'random_pure':
                psi = self.qt.random_pure_state(n_qubits)
                rho = np.outer(psi, psi.conj())
            else:
                rho = self.qt.random_mixed_state(n_qubits, purity=np.random.uniform(0.6, 0.95))
            
            # 添加噪声
            dim = 2 ** n_qubits
            noise = np.random.normal(0, noise_level, (dim, dim)) + 1j * np.random.normal(0, noise_level, (dim, dim))
            rho_noisy = rho + noise
            rho_noisy = (rho_noisy + rho_noisy.conj().T) / 2  # 确保Hermitian
            eigenvals = np.linalg.eigvals(rho_noisy)
            if np.any(eigenvals < 0):
                rho_noisy = rho_noisy - np.min(np.real(eigenvals)) * np.eye(dim)
            rho_noisy = rho_noisy / np.trace(rho_noisy)
            
            # 生成测量数据
            paulis = self.qt.multiqubit_pauli_matrices(n_qubits)
            measurements = []
            for P in paulis:
                expectation = np.real(np.trace(rho_noisy @ P))
                # 添加测量噪声
                measurement = expectation + np.random.normal(0, noise_level * 0.1)
                measurements.append(measurement)
            
            # 计算噪声严重度（真实值）
            noise_severity = min(1.0, noise_level * 2.0)
            
            # Cholesky参数
            target_params = self.qt.density_to_cholesky(rho)
            
            self.data.append({
                'measurements': np.array(measurements, dtype=np.float32),
                'target_params': target_params.astype(np.float32),
                'target_rho': rho,
                'noise_severity': noise_severity
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'measurements': torch.tensor(item['measurements']),
            'target_params': torch.tensor(item['target_params']),
            'target_rho': item['target_rho'],
            'noise_severity': torch.tensor(item['noise_severity'], dtype=torch.float32)
        }


class AblationTrainer:
    """消融实验训练器"""
    
    def __init__(self, model, device='cpu', warmup_epochs=5):
        self.model = model.to(device)
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.base_lr = 0.0015
        self.optimizer = optim.AdamW(model.parameters(), lr=self.base_lr, weight_decay=1e-4)
        self.mse_loss = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.fidelities = []
        self.constraint_violations = []
        self.current_epoch = 0
    
    def _get_lr(self):
        """获取当前学习率（带warmup）"""
        if self.current_epoch < self.warmup_epochs:
            return self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            return self.optimizer.param_groups[0]['lr']
    
    def _set_lr(self, lr):
        """设置学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            measurements = batch['measurements'].to(self.device)
            target_params = batch['target_params'].to(self.device)
            severity = batch.get('noise_severity')
            if severity is not None:
                severity = severity.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            if self.model.use_dynamic_weighting:
                pred, severity_pred = self.model(measurements, return_severity=True)
            else:
                pred = self.model(measurements)
            
            # 数据损失
            loss = self.mse_loss(pred, target_params)
            
            # 物理约束损失
            rho_real, rho_imag = self.model.cholesky_to_density_torch(pred)
            physics_loss = self.model.compute_physics_loss_torch(rho_real, rho_imag)
            adaptive_weight = self.model.get_physics_weight(severity)
            
            if not torch.is_tensor(adaptive_weight):
                adaptive_weight = torch.tensor(adaptive_weight, device=loss.device, dtype=loss.dtype)
            elif adaptive_weight.device != loss.device:
                adaptive_weight = adaptive_weight.to(loss.device)
            
            total_loss = loss + adaptive_weight * physics_loss
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
            self.optimizer.step()
            
            epoch_loss += total_loss.item()
            n_batches += 1
        
        return epoch_loss / n_batches
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        val_loss = 0
        fidelities = []
        constraint_violations = []
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                measurements = batch['measurements'].to(self.device)
                target_params = batch['target_params'].to(self.device)
                target_rhos = batch['target_rho']
                
                pred = self.model(measurements)
                loss = self.mse_loss(pred, target_params)
                val_loss += loss.item()
                
                # 计算保真度
                pred_np = pred.cpu().numpy()
                for i, target_rho in enumerate(target_rhos):
                    pred_rho = self.model.qt.cholesky_to_density_matrix(pred_np[i], self.model.n_qubits)
                    fid = self.model.qt.fidelity(target_rho, pred_rho)
                    fidelities.append(fid)
                    cv = self.model.qt.constraint_violation(pred_rho)
                    constraint_violations.append(cv)
                
                n_batches += 1
        
        return val_loss / n_batches, np.mean(fidelities), np.mean(constraint_violations)
    
    def train(self, train_loader, val_loader, epochs=30):
        """训练模型"""
        # 学习率调度器（cosine annealing）
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs - self.warmup_epochs, eta_min=1e-6
        )
        
        best_fidelity = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Warmup
            if epoch < self.warmup_epochs:
                lr = self._get_lr()
                self._set_lr(lr)
            else:
                scheduler.step()
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            val_loss, fidelity, cv = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.fidelities.append(fidelity)
            self.constraint_violations.append(cv)
            
            if fidelity > best_fidelity:
                best_fidelity = fidelity
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, Fidelity: {fidelity:.4f}, CV: {cv:.2e}")
        
        return best_fidelity


def run_ablation_study(n_qubits=3, n_train=3000, n_val=800, epochs=25, device='cpu'):
    """运行完整的消融实验"""
    
    print(f"\n{'='*60}")
    print(f"开始消融实验 - {n_qubits} qubits")
    print(f"{'='*60}\n")
    
    # 创建数据集
    print("生成数据集...")
    train_dataset = QuantumDataset(n_qubits, n_train, noise_level=0.1)
    val_dataset = QuantumDataset(n_qubits, n_val, noise_level=0.1)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    input_dim = len(train_dataset[0]['measurements'])
    
    # 定义消融实验配置
    configs = [
        {
            'name': 'Full Model',
            'use_residual': True,
            'use_attention': True,
            'use_dynamic_weighting': True,
            'fixed_weight': 0.15
        },
        {
            'name': 'w/o Residual',
            'use_residual': False,
            'use_attention': True,
            'use_dynamic_weighting': True,
            'fixed_weight': 0.15
        },
        {
            'name': 'w/o Attention',
            'use_residual': True,
            'use_attention': False,
            'use_dynamic_weighting': True,
            'fixed_weight': 0.15
        },
        {
            'name': 'Fixed Weight (0.15)',
            'use_residual': True,
            'use_attention': True,
            'use_dynamic_weighting': False,
            'fixed_weight': 0.15
        },
        {
            'name': 'Fixed Weight (0.05)',
            'use_residual': True,
            'use_attention': True,
            'use_dynamic_weighting': False,
            'fixed_weight': 0.05
        },
        {
            'name': 'Fixed Weight (0.30)',
            'use_residual': True,
            'use_attention': True,
            'use_dynamic_weighting': False,
            'fixed_weight': 0.30
        },
        {
            'name': 'Baseline (w/o all)',
            'use_residual': False,
            'use_attention': False,
            'use_dynamic_weighting': False,
            'fixed_weight': 0.15
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n训练配置: {config['name']}")
        print("-" * 60)
        
        # 创建模型
        model = AblationPINN(
            n_qubits=n_qubits,
            input_dim=input_dim,
            hidden_dims=[256, 128],
            use_residual=config['use_residual'],
            use_attention=config['use_attention'],
            use_dynamic_weighting=config['use_dynamic_weighting'],
            fixed_weight=config['fixed_weight']
        )
        
        # 训练
        trainer = AblationTrainer(model, device=device, warmup_epochs=5)
        best_fidelity = trainer.train(train_loader, val_loader, epochs=epochs)
        
        # 保存结果
        result = {
            'config': config['name'],
            'best_fidelity': best_fidelity,
            'final_fidelity': trainer.fidelities[-1],
            'final_cv': trainer.constraint_violations[-1],
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'fidelities': trainer.fidelities,
            'constraint_violations': trainer.constraint_violations
        }
        results.append(result)
        
        print(f"最佳保真度: {best_fidelity:.4f}")
    
    return results


def plot_ablation_results(results, save_dir='./ablation_results'):
    """绘制消融实验结果"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 保真度对比柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    configs = [r['config'] for r in results]
    fidelities = [r['best_fidelity'] for r in results]
    colors = sns.color_palette("husl", len(configs))
    
    bars = ax.bar(configs, fidelities, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Fidelity', fontsize=14, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_title('Ablation Study: Component Contribution Analysis', fontsize=16, fontweight='bold')
    ax.set_ylim([min(fidelities) * 0.95, max(fidelities) * 1.02])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar, fid in zip(bars, fidelities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{fid:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ablation_fidelity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 训练曲线对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 训练损失
    ax = axes[0, 0]
    for result in results:
        ax.plot(result['train_losses'], label=result['config'], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    
    # 验证损失
    ax = axes[0, 1]
    for result in results:
        ax.plot(result['val_losses'], label=result['config'], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    
    # 保真度曲线
    ax = axes[1, 0]
    for result in results:
        ax.plot(result['fidelities'], label=result['config'], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fidelity', fontsize=12, fontweight='bold')
    ax.set_title('Fidelity During Training', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    
    # 约束违反度
    ax = axes[1, 1]
    for result in results:
        ax.semilogy(result['constraint_violations'], label=result['config'], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Constraint Violation (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Constraint Violation During Training', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ablation_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 组件贡献热力图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算相对于baseline的改进
    baseline_fid = next(r['best_fidelity'] for r in results if 'Baseline' in r['config'])
    improvements = []
    labels = []
    
    for result in results:
        if 'Baseline' not in result['config']:
            improvement = (result['best_fidelity'] - baseline_fid) / baseline_fid * 100
            improvements.append(improvement)
            labels.append(result['config'])
    
    # 创建热力图数据
    heatmap_data = np.array(improvements).reshape(-1, 1)
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=['Improvement (%)'], yticklabels=labels,
                cbar_kws={'label': 'Improvement over Baseline (%)'},
                ax=ax, vmin=-5, vmax=15)
    ax.set_title('Component Contribution: Improvement over Baseline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ablation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 保存结果表格
    df = pd.DataFrame([
        {
            'Configuration': r['config'],
            'Best Fidelity': f"{r['best_fidelity']:.4f}",
            'Final Fidelity': f"{r['final_fidelity']:.4f}",
            'Final Constraint Violation': f"{r['final_cv']:.2e}"
        }
        for r in results
    ])
    df.to_csv(f'{save_dir}/ablation_results_table.csv', index=False)
    
    print(f"\n结果已保存到: {save_dir}")
    print("\n结果摘要:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    # 运行消融实验
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    results = run_ablation_study(n_qubits=3, n_train=3000, n_val=800, epochs=25, device=device)
    
    # 绘制结果
    plot_ablation_results(results, save_dir='./ablation_results')
    
    # 保存详细结果
    import json
    with open('./ablation_results/detailed_results.json', 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        json_results = []
        for r in results:
            json_r = {k: v for k, v in r.items() if k not in ['train_losses', 'val_losses', 'fidelities', 'constraint_violations']}
            json_r['final_train_loss'] = float(r['train_losses'][-1])
            json_r['final_val_loss'] = float(r['val_losses'][-1])
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)
    
    print("\n消融实验完成！")

