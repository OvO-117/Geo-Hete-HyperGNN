from pickle import TRUE
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet
from torch_geometric import nn
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from torch_geometric.nn.pool import global_mean_pool
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
# from pysmiles import read_smiles #Unused
from tqdm import tqdm
from time import time
from scipy.stats import linregress
#Random graphs
import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
from dataset_utils import *
from torcheval.metrics.functional import r2_score
from torch_geometric.data import DataLoader
# from pretrain_final_CHHTransformer_add_sah import *
# from final_CHHTransformer_add_sah import *
from CHHTrans_hete import HyperGrpahTransformer
import os
import argparse
# 新增：引入 ScaffoldSplitter
from splitters import ScaffoldSplitter
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.metrics import roc_auc_score  # 添加ROC-AUC
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 添加RMSE和MAE
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train(model, train_loader, optimizer, loss_fn, args, loss_weight=0.5):

    epoch_loss = 0
    bt_loss = 0
    total_pred_probs = []
    total_predictions = []  
    total_y = []
    model.train()
    num_loss = 0
    correct = 0
    total = 0
    total_masks = []  # 新增：收集掩码用于AUC

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(args.device)
        predict, bt,_ = model(batch)

        # 预测结果合法性检查
        if predict is None or torch.isnan(predict).any() or torch.isinf(predict).any():
            # print(f"Warning: Invalid prediction detected in training (None/NaN/Inf), skipping batch")
            continue

        if args.num_classification is not None:
            # 分类任务（支持多标签）
            if (batch.y.dim() == 2 and batch.y.size(1) > 1) or (predict.dim() == 2 and predict.size(1) > 1):
                # 多任务：掩码 + 标签映射（支持 0/2 编码或 NaN 缺失）
                target_raw = batch.y.float()
                has_two = (target_raw == 2).any()
                mask = (target_raw != 0).float() if has_two else (~torch.isnan(target_raw)).float()
                target_bin = (target_raw == 2).float() if has_two else torch.nan_to_num(target_raw, nan=0.0)

                # 按掩码计算逐元素 BCE，再按有效元素数归一
                bce = torch.nn.BCEWithLogitsLoss(reduction='none')(predict, target_bin)
                valid_count = mask.sum()
                if valid_count.item() == 0:
                    # 当前 batch 全是缺失标签，跳过
                    continue
                loss = (bce * mask).sum() / valid_count

                pred_probs = torch.sigmoid(predict)
                pred_labels = (pred_probs >= 0.5).long()

                # 仅统计有效标签位置的准确率
                correct += (((pred_labels == target_bin.long()) & mask.bool()).sum().item())
                total += valid_count.item()

                # 记录用于后续AUC计算的数据
                total_y.append(target_bin.cpu())
                total_pred_probs.append(pred_probs.detach().cpu())
                total_masks.append(mask.cpu())
            else:
                # Single-task classification
                if predict.dim() == 1 or predict.size(-1) == 1:
                    # Binary classification with single logit
                    target = batch.y.view(-1).float().unsqueeze(1)
                    loss = torch.nn.BCEWithLogitsLoss()(predict, target)
                    pred_probs = torch.sigmoid(predict)
                    pred_labels = (pred_probs >= 0.5).long().view(-1)
                    total_y.append(target.view(-1).long().cpu())
                    total_pred_probs.append(pred_probs.detach().cpu())
                    correct += (pred_labels == target.view(-1).long()).sum().item()
                    total += target.size(0)
                else:
                    # Multi-class classification
                    target = batch.y.view(-1).long()
                    loss = torch.nn.CrossEntropyLoss()(predict, target)
                    total_y.append(target.cpu())
                    pred_probs = F.softmax(predict, dim=1)
                    total_pred_probs.append(pred_probs.detach().cpu())
                    pred_labels = predict.argmax(dim=1)
                    correct += (pred_labels == target).sum().item()
                    total += target.size(0)
        else:
            # 回归任务
            target = batch.y.view(-1).float()
            predict = predict.squeeze()  # 确保预测值的维度正确
            loss = loss_fn(predict, target)
            total_y.append(target.cpu())
            total_predictions.append(predict.detach().cpu())

        total_loss = loss_weight * loss + (1 - loss_weight) * bt

        total_loss.backward()
        optimizer.step()
        num_loss += 1
        epoch_loss += loss.item()
        bt_loss += bt.item()

    if len(total_y) == 0:
        # print("Warning: No valid predictions found in training")
        if args.num_classification is not None:
            return float('inf'), 0.0, 0.0, [], [], float('inf')
        else:
            return float('inf'), 0.0, 0.0, 0.0, [], [], float('inf')

    if args.num_classification is not None:
        # 分类任务评价指标
        acc = correct / total if total > 0 else 0.0

        # 新增：多任务 AUC 使用掩码逐任务计算，排除缺失
        y_true = torch.cat(total_y, dim=0).numpy()
        y_probs = torch.cat(total_pred_probs, dim=0).numpy()
        try:
            if y_probs.ndim == 2 and y_probs.shape[1] > 1:
                mask = torch.cat(total_masks, dim=0).numpy().astype(bool)
                task_aucs = []
                num_tasks = y_probs.shape[1]
                for t in range(num_tasks):
                    valid = mask[:, t]
                    if valid.sum() < 2:
                        continue
                    y_t = y_true[valid, t]
                    p_t = y_probs[valid, t]
                    # 若只有单一类别，则 AUC 不可计算，跳过该任务
                    if len(np.unique(y_t)) < 2:
                        continue
                    task_aucs.append(roc_auc_score(y_t, p_t))
                auc = float(np.mean(task_aucs)) if len(task_aucs) > 0 else 0.0
            elif y_probs.ndim == 2 and y_probs.shape[1] == 2:
                auc = roc_auc_score(y_true, y_probs[:, 1])
            elif y_probs.ndim == 2 and y_probs.shape[1] > 2:
                auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
            else:
                auc = roc_auc_score(y_true, y_probs.reshape(-1))
        except Exception as e:
            print(f"Warning: Failed to calculate AUC in training: {e}")
            auc = 0.0

        return epoch_loss/num_loss, acc, auc, total_y, total_pred_probs, bt_loss/num_loss
    else:
        # 回归任务评价指标
        y_true = torch.cat(total_y).numpy()
        y_pred = torch.cat(total_predictions).numpy()
        
        # 计算RMSE和MAE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # 计算R²
        try:
            r2 = r2_score(torch.cat(total_predictions), torch.cat(total_y)).item()
        except:
            r2 = 0.0

        return epoch_loss/num_loss, rmse, mae, r2, total_y, total_predictions, bt_loss/num_loss


@torch.no_grad()
def test(model, test_dataloader, args):
    model.eval()
    epoch_loss = 0
    total_pred_probs = []
    total_pred_labels = []
    total_predictions = []  # 用于回归任务
    total_y = []
    num_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(args.device)
            predict, bt,_ = model(batch)

            if predict is None or torch.isnan(predict).any() or torch.isinf(predict).any():
                print(f"Warning: Invalid prediction detected in testing (None/NaN/Inf), skipping batch")
                continue

            if args.num_classification is not None:
                # 分类任务
                if (batch.y.dim() == 2 and batch.y.size(1) > 1) or (predict.dim() == 2 and predict.size(1) > 1):
                    # 多任务：掩码 + 标签映射（支持 0/2 编码或 NaN 缺失）
                    target_raw = batch.y.float()
                    has_two = (target_raw == 2).any()
                    mask = (target_raw != 0).float() if has_two else (~torch.isnan(target_raw)).float()
                    target_bin = (target_raw == 2).float() if has_two else torch.nan_to_num(target_raw, nan=0.0)

                    bce = torch.nn.BCEWithLogitsLoss(reduction='none')(predict, target_bin)
                    valid_count = mask.sum()
                    if valid_count.item() == 0:
                        # 当前 batch 无有效标签，跳过
                        continue
                    loss = (bce * mask).sum() / valid_count

                    pred_probs = torch.sigmoid(predict)
                    pred_labels = (pred_probs >= 0.5).long()

                    # 将缺失位置标记为 -1，用于后续混淆矩阵跳过
                    pred_labels_masked = pred_labels.clone()
                    pred_labels_masked[~mask.bool()] = -1
                    target_masked = target_bin.long().clone()
                    target_masked[~mask.bool()] = -1

                    total_pred_probs.append(pred_probs.cpu())
                    total_pred_labels.append(pred_labels_masked.cpu())
                    total_y.append(target_masked.cpu())

                    # 仅统计有效标签位置的准确率
                    correct += (((pred_labels == target_bin.long()) & mask.bool()).sum().item())
                    total += valid_count.item()
                else:
                    # Single-task classification
                    if predict.dim() == 1 or predict.size(-1) == 1:
                        # Binary classification with single logit
                        target = batch.y.view(-1).float().unsqueeze(1)
                        loss = torch.nn.BCEWithLogitsLoss()(predict, target)
                        pred_probs = torch.sigmoid(predict)
                        pred_labels = (pred_probs >= 0.5).long().view(-1)
                        total_pred_probs.append(pred_probs.cpu())
                        total_pred_labels.append(pred_labels.cpu())
                        total_y.append(target.view(-1).long().cpu())
                        correct += (pred_labels == target.view(-1).long()).sum().item()
                        total += target.size(0)
                    else:
                        # Multi-class classification
                        target = batch.y.view(-1).long()
                        loss = torch.nn.CrossEntropyLoss()(predict, target)
                        pred_probs = F.softmax(predict, dim=1)
                        total_pred_probs.append(pred_probs.cpu())
                        pred_labels = predict.argmax(dim=1)
                        total_pred_labels.append(pred_labels.cpu())
                        total_y.append(target.cpu())
                        correct += (pred_labels == target).sum().item()
                        total += target.size(0)
            else:
                target = batch.y.view(-1).float()
                predict = predict.squeeze()  # 确保预测值的维度正确
                loss = torch.nn.MSELoss()(predict, target)
                
                total_predictions.append(predict.cpu())
                total_y.append(target.cpu())

            epoch_loss += loss.item()
            num_loss += 1

    if len(total_y) == 0:
        print("Warning: No valid predictions found in testing")
        if args.num_classification is not None:
            return float('inf'), 0.0, 0.0, [], []
        else:
            return float('inf'), 0.0, 0.0, 0.0, [], []

    if args.num_classification is not None:
        # 分类任务评价指标
        acc = correct / total if total > 0 else 0.0

        # 新增：多任务 AUC 使用掩码逐任务计算（排除 -1）
        y_true = torch.cat(total_y, dim=0).numpy()
        y_probs = torch.cat(total_pred_probs, dim=0).numpy()
        try:
            if y_probs.ndim == 2 and y_probs.shape[1] > 1:
                task_aucs = []
                num_tasks = y_probs.shape[1]
                for t in range(num_tasks):
                    valid = (y_true[:, t] != -1)
                    if valid.sum() < 2:
                        continue
                    y_t = y_true[valid, t]
                    p_t = y_probs[valid, t]
                    if len(np.unique(y_t)) < 2:
                        continue
                    task_aucs.append(roc_auc_score(y_t, p_t))
                auc = float(np.mean(task_aucs)) if len(task_aucs) > 0 else 0.0
            elif y_probs.ndim == 2 and y_probs.shape[1] == 2:
                auc = roc_auc_score(y_true, y_probs[:, 1])
            elif y_probs.ndim == 2 and y_probs.shape[1] > 2:
                auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
            else:
                auc = roc_auc_score(y_true, y_probs.reshape(-1))
        except Exception as e:
            print(f"Warning: Failed to calculate AUC in testing: {e}")
            auc = 0.0

        return epoch_loss/num_loss, acc, auc, total_y, total_pred_labels
    else:
        # 回归任务评价指标
        y_true = torch.cat(total_y).numpy()
        y_pred = torch.cat(total_predictions).numpy()
        
        # 计算RMSE和MAE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # 计算R²
        try:
            r2 = r2_score(torch.cat(total_predictions), torch.cat(total_y)).item()
        except:
            r2 = 0.0

        return epoch_loss/num_loss, rmse, mae, r2, total_y, total_predictions

def perfect_curve(preds, trues, threshold=None):
    pred_list = []
    true_list = []
    for i in preds:
        pred_value = i.flatten().tolist()
        pred_list+=pred_value
    for i in trues:
        true_value = i.flatten().tolist()
        true_list+=true_value
    return pred_list, true_list

def plot_perfect_curve_regression(y_true, y_pred, save_path, title=None):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.6, color='tab:blue', label='Pred vs True')

    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    plt.plot([vmin, vmax], [vmin, vmax], 'r--', label='Perfect: y=x')

    slope, intercept, r_value, _, _ = linregress(y_true, y_pred)
    xx = np.linspace(vmin, vmax, 100)
    plt.plot(xx, slope * xx + intercept, color='tab:orange', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r_value**2

    ax = plt.gca()
    ax.text(0.04, 0.96, f'RMSE={rmse:.3f}\nMAE={mae:.3f}\nR2={r2:.3f}',
            transform=ax.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', alpha=0.2))

    plt.xlabel('True')
    plt.ylabel('Predicted')
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='CHHTrans Training Script (aligned with tuning_main.py)')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='./data/bbbp', help='数据集路径')
    parser.add_argument('--dataset_name', type=str, default='bace_final', help='数据集名称')
    parser.add_argument('--num_classification', type=int, default=1, help='分类类别数，设为None进行回归任务')
    parser.add_argument('--num_task', type=int, default=1, help='多任务标签数（多标签分类），如 tox21=12')
   

    # 模型参数
    parser.add_argument('--emb_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout率')
    parser.add_argument('--num_blocks', type=int, default=3, help='Transformer块数量')
    parser.add_argument('--heads', type=int, default=4, help='注意力头数')
    # parser.add_argument('--temperature', type=float, default=0.5, help='温度参数')
    parser.add_argument('--ratio', type=float, default=0.6, help='比例参数')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--loss_weight', type=float, default=0.9, help='损失权重 (CE/总损失权重)')
    parser.add_argument('--eta_min_ratio', type=float, default=0.01, help='学习率调度器最小比例 (eta_min=lr*ratio)')
    
    # 数据分割参数
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--random_seed', type=int, default=991, help='随机种子')
    # 新增：明确 valid/test 比例，默认 0.1/0.1（与 8:1:1 对应）
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='验证集比例（默认 0.1）')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例（默认 0.1）')

    # 输出与设备
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--experiment_name', type=str, default='chhtrans_experiment', help='实验名称')
    parser.add_argument('--save_model', action='store_true', help='是否保存最佳模型')
    parser.add_argument('--device', type=str, default='cpu', help='设备类型')
    
    # 新增参数
    parser.add_argument('--use_pretrain', type=bool, default=True, help='是否使用预训练模型')
    parser.add_argument('--same_onehot_trasnform', type=bool, default=False, help='是否使用和预训练一样的onehot编码器')
    # parser.add_argument('--pre_trained_model_path', type=str, default='', help='预训练模型路径')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    plots_dir = os.path.join(experiment_dir, "plots")
    models_dir = os.path.join(experiment_dir, "models")
    logs_dir = os.path.join(experiment_dir, "logs")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    print(f"实验结果将保存到: {experiment_dir}")

    device = torch.device(args.device)

    # 超图缓存
    # if f'{args.dataset_name}.pt' not in os.listdir('dataset'):
    #     print(args.dataset_name)
    #     dataset_untransformed = MoleculeNet('../data', args.dataset_name)
    #     transform = OneHotTransform(dataset_untransformed)
    #     dataset_transformed = [transform(data) for data in dataset_untransformed]
    #     print("创建超图数据集...")
    #     hypergraph_list = create_hypergraph_dataset(dataset_transformed)
    #     print(f"超图数据集创建完成，包含{len(hypergraph_list)}个样本")
    #     torch.save(hypergraph_list, f'{args.dataset_name}.pt')
    # else:
    print('加载超图数据集...')
    
    # hypergraph_list = torch.load(f'dataset/{args.dataset_name}.pt')[1]
    hypergraph_list = torch.load(f'dataset/{args.dataset_name}.pt')[1]
    # hypergraph_list = torch.load(f'./processed_data/{args.dataset_name}.pt')
    print(hypergraph_list[0].x.shape)

    print(f"超图数据集加载完成，包含{len(hypergraph_list)}个样本")
        # 新增：如果缓存数据没有 smiles，则重建（用于 ScaffoldSplitter）
        # if len(hypergraph_list) > 0 and not hasattr(hypergraph_list[0], 'smiles'):
        #     print('检测到缓存超图样本缺少 smiles，正在重新生成以支持 Scaffold 划分...')
        #     dataset_untransformed = MoleculeNet('../data', args.dataset_name)
        #     transform = OneHotTransform(dataset_untransformed)
        #     dataset_transformed = [transform(data) for data in dataset_untransformed]
        #     hypergraph_list = create_hypergraph_dataset(dataset_transformed)
        #     torch.save(hypergraph_list, 'hypergraph_list.pt')
        #     print(f"重建完成，超图数据集包含{len(hypergraph_list)}个样本")
    # train_size = int(args.train_ratio * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(
    #     dataset, [train_size, test_size],
    #     generator=torch.Generator().manual_seed(args.random_seed)
    # )
    dataset = GraphDataset(hypergraph_list)
    splitter = ScaffoldSplitter()
    # 按 8:1:1 划分（允许通过 args.valid_ratio / args.test_ratio 调节）
    train_ratio = 1.0 - args.valid_ratio - args.test_ratio

    def _seed_worker(worker_id):
        worker_seed = args.random_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_dataset, valid_dataset, test_dataset = splitter.split(
        dataset,
        frac_train=train_ratio,
        frac_valid=args.valid_ratio,
        frac_test=args.test_ratio
    )
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True, worker_init_fn=_seed_worker)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True, worker_init_fn=_seed_worker)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True, worker_init_fn=_seed_worker)


    atom_dim = train_dataset[1].x.shape[1]
    bond_dim = train_dataset[1].bond_feature.shape[1]
    angle_dim = train_dataset[1].angle_feature.shape[1]
    # print(angle_dim)
    if args.use_pretrain:
        print('使用预训练模型')
        if args.same_onehot_trasnform:
            from CHHTrans_hete_change_init import PretrainedCHHTransformer
            model = PretrainedCHHTransformer(
                atom_dim=atom_dim,
                bond_dim=bond_dim,
                angle_dim=angle_dim,
                conj_dim=10,
                emb_dim=args.emb_dim,
                batch_size=args.batch_size,
                dropout=args.dropout,
                num_blocks=args.num_blocks,
                heads=args.heads,
                # ratio=args.ratio,
                num_classes=args.num_classification,  # None 表示回归；整数表示分类
                self_supervised=False,
                num_atom_classes=None, num_bond_classes=None,
                num_angle_classes=None, num_conj_classes=None,
                pretrained_path="/home/dt3/桌面/XZ/visial_chh/best_model_9.pth",
                map_location="cpu"
            ).to(device)
        else:
            from CHHTrans_hete import PretrainedCHHTransformer
            model = PretrainedCHHTransformer(
                atom_dim=atom_dim,
                bond_dim=bond_dim,
                angle_dim=angle_dim,
                conj_dim=10,
                emb_dim=args.emb_dim,
                batch_size=args.batch_size,
                dropout=args.dropout,
                num_blocks=args.num_blocks,
                heads=args.heads,
                # ratio=args.ratio,
                num_classes=args.num_classification,  # None 表示回归；整数表示分类
                self_supervised=False,
                num_atom_classes=None, num_bond_classes=None,
                num_angle_classes=None, num_conj_classes=None,
                pretrained_path="/home/dt3/桌面/XZ/visial_chh/best_model_9.pth",
                map_location="cpu"
            ).to(device)
    else:
        print('不使用预训练模型')
        model = HyperGrpahTransformer(
            atom_dim=atom_dim, bond_dim=bond_dim, angle_dim=angle_dim,batch_size=args.batch_size,
            emb_dim=args.emb_dim, dropout=args.dropout, conj_dim=10,heads=args.heads,
            num_blocks=args.num_blocks,
            num_classes=args.num_classification if args.num_classification is not None else 1,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * args.eta_min_ratio)
    if args.num_classification is not None:
        # 新增：区分多任务与单任务的损失
        
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
    else:
        loss_fn = torch.nn.MSELoss()

    # 训练记录
    all_epochs = np.arange(args.num_epochs)
    all_train_loss, all_train_bt = [], []
    all_test_loss = []
    
    # 根据任务类型初始化不同的评价指标记录
    if args.num_classification is not None:
        # 分类任务
        all_train_auc, all_test_auc = [], []
        best_test_metric = 0.0  # AUC越大越好
        metric_name = "AUC"
        is_higher_better = True
    else:
        # 回归任务
        all_train_rmse, all_train_mae, all_train_r2 = [], [], []
        all_test_rmse, all_test_mae, all_test_r2 = [], [], []
        best_test_metric = float('inf')  # RMSE越小越好
        metric_name = "RMSE"
        is_higher_better = False

    # 训练日志
    log_file = os.path.join(logs_dir, "training_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"实验开始时间: {datetime.now()}\n")
        f.write(f"参数配置:\n")
        for arg, value in vars(args).items():
            f.write(f"  {arg}: {value}\n")
        f.write("\n训练过程:\n")
        if args.num_classification is not None:
            f.write("Epoch\tTrain_AUC\tTest_AUC\tTrain_CE\tTrain_bt\tTest_CE\n")
        else:
            f.write("Epoch\tTrain_RMSE\tTrain_MAE\tTrain_R2\tTest_RMSE\tTest_MAE\tTest_R2\tTrain_MSE\tTrain_bt\tTest_MSE\n")

    # Early stopping
    best_epoch = -1
    best_model_state = None
    best_test_y_true = None
    best_test_y_pred = None
    patience, patience_counter = 1000, 0

    best_val_y_true = None
    best_val_y_pred = None
    for epoch in all_epochs:
        if args.num_classification is not None:
            # 分类任务
            train_ce, train_acc, train_auc, train_trues, train_probs, train_bt = train(model, train_loader, optimizer, loss_fn, args, args.loss_weight)
            test_ce, test_acc, test_auc, test_trues, test_pred_labels = test(model, test_loader, args)

            # 学习率步进
            scheduler.step()

            all_train_loss.append(train_ce)
            all_train_auc.append(train_auc)
            all_train_bt.append(train_bt)

            all_test_loss.append(test_ce)
            all_test_auc.append(test_auc)

            log_info = f'{epoch}\t{train_auc:.4f}\t{test_auc:.4f}\t{train_ce:.4f}\t{train_bt:.4f}\t{test_ce:.4f}'
            print(f'epoch:{epoch}, train_AUC:{train_auc:.4f}, test_AUC:{test_auc:.4f}, train_CE:{train_ce:.4f}, train_bt:{train_bt:.4f}, test_CE:{test_ce:.4f}')
            
            # 更新最佳
            current_test_metric = test_auc
            if test_auc > best_test_metric:
                best_test_metric = test_auc
                best_epoch = epoch
                patience_counter = 0
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_test_y_true = torch.cat(test_trues).numpy()
                best_test_y_pred = torch.cat(test_pred_labels).numpy()
            else:
                patience_counter += 1
        else:
            # 回归任务
            train_mse, train_rmse, train_mae, train_r2, train_trues, train_preds, train_bt = train(model, train_loader, optimizer, loss_fn, args, args.loss_weight)
            test_mse, test_rmse, test_mae, test_r2, test_trues, test_preds = test(model, test_loader, args)

            # 学习率步进
            scheduler.step()

            all_train_loss.append(train_mse)
            all_train_rmse.append(train_rmse)
            all_train_mae.append(train_mae)
            all_train_r2.append(train_r2)
            all_train_bt.append(train_bt)

            all_test_loss.append(test_mse)
            all_test_rmse.append(test_rmse)
            all_test_mae.append(test_mae)
            all_test_r2.append(test_r2)

            log_info = f'{epoch}\t{train_rmse:.4f}\t{train_mae:.4f}\t{train_r2:.4f}\t{test_rmse:.4f}\t{test_mae:.4f}\t{test_r2:.4f}\t{train_mse:.4f}\t{train_bt:.4f}\t{test_mse:.4f}'
            print(f'epoch:{epoch}, train_RMSE:{train_rmse:.4f}, train_MAE:{train_mae:.4f}, train_R2:{train_r2:.4f}, test_RMSE:{test_rmse:.4f}, test_MAE:{test_mae:.4f}, test_R2:{test_r2:.4f}, train_MSE:{train_mse:.4f}, train_bt:{train_bt:.4f}, test_MSE:{test_mse:.4f}')
            
            # 更新最佳（基于RMSE，越小越好）
            current_test_metric = test_rmse
            if test_rmse < best_test_metric:
                best_test_metric = test_rmse
                best_epoch = epoch
                patience_counter = 0
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_test_y_true = torch.cat(test_trues).numpy()
                best_test_y_pred = torch.cat(test_preds).numpy()
            else:
                patience_counter += 1

        with open(log_file, 'a') as f:
            f.write(log_info + '\n')

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_epoch < 0:
        if args.num_classification is not None:
            best_epoch = int(np.argmax(all_test_auc))
        else:
            best_epoch = int(np.argmin(all_test_rmse))

    # 输出最终结果
    if args.num_classification is not None:
        print(f'Train AUC: {max(all_train_auc):.4f} | Train CE: {min(all_train_loss):.4f}')
        print(f'Test AUC: {max(all_test_auc):.4f} | Test CE: {min(all_test_loss):.4f}')
        print(f'Best epoch: {best_epoch} with test_auc: {all_test_auc[best_epoch]:.4f} | test_ce: {all_test_loss[best_epoch]:.4f}')
        with open(log_file, 'a') as f:
            f.write(f"\n最终结果:\n")
            f.write(f"Train AUC: {max(all_train_auc):.4f} | Train CE: {min(all_train_loss):.4f}\n")
            f.write(f"Test AUC: {max(all_test_auc):.4f} | Test CE: {min(all_test_loss):.4f}\n")
            f.write(f"Best epoch: {best_epoch} with test_auc: {all_test_auc[best_epoch]:.4f} | test_ce: {all_test_loss[best_epoch]:.4f}\n")
            f.write(f"实验结束时间: {datetime.now()}\n")
    else:
        print(f'Train RMSE: {min(all_train_rmse):.4f} | Train MAE: {min(all_train_mae):.4f} | Train R2: {max(all_train_r2):.4f} | Train MSE: {min(all_train_loss):.4f}')
        print(f'Test RMSE: {min(all_test_rmse):.4f} | Test MAE: {min(all_test_mae):.4f} | Test R2: {max(all_test_r2):.4f} | Test MSE: {min(all_test_loss):.4f}')
        print(f'Best epoch: {best_epoch} with test_rmse: {all_test_rmse[best_epoch]:.4f} | test_mae: {all_test_mae[best_epoch]:.4f} | test_r2: {all_test_r2[best_epoch]:.4f} | test_mse: {all_test_loss[best_epoch]:.4f}')
        with open(log_file, 'a') as f:
            f.write(f"\n最终结果(训练过程统计，Val=Test)：\n")
            f.write(f"Train RMSE: {min(all_train_rmse):.4f} | Train MAE: {min(all_train_mae):.4f} | Train R2: {max(all_train_r2):.4f} | Train MSE: {min(all_train_loss):.4f}\n")
            f.write(f"Test RMSE: {min(all_test_rmse):.4f} | Test MAE: {min(all_test_mae):.4f} | Test R2: {max(all_test_r2):.4f} | Test MSE: {min(all_test_loss):.4f}\n")
            f.write(f"Best epoch(by Test RMSE): {best_epoch} | test_rmse: {all_test_rmse[best_epoch]:.4f} | test_mae: {all_test_mae[best_epoch]:.4f} | test_r2: {all_test_r2[best_epoch]:.4f} | test_mse: {all_test_loss[best_epoch]:.4f}\n")
            f.write(f"实验结束时间: {datetime.now()}\n")

    # 使用“test 上 RMSE 最优的 epoch 权重”在 valid 上评估
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.eval()
        gating_probs = []
        hook = None
        if hasattr(model, 'router_cls'):
            def _router_hook(module, inputs, output):
                probs = torch.softmax(output, dim=-1).detach().cpu()  # [B, 4]
                gating_probs.append(probs)
            hook = model.router_cls.register_forward_hook(_router_hook)

        if args.num_classification is not None:
            val_ce, val_acc, val_auc, val_trues, val_pred_labels = test(model, valid_loader, args)
            print(f'Valid AUC (best-test weights): {val_auc:.4f} | Valid CE: {val_ce:.4f}')
            with open(log_file, 'a') as f:
                f.write(f"Valid AUC (best-test weights): {val_auc:.4f} | Valid CE: {val_ce:.4f}\n")
            best_val_y_true = torch.cat(val_trues).numpy()
            best_val_y_pred = torch.cat(val_pred_labels).numpy()
        else:
            val_mse, val_rmse, val_mae, val_r2, val_trues, val_preds = test(model, valid_loader, args)
            print(f'Valid RMSE (best-test weights): {val_rmse:.4f} | Valid MAE: {val_mae:.4f} | Valid R2: {val_r2:.4f} | Valid MSE: {val_mse:.4f}')
            with open(log_file, 'a') as f:
                f.write(f"Valid RMSE (best-test weights): {val_rmse:.4f} | Valid MAE: {val_mae:.4f} | Valid R2: {val_r2:.4f} | Valid MSE: {val_mse:.4f}\n")
            best_val_y_true = torch.cat(val_trues).numpy()
            best_val_y_pred = torch.cat(val_preds).numpy()

        # 移除 hook
        if hook is not None:
            hook.remove()

        # 汇总并保存验证集专家权重图（最佳测试权重）
        if len(gating_probs) > 0:
            gating_cat = torch.cat(gating_probs, dim=0)      # [N, 4]
            gating_mean = gating_cat.mean(dim=0).numpy()      # [4]
            expert_names = ['Spatial Expert', 'Atom Expert', 'Bond Expert', 'Angle Expert', 'Conj Expert']
            plt.figure(figsize=(6, 4))
            colors = ['red', 'purple', 'orange', 'green', 'blue']
            plt.bar(expert_names, gating_mean, color=colors, alpha=0.25)
            plt.ylabel('Average Weight')
            plt.title('MoMHE Expert Weights')
            moe_path = os.path.join(plots_dir, "moe_expert_weights_valid.png")
            plt.tight_layout()
            plt.savefig(moe_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"验证集 MoE 专家权重图已保存到: {moe_path}")
    # 若需要可在此更新 best_val_y_true/best_val_y_pred 为 valid 的结果（分类任务）
    else:
        val_mse, val_rmse, val_mae, val_r2, val_trues, val_preds = test(model, valid_loader, args)
        print(f'Valid RMSE (best-test weights): {val_rmse:.4f} | Valid MAE: {val_mae:.4f} | Valid R2: {val_r2:.4f} | Valid MSE: {val_mse:.4f}')
        with open(log_file, 'a') as f:
            f.write(f"Valid RMSE (best-test weights): {val_rmse:.4f} | Valid MAE: {val_mae:.4f} | Valid R2: {val_r2:.4f} | Valid MSE: {val_mse:.4f}\n")
        # 更新散点图为 valid 集（使用最佳 test 权重）
        best_val_y_true = torch.cat(val_trues).numpy()
        best_val_y_pred = torch.cat(val_preds).numpy()
    # 混淆矩阵（仅分类任务）或散点图（回归任务）
    if args.num_classification is not None and best_val_y_true is not None and best_val_y_pred is not None:
        # 多任务多标签（二分类每任务）按任务分别画 2x2 混淆矩阵
        if isinstance(best_val_y_pred, np.ndarray) and best_val_y_pred.ndim == 2 and best_val_y_pred.shape[1] > 1:
            y_true_all = np.array(best_val_y_true)
            y_pred_all = np.array(best_val_y_pred)
            num_tasks = y_pred_all.shape[1]

            # 保障为二维 [N, num_task]
            if y_true_all.ndim == 1:
                y_true_all = y_true_all.reshape(-1, 1)
            if y_pred_all.ndim == 1:
                y_pred_all = y_pred_all.reshape(-1, 1)

            for task_idx in range(num_tasks):
                y_true_task = y_true_all[:, task_idx].astype(int)
                y_pred_task = y_pred_all[:, task_idx].astype(int)

                cm = np.zeros((2, 2), dtype=int)
                for t, p in zip(y_true_task, y_pred_task):
                    if t in (0, 1) and p in (0, 1):
                        cm[t, p] += 1

                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=['Pred 0', 'Pred 1'],
                            yticklabels=['True 0', 'True 1'])
                plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix (Task {task_idx}, Best Epoch)')
                cm_path = os.path.join(plots_dir, f"confusion_matrix_task{task_idx}.png")
                plt.tight_layout(); plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                print(f"任务 {task_idx} 的混淆矩阵已保存到: {cm_path}")
                plt.close()
        else:
            # 单标签分类：保持原逻辑
            num_classes = args.num_classification
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(best_val_y_true, best_val_y_pred):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    cm[t, p] += 1

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=[f'Pred {i}' for i in range(num_classes)],
                        yticklabels=[f'True {i}' for i in range(num_classes)])
            plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (Best Epoch)')
            cm_path = os.path.join(plots_dir, "confusion_matrix.png")
            plt.tight_layout(); plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {cm_path}")
            plt.close()
    elif args.num_classification is None and best_val_y_true is not None and best_val_y_pred is not None:
        plot_val_path = os.path.join(plots_dir, "perfect_curve_valid.png")
        plot_perfect_curve_regression(
            best_val_y_true,
            best_val_y_pred,
            plot_val_path,
            title="Regression Parity (Valid, Best-Test Weights)"
        )
        print(f"回归完美曲线（验证集）已保存到: {plot_val_path}")
    # 保存最佳模型（可选）
    if args.save_model and best_model_state is not None:
        model_path = os.path.join(models_dir, f"best_model_epoch_{best_epoch}.pth")
        save_dict = {
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'args': vars(args)
        }
        
        if args.num_classification is not None:
            save_dict.update({
                'train_auc': all_train_auc[best_epoch],
                'test_auc': all_test_auc[best_epoch],
                'train_ce': all_train_loss[best_epoch],
                'test_ce': all_test_loss[best_epoch],
            })
        else:
            save_dict.update({
                'train_rmse': all_train_rmse[best_epoch],
                'train_mae': all_train_mae[best_epoch],
                'train_r2': all_train_r2[best_epoch],
                'test_rmse': all_test_rmse[best_epoch],
                'test_mae': all_test_mae[best_epoch],
                'test_r2': all_test_r2[best_epoch],
                'train_mse': all_train_loss[best_epoch],
                'test_mse': all_test_loss[best_epoch],
            })
        
        torch.save(save_dict, model_path)
        print(f"最佳模型已保存到: {model_path}")

    # 保存训练过程数据
    training_data = {
        'epochs': list(all_epochs[:len(all_train_loss)]),
        'train_loss': all_train_loss,
        'test_loss': all_test_loss,
        'train_bt': all_train_bt,
        'best_epoch': best_epoch
    }
    
    if args.num_classification is not None:
        training_data.update({
            'train_auc': all_train_auc,
            'test_auc': all_test_auc,
        })
    else:
        training_data.update({
            'train_rmse': all_train_rmse,
            'train_mae': all_train_mae,
            'train_r2': all_train_r2,
            'test_rmse': all_test_rmse,
            'test_mae': all_test_mae,
            'test_r2': all_test_r2,
        })
    
    training_data_path = os.path.join(logs_dir, "training_data.pt")
    torch.save(training_data, training_data_path)
    print(f"训练数据已保存到: {training_data_path}")

    # 保存整体训练 Loss 曲线
    epochs_used = list(all_epochs[:len(all_train_loss)])
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_used, all_train_loss, label='Train Loss', color='tab:blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    train_loss_plot_path = os.path.join(plots_dir, "training_loss.png")
    plt.tight_layout()
    plt.savefig(train_loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练 Loss 曲线已保存到: {train_loss_plot_path}")
    print(f"\n所有结果已保存到实验目录: {experiment_dir}")
    print(f"  - 图片保存在: {plots_dir}")
    print(f"  - 日志保存在: {logs_dir}")
    if args.save_model:
        print(f"  - 模型保存在: {models_dir}")

if __name__ == '__main__':
    # args = parse_args()
    main()
