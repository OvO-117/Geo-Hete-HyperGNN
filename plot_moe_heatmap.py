import os
import argparse
import torch
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.data import DataLoader
from dataset_utils import GraphDataset
from final_CHHTransformer_add_sah import HyperGrpahTrasnformer

EXPERT_NAMES = ['Atom(X)', 'Bond', 'Angle', 'Conj']

def register_router_hooks(model, tasks):
    """
    为各任务的路由器注册 forward hook，抓取输出 logits，
    在 hook 中转为 softmax 概率并累积到内存。
    """
    gating = {task: [] for task in tasks}
    hooks = []

    def make_hook(task_name):
        def hook(module, inputs, output):
            probs = torch.softmax(output, dim=-1).detach().cpu()  # [B, 4]
            gating[task_name].append(probs)
        return hook

    if 'morgan' in tasks and hasattr(model, 'router_morgan'):
        hooks.append(model.router_morgan.register_forward_hook(make_hook('morgan')))
    if 'logp' in tasks and hasattr(model, 'router_logp'):
        hooks.append(model.router_logp.register_forward_hook(make_hook('logp')))
    if 'tpsa' in tasks and hasattr(model, 'router_tpsa'):
        hooks.append(model.router_tpsa.register_forward_hook(make_hook('tpsa')))

    # 单任务分类的情况
    if 'classification' in tasks and hasattr(model, 'router_cls'):
        hooks.append(model.router_cls.register_forward_hook(make_hook('classification')))

    return hooks, gating

def aggregate_gating(gating_dict):
    """
    将各任务收集到的 gating 拼接并计算均值，返回热力图矩阵。
    """
    task_order = list(gating_dict.keys())
    heatmap = []
    for task in task_order:
        if len(gating_dict[task]) == 0:
            heatmap.append(np.zeros((len(EXPERT_NAMES),), dtype=np.float32))
            continue
        mat = torch.cat(gating_dict[task], dim=0).numpy()  # [N, 4]
        mean_vec = mat.mean(axis=0)  # [4]
        heatmap.append(mean_vec)
    return np.stack(heatmap, axis=0), task_order

def plot_heatmap(heatmap, tasks, save_path='plots/moe_task_expert_heatmap.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 3 + 0.4 * len(tasks)))
    ax = sns.heatmap(
        heatmap,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=EXPERT_NAMES,
        yticklabels=[t.upper() for t in tasks],
        vmin=0.0,
        vmax=1.0,
        cbar=True
    )
    plt.xlabel('Experts')
    plt.ylabel('Tasks')
    plt.title('MoE Expert Contribution by Task')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'热力图已保存：{save_path}')
    plt.close()

def find_latest_best_checkpoint(results_root='./results'):
    import os
    if not os.path.isdir(results_root):
        return None
    candidates = []
    for name in os.listdir(results_root):
        dirpath = os.path.join(results_root, name)
        models_dir = os.path.join(dirpath, 'models')
        ckpt_path = os.path.join(models_dir, 'best_model.pth')
        if os.path.isfile(ckpt_path):
            candidates.append((os.path.getmtime(ckpt_path), ckpt_path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sample.pt', help='包含多任务标签的图数据（.pt）')
    parser.add_argument('--checkpoint', type=str, default='/home/ubuntu/桌面/AW/chhtransformer/results/chhtrans_experiment_20251016_175356/models/best_model.pth', help='模型权重路径（可选）')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--out', type=str, default='plots/moe_task_expert_heatmap.png')
    parser.add_argument('--max_batches', type=int, default=-1, help='限制评估批次数（-1为全部）')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 自动寻找最新的最佳权重（若未显式提供）
    if not args.checkpoint:
        auto_ckpt = find_latest_best_checkpoint(results_root='./results')
        if auto_ckpt:
            args.checkpoint = auto_ckpt
            print(f'自动加载最新最佳权重: {args.checkpoint}')
        else:
            print('警告：未找到最佳权重，将使用随机初始化模型')

    # 载入数据（sample.pt 由 pretrain_chh.py 生成，已包含 morgan/logp/tpsa）
    print(f'加载数据：{args.dataset}')
    hypergraph_list = torch.load(args.dataset)
    dataset = GraphDataset(hypergraph_list)
    loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=8)

    # 维度推断
    sample_graph = dataset[0]
    atom_dim = sample_graph.x.shape[1]
    bond_dim = sample_graph.bond_feature.shape[1]
    angle_dim = sample_graph.angle_feature.shape[1]

    # 构建模型：多任务（自监督）路径
    print('构建模型 HyperGrpahTrasnformer（self_supervised=True）...')
    model = HyperGrpahTrasnformer(
        atom_dim=atom_dim, bond_dim=bond_dim, angle_dim=angle_dim,
        emb_dim=args.emb_dim, dropout=args.dropout, conj_dim=10,
        num_blocks=args.num_blocks, heads=args.heads,
        temperature=args.temperature, ratio=args.ratio,
        num_classes=None, self_supervised=True
    ).to(device)
    model.eval()

    # 加载 checkpoint（可选）
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f'加载权重：{args.checkpoint}')
        state = torch.load(args.checkpoint, map_location=device)
        # 兼容仅保存 state_dict 的情况
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        elif isinstance(state, dict):
            model.load_state_dict(state)
        else:
            print('警告：无法识别的checkpoint格式，跳过加载')

    # 注册路由器 hooks
    tasks = ['morgan', 'logp', 'tpsa']  # 多任务可视化
    hooks, gating = register_router_hooks(model, tasks)

    # 前向遍历并收集 gating
    print('开始收集 gating 概率...')
    seen_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch)  # forward 会调用各路由器；hook 捕获输出
            seen_batches += 1
            if args.max_batches > 0 and seen_batches >= args.max_batches:
                break

    # 反注册 hooks
    for h in hooks:
        h.remove()

    # 聚合并绘制热力图
    heatmap, ordered_tasks = aggregate_gating(gating)
    print('任务-专家平均占比（行任务/列专家）：')
    for i, t in enumerate(ordered_tasks):
        row = ', '.join([f'{name}={heatmap[i, j]:.2f}' for j, name in enumerate(EXPERT_NAMES)])
        print(f'  {t}: {row}')
    plot_heatmap(heatmap, ordered_tasks, save_path=args.out)

if __name__ == '__main__':
    main()