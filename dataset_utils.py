import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from itertools import combinations
import numpy as np 
from rdkit import Chem
from rdkit.Chem import AllChem
# from mordred import Calculator, descriptors,is_missing
from compound_tools import mol_to_geognn_graph_data_MMFF3d,mord
from hypergraph_utils_data_conj import *
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

class OneHotTransform:
    def __init__(self, untransformed):
        self._data = untransformed.data
        self._slices = untransformed.slices

        self.full_x = self._data.x
        self.full_x_slices = self._slices['x']

        self.full_edges = self._data.edge_attr
        self.full_edge_slices = self._slices['edge_attr']

        x_tensors = []
        for i in range(self.full_x.shape[1]):
            unique, unique_indices = torch.unique(self.full_x[:, i], return_inverse=True)
            if len(unique) > 2:
                x_tensors.append(F.one_hot(unique_indices, num_classes=len(unique)).to(torch.float32))
            elif len(unique) == 2:
                uv = set([float(v) for v in unique.tolist()])
                if uv.issubset({0.0, 1.0}):
                    x_tensors.append(self.full_x[:, i].reshape(-1, 1).to(torch.float32))
                else:
                    x_tensors.append(F.one_hot(unique_indices, num_classes=len(unique)).to(torch.float32))
            else:
                x_tensors.append(self.full_x[:, i].reshape(-1, 1).to(torch.float32))
        self.new_full_x = torch.cat(x_tensors, dim=1)

        e_tensors = []
        for i in range(self.full_edges.shape[1]):
            unique, unique_indices = torch.unique(self.full_edges[:, i], return_inverse=True)
            if len(unique) > 2:
                e_tensors.append(F.one_hot(unique_indices, num_classes=len(unique)).to(torch.float32))
            elif len(unique) == 2:
                uv = set([float(v) for v in unique.tolist()])
                if uv.issubset({0.0, 1.0}):
                    e_tensors.append(self.full_edges[:, i].reshape(-1, 1).to(torch.float32))
                else:
                    e_tensors.append(F.one_hot(unique_indices, num_classes=len(unique)).to(torch.float32))
            else:
                e_tensors.append(self.full_edges[:, i].reshape(-1, 1).to(torch.float32))
        self.new_full_edges = torch.cat(e_tensors, dim=1)

        self.index = 0
        self.edge_index = 0

    def __call__(self, data: Data):
        start = self.full_x_slices[self.index]
        end = self.full_x_slices[self.index + 1]
        data.x = self.new_full_x[start:end]

        start_e = self.full_edge_slices[self.index]
        end_e = self.full_edge_slices[self.index + 1]
        data.edge_attr = self.new_full_edges[start_e:end_e]

        self.index += 1
        return data

class FrozenOneHotTransform:
    def __init__(self, spec):
        self.spec = spec
        self.x_uniques = spec["x_uniques"]
        self.e_uniques = spec["e_uniques"]
        self.x_bin_mask = spec["x_bin_mask"]
        self.e_bin_mask = spec["e_bin_mask"]

        self.x_maps = []
        for u in self.x_uniques:
            if u is None:
                self.x_maps.append(None)
            else:
                self.x_maps.append({float(v): i for i, v in enumerate(u)})

        self.e_maps = []
        for u in self.e_uniques:
            if u is None:
                self.e_maps.append(None)
            else:
                self.e_maps.append({float(v): i for i, v in enumerate(u)})

    @staticmethod
    def build_spec(untransformed):
        data = untransformed.data
        full_x = data.x
        full_e = data.edge_attr

        x_uniques = []
        x_bin_mask = []
        for i in range(full_x.shape[1]):
            u = torch.unique(full_x[:, i]).tolist()
            if len(u) > 2:
                u = sorted([float(v) for v in u])
                x_uniques.append(u)
                x_bin_mask.append(0)
            elif len(u) == 2:
                uv = set([float(v) for v in u])
                if uv.issubset({0.0, 1.0}):
                    x_uniques.append(None)
                    x_bin_mask.append(1)
                else:
                    u = sorted([float(v) for v in u])
                    x_uniques.append(u)
                    x_bin_mask.append(0)
            else:
                x_uniques.append(None)
                x_bin_mask.append(1)

        e_uniques = []
        e_bin_mask = []
        for i in range(full_e.shape[1]):
            u = torch.unique(full_e[:, i]).tolist()
            if len(u) > 2:
                u = sorted([float(v) for v in u])
                e_uniques.append(u)
                e_bin_mask.append(0)
            elif len(u) == 2:
                uv = set([float(v) for v in u])
                if uv.issubset({0.0, 1.0}):
                    e_uniques.append(None)
                    e_bin_mask.append(1)
                else:
                    u = sorted([float(v) for v in u])
                    e_uniques.append(u)
                    e_bin_mask.append(0)
            else:
                e_uniques.append(None)
                e_bin_mask.append(1)

        return {
            "x_uniques": x_uniques,
            "e_uniques": e_uniques,
            "x_bin_mask": x_bin_mask,
            "e_bin_mask": e_bin_mask,
        }

    def _one_hot_from_map(self, values, mapping):
        n = len(values)
        k = len(mapping)
        out = torch.zeros((n, k), dtype=torch.float32)
        idx = [mapping.get(float(v), -1) for v in values]
        idx = torch.tensor(idx, dtype=torch.long)
        m = idx >= 0
        if torch.any(m):
            rows = torch.arange(n, dtype=torch.long)[m]
            cols = idx[m]
            out[rows, cols] = 1.0
        return out

    def __call__(self, data: Data):
        x_out = []
        for i in range(data.x.shape[1]):
            if self.x_bin_mask[i] == 1:
                x_out.append(data.x[:, i].reshape(-1, 1).to(torch.float32))
            else:
                x_out.append(self._one_hot_from_map(data.x[:, i].tolist(), self.x_maps[i]))
        data.x = torch.cat(x_out, dim=1)

        e_out = []
        for i in range(data.edge_attr.shape[1]):
            if self.e_bin_mask[i] == 1:
                e_out.append(data.edge_attr[:, i].reshape(-1, 1).to(torch.float32))
            else:
                e_out.append(self._one_hot_from_map(data.edge_attr[:, i].tolist(), self.e_maps[i]))
        data.edge_attr = torch.cat(e_out, dim=1)
        return data

def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses



def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """the atoms of mol will be changed in some cases."""
        # conf = mol.GetConformer()
        # atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        # return mol,atom_poses
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
            ### MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            index = np.argmin([x[1] for x in res])
            energy = res[index][1]
            conf = new_mol.GetConformer(id=int(index))
        except:
            print('error for get MMFF atom poses')
            new_mol = mol 
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()
        
        atom_poses = get_atom_poses(new_mol, conf)
        if return_energy:
            return new_mol, atom_poses, energy
        else:
            return new_mol, atom_poses
            
def create_hypergraph_dataset(dataset,multi_task=None,num_workers=None, return_label_dicts=False):
    graph_list=[]
    total_erro=0
    mismatch=0
    atom_type_label_list=[]
    bond_type_label_list=[]
    angle_type_label_list=[]
    atom_type_mapped=[]
    bond_type_mapped=[]
    angle_type_mapped=[]
    if num_workers is None:
        v=os.environ.get('NUM_THREADS')
        if v:
            try:
                num_workers=int(v)
            except:
                num_workers=None
    if not num_workers or num_workers<=1:
        for data in tqdm(dataset):
            try:
                smile=data.smiles
                mol = AllChem.MolFromSmiles(smile)
                mol_3d_info,atom_poses = mol_to_geognn_graph_data_MMFF3d(mol)
                if multi_task:
                    graph,atom_type_label,bond_type_label,angle_type_label=generate_mol_graph(data,mol_3d_info,mol,multi_task,atom_poses)
                    atom_type_label_list.append(atom_type_label)
                    atom_type_mapped+=atom_type_label
                    bond_type_label_list.append(bond_type_label)
                    bond_type_mapped+=bond_type_label
                    angle_type_label_list.append(angle_type_label)
                    angle_type_mapped+=angle_type_label
                else:
                    graph=generate_mol_graph(data,mol_3d_info,mol,multi_task,atom_poses)
                graph_list.append(graph)
                if  mol_3d_info['bond_angle'].shape[0] != 0:
                    if len(graph.conj_type_index[0])!=0:
                        if len(graph.atom_type_index[0])==graph.x.shape[0] and len(graph.bond_type_index[0])==graph.bond_feature.shape[0] and len(graph.angle_type_index[0])==graph.angle_feature.shape[0] and len(graph.conj_type_index[0])==graph.conj_feature.shape[0]:
                            pass
                        else:
                            print('mismatch!!!')
                            print(len(graph.atom_type_index[0]),graph.x.shape[0])
                            print(len(graph.bond_type_index[0]),graph.bond_feature.shape[0])
                            print(len(graph.angle_type_index[0]),graph.angle_feature.shape[0])
                            print(len(graph.conj_type_index[0]),graph.conj_feature.shape[0])
                            mismatch+=1
                else:
                    pass
            except:
                total_erro+=1
                pass
    else:
        def _worker(idx,data):
            try:
                smile=data.smiles
                mol = AllChem.MolFromSmiles(smile)
                mol_3d_info,atom_poses = mol_to_geognn_graph_data_MMFF3d(mol)
                if multi_task:
                    graph,atom_type_label,bond_type_label,angle_type_label=generate_mol_graph(data,mol_3d_info,mol,multi_task,atom_poses)
                else:
                    graph=generate_mol_graph(data,mol_3d_info,mol,multi_task,atom_poses)
                    atom_type_label=bond_type_label=angle_type_label=None
                m=0
                if  mol_3d_info['bond_angle'].shape[0] != 0:
                    if len(graph.conj_type_index[0])!=0:
                        if len(graph.atom_type_index[0])==graph.x.shape[0] and len(graph.bond_type_index[0])==graph.bond_feature.shape[0] and len(graph.angle_type_index[0])==graph.angle_feature.shape[0] and len(graph.conj_type_index[0])==graph.conj_feature.shape[0]:
                            pass
                        else:
                            print('mismatch!!!')
                            print(len(graph.atom_type_index[0]),graph.x.shape[0])
                            print(len(graph.bond_type_index[0]),graph.bond_feature.shape[0])
                            print(len(graph.angle_type_index[0]),graph.angle_feature.shape[0])
                            print(len(graph.conj_type_index[0]),graph.conj_feature.shape[0])
                            m=1
                else:
                    pass
                return idx,graph,atom_type_label,bond_type_label,angle_type_label,m
            except:
                return idx,None,None,None,None,0
        total=len(dataset)
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            pbar=tqdm(total=total)
            futures=[ex.submit(_worker,i,data) for i,data in enumerate(dataset)]
            results=[]
            for f in as_completed(futures):
                results.append(f.result())
                pbar.update(1)
            pbar.close()
        results.sort(key=lambda x:x[0])
        for _,graph,atom_type_label,bond_type_label,angle_type_label,m in results:
            if graph is None:
                total_erro+=1
                continue
            if multi_task:
                atom_type_label_list.append(atom_type_label)
                atom_type_mapped+=atom_type_label
                bond_type_label_list.append(bond_type_label)
                bond_type_mapped+=bond_type_label
                angle_type_label_list.append(angle_type_label)
                angle_type_mapped+=angle_type_label
            graph_list.append(graph)
            mismatch+=m
    print('create_hypergraph_data')
    hyperedge_type_list=[]
    for graph in graph_list:
        hyperedge_type_list+=list(graph.hyperedge_type.keys())
    hyperedge_type_list=list(set(hyperedge_type_list))
    alphabet={k:i for i,k in enumerate(hyperedge_type_list)}
    print('created_hyperedge_type_alphabet')
    for id,graph in enumerate(graph_list):
        graph.hyperedge_type=torch.tensor([alphabet[k]for k in graph.hyperedge_type.keys()],dtype=torch.long)
    print('mapping hyperedge type to id')
    if multi_task:
        atom_type_label_dict={k:i for i,k in enumerate(set(atom_type_mapped))}
        bond_type_label_dict={k:i for i,k in enumerate(set(bond_type_mapped))}
        angle_type_label_dict={k:i for i,k in enumerate(set(angle_type_mapped))}
        for id,graph in enumerate(graph_list):
            graph.atom_type_label=torch.tensor([atom_type_label_dict[k] for k in atom_type_label_list[id]],dtype=torch.long)
            graph.bond_type_label=torch.tensor([bond_type_label_dict[k] for k in bond_type_label_list[id]],dtype=torch.long)
            graph.angle_type_label=torch.tensor([angle_type_label_dict[k] for k in angle_type_label_list[id]],dtype=torch.long)
    if return_label_dicts and multi_task:
        return (total_erro,mismatch),graph_list,alphabet,atom_type_label_dict,bond_type_label_dict,angle_type_label_dict
    else:
        return (total_erro,mismatch),graph_list,alphabet

class GraphDataset(Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        # 支持按索引列表/ndarray/tensor 切片，返回子数据集，供 ScaffoldSplitter 使用
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            subset = [self.graph_list[i] for i in idx]
            return GraphDataset(subset)
        # 单个索引返回单个图
        return self.graph_list[idx]

def convert_hypergraph_matrix_to_edge_index(adjacency_matrix):
    num_nodes, num_edges = adjacency_matrix.shape
    edge_index_list = []

    for edge_id in range(num_edges):
        for node_id in range(num_nodes):
            if adjacency_matrix[node_id, edge_id] != 0:
                edge_index_list.append([node_id, edge_id])

    # Convert list to a 2D PyTorch tensor and transpose it
    edge_index = torch.tensor(edge_index_list).t()
    return edge_index


def get_jaccard_matrix(hyperedge_index):
    hyperedge_index=hyperedge_index.numpy().transpose()
    unique_values = np.unique(hyperedge_index[:, 1])

    # 使用列表推导式生成结果
    result = [[row[0] for row in hyperedge_index if row[1] == value] for value in unique_values]

    sets = [set(sub_list) for sub_list in result]


    def jaccard_similarity(set_a, set_b):
        """计算两个集合的 Jaccard 相似系数"""
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union != 0 else 0
    # 初始化相似系数矩阵
    n = len(sets)
    jaccard_matrix = np.eye(n, n)

    # 计算每对集合的 Jaccard 相似系数
    for i in range(n):
        for j in range(i + 1, n):
            similarity = jaccard_similarity(sets[i], sets[j])
            jaccard_matrix[i, j] = similarity
            jaccard_matrix[j, i] = similarity  # 矩阵是对称的

    # 输出矩阵
    rows, cols = np.nonzero(jaccard_matrix)
    weights = jaccard_matrix[rows, cols]

    # 将索引和权重转换为 Tensor
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_weights = torch.tensor(weights, dtype=torch.float)
    return edge_index,edge_weights

# def create_batch_data(dataset, batch_size,shuffle=True):
#     indices = list(range(len(dataset)))
#     if shuffle:
#         random.shuffle(indices)
#     all_data=[]
#     all_label=[]
#     for start_idx in range(0, len(dataset), batch_size):
#         end_idx = min(start_idx + batch_size, len(dataset))
#         batch_indices = indices[start_idx:end_idx]
#         # print(batch_indices)
#         batch = [dataset[i] for i in batch_indices]
#         # print(len(batch[0][0]))
#         # 提取图数据和目标值
#         hyper_matrix=[]
#         atom_feature=[]
#         bond_feature=[]
#         angle_feature=[]
#         num_bond_num_angle=[]
#         num_atom_num_bond=[]
#         target=[]
#         for i in range(len(batch)):
#             graph=batch[i]
#             hyper_matrix.append(graph['hypergraph'])
#             atom_feature.append(graph['atom_feature'])
#             bond_feature.append(graph['bond_feature'])
#             angle_feature.append(graph['angle_feature'])
#             num_bond_num_angle.append(graph['num_bond_num_angle'])
#             num_atom_num_bond.append(graph['num_atom_num_bond'])

#             label=torch.tensor(graph['label'],dtype=torch.float32)
#             target.append(label)
#         node_batch_index = torch.cat([torch.full((tensor.size(0),), i, dtype=torch.int) for i, tensor in enumerate(atom_feature)])
#         edge_batch_index = torch.cat([torch.full((tensor.size(1),), i, dtype=torch.int) for i, tensor in enumerate(hyper_matrix)])
#         hyper_matrix=combine_batch(hyper_matrix)
#         hypergraph=convert_hypergraph_matrix_to_edge_index(hyper_matrix)
#         jaccard=get_jaccard_matrix(hypergraph)
#         atom_feature=torch.concat(atom_feature,dim=0)
#         bond_feature=torch.concat(bond_feature,dim=0)
#         angle_feature=torch.concat(angle_feature)
#         num_bond_num_angle=combine_batch(num_bond_num_angle)
#         num_atom_num_bond=combine_batch(num_atom_num_bond)
#         target=torch.stack(target,dim=0)
#         # print(target.shape)
#         batch_graph={
#                         'hypergraph': hypergraph,
#                         'atom_feature': atom_feature,
#                         'bond_feature': bond_feature,
#                         'angle_feature': angle_feature,
#                         'num_bond_num_angle': num_bond_num_angle,
#                         'num_atom_num_bond':num_atom_num_bond,
#                         'batch_idx':node_batch_index,
#                         'hyper_matrix':hyper_matrix,
#                         'edge_batch_idx':edge_batch_index,
#                         'jaccard_matrix':jaccard
#                     }
#         all_data.append(batch_graph)
#         all_label.append(target)
#     return  all_data, all_label

def combine_batch(tensors):
    # 计算所有Tensor的总行数和总列数
    total_rows = sum(tensor.size(0) for tensor in tensors)
    total_cols = sum(tensor.size(1) for tensor in tensors)

    # 创建一个全零的大Tensor
    combine_matrix = torch.zeros((total_rows, total_cols))

    # 当前行和列的起始位置
    current_row = 0
    current_col = 0

    # 遍历每个Tensor，将其复制到大Tensor中对应的位置
    for tensor in tensors:
        rows, cols = tensor.size()
        combine_matrix[current_row:current_row + rows, current_col:current_col + cols] = tensor
        current_row += rows  # 更新当前行的位置
        current_col += cols  # 更新当前列的位置
    return combine_matrix

def  get_batch_data(graphs,targets):
    for i in range(len(graphs)):
        batch=(graphs[i],targets[i])
        yield batch

class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights=None):
        super(MultiTaskLoss, self).__init__()
        
        # 默认权重设置
        if task_weights is None:
            task_weights = {
                'morgan': 1.0,
                'logp': 1.0,
                'tpsa': 1.0,
            }
        
        self.task_weights = task_weights
        
        # 不同任务的损失函数
        self.morgan_loss = nn.BCEWithLogitsLoss()
        self.logp_loss = nn.MSELoss()
        self.tpsa_loss = nn.MSELoss()
    
    def forward(self, predictions, targets, masks=None):
        losses = {}
        total_loss = 0
        # print(type(predictions['morgan']),predictions['morgan'].shape)
        # print(type(targets['morgan']),targets['morgan'].shape)
        # Morgan指纹损失（多标签二分类）
        morgan_loss = self.morgan_loss(predictions['morgan'], targets['morgan'].float())
        losses['morgan'] = morgan_loss
        total_loss += self.task_weights['morgan'] * morgan_loss
        
        # LogP损失（回归）
        logp_loss = self.logp_loss(predictions['logp'], targets['logp'])
        losses['logp'] = logp_loss
        total_loss += self.task_weights['logp'] * logp_loss
        
        # TPSA损失（回归）
        tpsa_loss = self.tpsa_loss(predictions['tpsa'], targets['tpsa'])
        losses['tpsa'] = tpsa_loss
        total_loss += self.task_weights['tpsa'] * tpsa_loss
        
        losses['total'] = total_loss
        return losses
