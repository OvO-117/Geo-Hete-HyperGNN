from collections import defaultdict
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from rdkit import Chem
class HData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x.size(0)], [int(self.num_hyperedges)]], device=value.device)
        # if key == 'atom_bond_index':
        #     return torch.tensor([[self.x.size(0)], [int(self.num_bonds)]], device=value.device)
        # if key == 'atom_angle_index':
        #     return torch.tensor([[self.x.size(0)], [int(self.num_angles)]], device=value.device)
        # if key == 'bond_angle_index':
        #     return torch.tensor([[int(self.num_bonds)], [int(self.num_angles)]], device=value.device)
        # if key == 'jaccard_index':
        #     return torch.tensor([[int(self.num_hyperedges)], [int(self.num_hyperedges)]], device=value.device)
        # if key == 'atom_conj_index':
        #     return torch.tensor([[self.x.size(0)], [int(self.num_conj)]], device=value.device)
        # if key == 'atom_atom_index':
        #     return torch.tensor([[self.x.size(0)], [self.x.size(0)]], device=value.device)
        if key == 'atom_type_index':
            return torch.tensor([[self.x.size(0)], [int(self.num_hyperedges)]], device=value.device)
        if key == 'bond_type_index':
            return torch.tensor([[int(self.num_bonds)], [int(self.num_hyperedges)]], device=value.device)
        if key == 'angle_type_index':
            return torch.tensor([[int(self.num_angles)], [int(self.num_hyperedges)]], device=value.device)
        if key == 'conj_type_index':
            return torch.tensor([[int(self.num_conj)], [int(self.num_hyperedges)]], device=value.device)
        if key == 'atom_type_batch_index':
            return torch.tensor(1, device=value.device)
        if key == 'bond_type_batch_index':
            return torch.tensor(1, device=value.device)
        if key == 'angle_type_batch_index':
            return torch.tensor(1, device=value.device)
        if key == 'conj_type_batch_index':
            return torch.tensor(1, device=value.device)
        if key == 'hyperedge_batch_index':
            return torch.tensor(1, device=value.device)
        if key == 'node_batch_index':
            return torch.tensor(1, device=value.device)
        return super().__inc__(key, value, *args, **kwargs)
# from CHHTrans.ESOL.show_asprin import same_atom_dict, same_bond_dict
from hyperedge_construction import *
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import rdchem
from compound_tools import mol_to_geognn_graph_data_MMFF3d,mord
from rdkit.Chem import AllChem, MACCSkeys, DataStructs, Descriptors, Crippen
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, scatter)
def group_keys_by_value(input_dict):
    # 创建一个 defaultdict，按值来分组键
    grouped = defaultdict(list)
    
    # 遍历字典，将键放到以值为键的列表中
    for key, value in input_dict.items():
        grouped[value].append(key)
    
    # 将 defaultdict 转换成普通字典并返回
    result_dict = dict(grouped)
    
    return result_dict

def map_indices_to_values(index_dict, reference_list):
    # 创建一个新的字典，用于存储最终结果
    result_dict = {}

    # 遍历原字典，提取索引信息并根据索引获取对应的元素
    for key, indices in index_dict.items():
        # 获取根据索引值从 reference_list 中提取的元素
        result_dict[key] = list(set(list(itertools.chain.from_iterable([reference_list[i] for i in indices]))))
    return result_dict

def remove_duplicate_tuple_keys(input_dict):
        seen_keys = set()  # 用于追踪已出现的标准化键
        result_dict = {}    # 用于存储去重后的字典

        for key, value in input_dict.items():
            # 将元组中的元素进行排序，标准化元组
            normalized_key = tuple(sorted(key)) if isinstance(key, tuple) else key

            # 如果标准化后的元组未出现过，则将其添加到结果字典
            if normalized_key not in seen_keys:
                result_dict[key] = value
                seen_keys.add(normalized_key)

        return result_dict
def map_nested_list(nested_list, mapping_dict):
    result = []
    for item in nested_list:
        if isinstance(item, list): 
            result.append(tuple(map_nested_list(item, mapping_dict)))
        else:  
            result.append(mapping_dict.get(item, item))  
    return result

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

def get_jaccard_matrix_tensor_optimized(hyperedge_index):
    """
    优化版本：使用矩阵运算加速计算
    
    Args:
        hyperedge_index: torch.Tensor, shape [2, num_edges]
    
    Returns:
        edge_index: torch.Tensor, shape [2, num_nonzero_edges]
        edge_weights: torch.Tensor, shape [num_nonzero_edges]
    """
    device = hyperedge_index.device
    
    # 转置并获取唯一的超边ID
    hyperedge_index_t = hyperedge_index.t()
    unique_hyperedges = torch.unique(hyperedge_index_t[:, 1])
    n_hyperedges = len(unique_hyperedges)
    
    # 创建超边-节点关联矩阵
    max_nodes = hyperedge_index_t[:, 0].max() + 1
    hyperedge_node_matrix = torch.zeros(n_hyperedges, max_nodes, dtype=torch.float, device=device)
    
    for i, hyperedge_id in enumerate(unique_hyperedges):
        node_mask = hyperedge_index_t[:, 1] == hyperedge_id
        nodes_in_hyperedge = hyperedge_index_t[node_mask, 0]
        hyperedge_node_matrix[i, nodes_in_hyperedge] = 1.0
    
    # 使用矩阵运算计算 Jaccard 相似系数
    # 交集：A ∩ B = min(A, B) 的和
    intersection_matrix = torch.mm(hyperedge_node_matrix, hyperedge_node_matrix.t())
    
    # 并集：A ∪ B = A + B - A ∩ B
    hyperedge_sizes = hyperedge_node_matrix.sum(dim=1, keepdim=True)
    union_matrix = hyperedge_sizes + hyperedge_sizes.t() - intersection_matrix
    
    # Jaccard 相似系数：|A ∩ B| / |A ∪ B|
    jaccard_matrix = torch.where(union_matrix > 0, 
                                intersection_matrix / union_matrix, 
                                torch.zeros_like(intersection_matrix))
    
    # 获取非零元素
    rows, cols = torch.nonzero(jaccard_matrix, as_tuple=True)
    weights = jaccard_matrix[rows, cols]
    
    edge_index = torch.stack([rows, cols], dim=0)
    
    return edge_index, weights
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

def he_conj_enhanced(mol):
    """ get node index, hyperedge index and conjugation type of conjugated structure in a molecule

    Args:
        mol (RDKit MOL): input molecule

    Returns:
        tuple: node index, hyperedge index, num_he, e_type_idx, conj_feature_matrix, atom_conj_index
    """
    num_atom = mol.GetNumAtoms()
    reso = Chem.ResonanceMolSupplier(mol)
    num_he = reso.GetNumConjGrps()
    # print('num_he',num_he)
    # exit()
    n_idx, e_idx = [], []
    conj_types = []  # 存储每个共轭组的类型信息
    
    # 收集参与共轭的原子信息
    conj_atom_ids = []
    conj_group_ids = []
    
    # 获取基本的共轭信息
    for i in range(num_atom):
        _conj = reso.GetAtomConjGrpIdx(i)
        if _conj > -1 and _conj < num_he:
            n_idx.append(i)
            e_idx.append(_conj)
            conj_atom_ids.append(i)  # 记录参与共轭的原子id
            conj_group_ids.append(_conj)  # 记录对应的共轭组id
    # print()
    # 分析每个共轭组的类型
    for he_idx in range(num_he):
        conj_atoms = [n_idx[i] for i, e in enumerate(e_idx) if e == he_idx]
        conj_type = analyze_conjugation_type(mol, conj_atoms)
        conj_types.append(conj_type)
    # print('conj_atom_ids',conj_atom_ids)
    # print('conj_group_ids',conj_group_ids)
    # print('conj_types',conj_types)
    # exit()
    # 为每个参与共轭的原子分配对应的共轭类型
    e_type_idx = []
    for i, conj_group_idx in enumerate(e_idx):
        e_type_idx.append(conj_types[conj_group_idx])
    # print('e_type_idx',e_type_idx)
    # exit()
    # 构建特征矩阵
    # conj_feature_matrix = build_conjugation_feature_matrix(conj_types, num_he)
    
    # 构建二维atom_conj_index: [节点id, 共轭组id]
    if len(conj_atom_ids) > 0:
        atom_conj_index = torch.stack([
            torch.tensor(conj_atom_ids, dtype=torch.long),
            torch.tensor(conj_group_ids, dtype=torch.long)
        ])
    else:
        # 如果没有共轭原子，返回空的2x0张量
        atom_conj_index = torch.empty((2, 0), dtype=torch.long)
    # print('atom_conj_index',atom_conj_index)
    # exit()
    return n_idx, e_idx, num_he, e_type_idx, atom_conj_index

def analyze_conjugation_type(mol, conj_atoms):
    """分析共轭结构的类型
    
    Args:
        mol: RDKit分子对象
        conj_atoms: 参与共轭的原子索引列表
    
    Returns:
        int: 共轭类型索引
            0: 芳香环共轭 (Aromatic)
            1: 烯烃共轭 (Alkene conjugation)
            2: 羰基共轭 (Carbonyl conjugation) 
            3: 亚胺共轭 (Imine conjugation)
            4: 混合共轭 (Mixed conjugation)
    """
    if not conj_atoms:
        return 0
    
    # 统计共轭原子的特征
    aromatic_count = 0
    carbonyl_count = 0
    imine_count = 0
    double_bond_count = 0
    
    atoms = [mol.GetAtomWithIdx(idx) for idx in conj_atoms]
    
    # 分析原子类型
    for atom in atoms:
        if atom.GetIsAromatic():
            aromatic_count += 1
        
        # 检查羰基 (C=O)
        if atom.GetAtomicNum() == 6:  # 碳原子
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 8:  # 氧原子
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                    if bond and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        carbonyl_count += 1
        
        # 检查亚胺 (C=N)
        if atom.GetAtomicNum() == 6:  # 碳原子
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 7:  # 氮原子
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                    if bond and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        imine_count += 1
    
    # 检查双键数量
    for i, atom_idx1 in enumerate(conj_atoms):
        for atom_idx2 in conj_atoms[i+1:]:
            bond = mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
            if bond and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                double_bond_count += 1
    
    # 修正判断逻辑：优先检查特殊官能团，最后检查芳香性
    total_atoms = len(conj_atoms)
    
    # 羰基共轭：含有羰基（优先级最高）
    if carbonyl_count > 0:
        return 2
    
    # 亚胺共轭：含有亚胺
    elif imine_count > 0:
        return 3
    
    # 芳香环共轭：所有原子都是芳香性的
    elif aromatic_count == total_atoms and aromatic_count > 0:
        return 0
    
    # 烯烃共轭：主要是C=C双键
    elif double_bond_count > 0:
        return 1
    
    # 混合共轭或其他情况
    else:
        return 4

def build_conjugation_feature_matrix(conj_types, num_he):
    """构建共轭特征矩阵
    
    Args:
        conj_types: 共轭类型列表
        num_he: 共轭组数量
    
    Returns:
        np.ndarray: 特征矩阵 (num_he, feature_dim)
    """
    # 定义特征维度：基础类型(5) + 化学特征(5) = 10维
    feature_dim = 10
    feature_matrix = np.zeros((num_he, feature_dim))
    
    # 定义五种共轭类型的基础特征模板
    conjugation_templates = {
        0: {'stability': 0.9, 'reactivity': 0.3, 'planarity': 0.95},  # 芳香环共轭
        1: {'stability': 0.6, 'reactivity': 0.7, 'planarity': 0.8},   # 烯烃共轭
        2: {'stability': 0.7, 'reactivity': 0.8, 'planarity': 0.85},  # 羰基共轭
        3: {'stability': 0.5, 'reactivity': 0.9, 'planarity': 0.75},  # 亚胺共轭
        4: {'stability': 0.65, 'reactivity': 0.6, 'planarity': 0.7}   # 混合共轭
    }
    # print(conj_types)
    # exit()
    for he_idx, conj_type in enumerate(conj_types):
        # 基础类型特征 (one-hot编码) - 维度 0-4
        if conj_type < 5:
            feature_matrix[he_idx, conj_type] = 1.0
        
        # 获取模板特征
        template = conjugation_templates.get(conj_type, conjugation_templates[4])
        
        # 化学特征 - 维度 5-9
        feature_matrix[he_idx, 5] = template['stability']   # 稳定性
        feature_matrix[he_idx, 6] = template['reactivity']  # 反应活性
        feature_matrix[he_idx, 7] = template['planarity']   # 平面性
        feature_matrix[he_idx, 8] = 1.0  # 占位符特征1
        feature_matrix[he_idx, 9] = 1.0  # 占位符特征2
    
    return feature_matrix

def conj(mol,num_node_bond_angle_type):
    ## Conjugation processing
    same_conj_dict = {}
    group_conj_dict={}
    conj_feature = None
    atom_conj_index = None
    
    if mol is not None:
        try:
            n_idx, e_idx, num_he, e_type_idx, atom_conj_index = he_conj_enhanced(mol)
            if num_he > 0 and len(n_idx) > 0:
                conj_groups = {}
                for i, conj_group_idx in enumerate(e_idx):
                    if conj_group_idx not in conj_groups:
                        conj_groups[conj_group_idx] = []
                    conj_groups[conj_group_idx].append(n_idx[i])
                
                
                from collections import defaultdict
                for conj_group_idx, atoms in conj_groups.items():
                    conj_atom_set = set(atoms)
                    type_buckets = defaultdict(set)  

                    
                    for a in atoms:
                        atom_types = classify_atom_conj_type(mol, a, conj_atom_set)  # 返回可能的多种类型
                        for t in atom_types:
                            related_atoms = get_conjugation_related_atoms(mol, a, t, conj_atom_set)
                            
                            type_buckets[t].update(x for x in related_atoms if x in conj_atom_set)
                    
                    
                    for t, atom_set in type_buckets.items():
                        if atom_set:
                            conj_key = f"conj_type_{t}_group_{conj_group_idx}"
                            group_conj_dict[conj_key] = sorted(list(atom_set))
                            conj_type_key=f"conj_type_{t}"
                            if conj_type_key not in same_conj_dict:
                                same_conj_dict[conj_type_key] = sorted(list(atom_set))
                            else:
                                same_conj_dict[conj_type_key].extend(sorted(list(atom_set)))
                conj_types=[int(conj_name.split('_')[2]) for conj_name in group_conj_dict.keys()]   
                conj_feature_matrix = build_conjugation_feature_matrix(conj_types, len(conj_types))
                # 转换特征矩阵为tensor
                conj_feature = torch.from_numpy(conj_feature_matrix.astype(np.float32))
            else:
                # 没有共轭结构时，提供默认的单行零特征
                conj_feature = torch.zeros((1, 10), dtype=torch.float32)
                
        except Exception as e:
            print(f"Warning: Failed to process conjugation information: {e}")
            # 如果处理共轭信息失败，使用默认值
            group_conj_dict = {}
            conj_feature = torch.zeros((1, 10), dtype=torch.float32)
            atom_conj_index = torch.empty((2, 0), dtype=torch.long)
    else:
        # 如果没有提供mol对象，使用默认值
        same_conj_dict = {}
        conj_feature = torch.zeros((1, 10), dtype=torch.float32)
        atom_conj_index = torch.empty((2, 0), dtype=torch.long)


    group_type=[]
    try :
        
        conj_type_index=torch.zeros((2,len(group_conj_dict.keys())),dtype=torch.long)

        
        for conj in group_conj_dict.keys():
            conj_type=int(conj.split('_')[2])
            group_type.append(conj_type)
        remap_group_type={conj_type:id for id,conj_type in enumerate(set(group_type))}
        # gropu_type=set(group_type)
        
        for id,conj in enumerate(group_conj_dict.keys()):
            conj_type=int(conj.split('_')[2])
            conj_type_index[0,id]=id
            conj_type_index[1,id]=remap_group_type[conj_type]+num_node_bond_angle_type
    except:
        conj_type_index=torch.empty((2, 0), dtype=torch.long)
    # print('group_type',group_type)
    # print(same_conj_dict)
    # print(group_conj_dict)
    # print(conj_type_index)
    # print(conj_feature)
    # exit()
    conj_type_label=torch.tensor(group_type,dtype=torch.long)
    return same_conj_dict, conj_feature, atom_conj_index,conj_type_index,conj_type_label

def classify_atom_conj_type(mol, atom_idx, conj_atom_set=None):
    """
    判别该原子在共轭中的所有类型（多选）：
      0: 芳香（在芳香组件中）
      1: 烯烃（位于 C=C 连通段上）
      2: 羰基（参与 C=O 的 C 或 O）
      3: 亚胺（参与 C=N 的 C 或 N）
    仅返回确有共轭证据的类型；如果没有任何类型，则返回空列表（便于“只返回有共轭的”）。
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    types = []

    # 0 芳香：原子本身芳香
    if atom.GetIsAromatic():
        types.append(0)

    # 2 羰基：C=O 双键的 C 或 O
    Z = atom.GetAtomicNum()
    if Z in (6, 8):  # C 或 O
        for nbr in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx())
            if not bond:
                continue
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                nZ = nbr.GetAtomicNum()
                if (Z == 6 and nZ == 8) or (Z == 8 and nZ == 6):
                    types.append(2)
                    break

    # 3 亚胺：C=N 双键的 C 或 N
    if Z in (6, 7):  # C 或 N
        for nbr in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx())
            if not bond:
                continue
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                nZ = nbr.GetAtomicNum()
                if (Z == 6 and nZ == 7) or (Z == 7 and nZ == 6):
                    types.append(3)
                    break

    # 1 烯烃：C=C 双键连通段（限定在当前共轭组内）
    if atom.GetAtomicNum() == 6:
        is_vinyl = False
        for nbr in atom.GetNeighbors():
            if conj_atom_set is not None and nbr.GetIdx() not in conj_atom_set:
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx())
            if bond and nbr.GetAtomicNum() == 6 and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                is_vinyl = True
                break
        if is_vinyl:
            types.append(1)

    # 不返回“其他/混合”；若没有命中任何类型，返回空列表
    return types

def get_conjugation_related_atoms(mol, atom_idx, conj_type, conj_atom_set):
    """
    给定一个原子及其某个共轭类型，返回该类型下与其结构相关的所有原子集合。
    - 0 芳香：返回与该原子相连的芳香组件（限制在 conj_atom_set 内）
    - 1 烯烃：沿 C=C 双键连通的碳原子集合（限制在 conj_atom_set 内）
    - 2 羰基：返回 C=O 双键的碳和氧
    - 3 亚胺：返回 C=N 双键的碳和氮
    """
    related = set()
    atom = mol.GetAtomWithIdx(atom_idx)

    if conj_type == 0:
        # 芳香组件连通搜索
        if not atom.GetIsAromatic():
            return related
        stack, visited = [atom_idx], set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            cur_atom = mol.GetAtomWithIdx(cur)
            if cur_atom.GetIsAromatic() and (conj_atom_set is None or cur in conj_atom_set):
                related.add(cur)
                for nbr in cur_atom.GetNeighbors():
                    nid = nbr.GetIdx()
                    if nid not in visited and nbr.GetIsAromatic() and (conj_atom_set is None or nid in conj_atom_set):
                        stack.append(nid)
        return related

    if conj_type == 1:
        # 烯烃连通搜索：只沿 C=C 双键在碳节点上扩展
        if atom.GetAtomicNum() != 6:
            return related
        stack, visited = [atom_idx], set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            cur_atom = mol.GetAtomWithIdx(cur)
            if cur_atom.GetAtomicNum() == 6 and (conj_atom_set is None or cur in conj_atom_set):
                related.add(cur)
                for nbr in cur_atom.GetNeighbors():
                    nid = nbr.GetIdx()
                    if conj_atom_set is not None and nid not in conj_atom_set:
                        continue
                    bond = mol.GetBondBetweenAtoms(cur, nid)
                    if bond and mol.GetAtomWithIdx(nid).GetAtomicNum() == 6 and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        if nid not in visited:
                            stack.append(nid)
        return related

    if conj_type == 2:
        # 羰基：C=O 双键的 C 与 O
        Z = atom.GetAtomicNum()
        if Z in (6, 8):
            for nbr in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx())
                if bond and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    nZ = nbr.GetAtomicNum()
                    if (Z == 6 and nZ == 8) or (Z == 8 and nZ == 6):
                        related.add(atom_idx)
                        related.add(nbr.GetIdx())
                        break
        return related

    if conj_type == 3:
        # 亚胺：C=N 双键的 C 与 N
        Z = atom.GetAtomicNum()
        if Z in (6, 7):
            for nbr in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx())
                if bond and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    nZ = nbr.GetAtomicNum()
                    if (Z == 6 and nZ == 7) or (Z == 7 and nZ == 6):
                        related.add(atom_idx)
                        related.add(nbr.GetIdx())
                        break
        return related

    return related
def get_rw_landing_probs_and_edge_features(edge_index, ksteps,edge_weight=None,
                                           num_nodes=None, space_dim=0):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    # Transition matrix P = D^-1 * A
    adj = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes).squeeze(0)  # (Num nodes) x (Num nodes)
    P = torch.diag(deg_inv) @ adj  # Transition matrix
    rws = []
    edge_features = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            edge_features.append(Pk[source, dest] * (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            Pk = P.matrix_power(k)
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            edge_features.append(Pk[source, dest] * (k ** (space_dim / 2)))
    rw_landing = torch.stack(rws, dim=1)  # (Num nodes) x (K steps)
    edge_features = torch.stack(edge_features, dim=1)  # (Num edges) x (K steps)
    return rw_landing, edge_features
# def get_hyperedge_type(hyperedges):
#     atom_type=[]
#     bond_type=[]
#     angle_type=[]
#     conj_type=[]
#     num_atom_type=0
#     num_bond_type=0
#     num_angle_type=0
#     num_conj_type=0
#     for edge_type,nodes in hyperedges.items():
#         if edge_type is int:
#             num_atom_type+=1
#             atom_type.append(nodes)
#         if edge_type is tuple:
#             num_bond_type+=1    
#             bond_type.append(nodes)
#         if edge_type is str:
#             if 'conj' not in edge_type:
#                 num_angle_type+=1
#                 angle_type.append(nodes)
#             else:
#                 num_conj_type+=1
#                 conj_type.append(nodes)
    
        
    
#     return hyperbond_type_index
def generate_mol_graph(base_graph, mol_3d_info, mol,multi_task=False,atom_poses=None):  
    data = mol_3d_info
    
    # ## Atom type
    # atom_number_list = data["atomic_num"]
    # id_atom_num = {}
    # for id, atom_num in enumerate(atom_number_list):
    #     id_atom_num[id] = atom_num
    # same_atom_dict = group_keys_by_value(id_atom_num)

    # ## Bond link
    # atom_atom_link = data['edges']
    # mapped_list = [[id_atom_num.get(item, item) for item in row] for row in atom_atom_link]
    # atom_atom_index_dict = {}
    # for atom_atom_index, atom_atom_type in enumerate(mapped_list):
    #     atom_atom_index_dict[atom_atom_index] = tuple(atom_atom_type)
    # dict1 = group_keys_by_value(atom_atom_index_dict)
    # result_dict = map_indices_to_values(dict1, atom_atom_link.tolist())
    # same_bond_dict = remove_duplicate_tuple_keys(result_dict)

    # ## angle link
    # bond_bond_link = data['BondAngleGraph_edges']
    # atom_atom_atom_link = []
    # edges_to_nodes = {i: (atom_atom_link[i, 0], atom_atom_link[i, 1]) for i in range(atom_atom_link.shape[0])}
    # for bond1, bond2 in bond_bond_link:
    #     node1_bond1, node2_bond1 = edges_to_nodes[bond1]
    #     node1_bond2, node2_bond2 = edges_to_nodes[bond2]
    #     atom_atom_atom_link.append([(node1_bond1, node2_bond1), (node1_bond2, node2_bond2)])
    # atom_atom_atom_link = np.array(atom_atom_atom_link, dtype=object)
    # atom_atom_atom_link = atom_atom_atom_link.reshape(-1, 4)
    # unique_elements_list = []
    # for row in atom_atom_atom_link:
    #     unique_row = list(set(row))
    #     if len(unique_row) == 3:
    #         if unique_row not in unique_elements_list:
    #             unique_elements_list.append(unique_row)
    # atom_atom_atom_link = unique_elements_list
    # list3 = map_nested_list(atom_atom_atom_link, id_atom_num)
    # angle_index_dict = {}
    # for atom_atom_index, atom_atom_type in enumerate(list3):
    #     angle_index_dict[atom_atom_index] = tuple(atom_atom_type)
    # angle_dict = group_keys_by_value(angle_index_dict)
    # same_angle_dict = map_indices_to_values(angle_dict, atom_atom_atom_link)
    same_atom_dict,same_bond_dict,same_angle_dict, atom_type_index,bond_type_index,angle_type_index,\
    num_node_bond_angle_type,valid_angles_in_all,atom_type_label,bond_type_label,angle_type_label = get_edges(data)
    same_conj_dict, conj_feature, atom_conj_index,conj_type_index,conj_type_label = conj(mol,num_node_bond_angle_type)
    # print('atom_type_label:',atom_type_label)
    # print('bond_type_label:',bond_type_label)
    # print('angle_type_label:',angle_type_label)
    # print('conj_type_label:',conj_type_label)
    # print('atom_type_index',atom_type_index)
    # print('bond_type_index',bond_type_index)
    # print('angle_type_index',angle_type_index)
    # print('conj_type_index',conj_type_index)
    # exit()
    ## hypergraph_matrix
    hyper_graph_dict = {}
    hyper_graph_dict.update(same_atom_dict)
    hyper_graph_dict.update(same_bond_dict)
    hyper_graph_dict.update(same_angle_dict)
    hyper_graph_dict.update(same_conj_dict)  # 添加共轭字典
    
    hyperedges = hyper_graph_dict
    # print(hyperedges)

    # exit()
    all_nodes = sorted(set([node for nodes in hyperedges.values() for node in nodes]))
    num_nodes = len(all_nodes)
    num_hyperedges = len(hyperedges)
    hyper_graph_matrix = np.zeros((num_nodes, num_hyperedges), dtype=int)
    node_to_row = {node: idx for idx, node in enumerate(all_nodes)}
    for col_idx, (edge, nodes) in enumerate(hyperedges.items()):
        for node in nodes:
            row_idx = node_to_row[node]
            hyper_graph_matrix[row_idx, col_idx] = 1 
    hyper_graph_matrix = torch.tensor(hyper_graph_matrix)

    ## create_jaccard_matrix
    hypergraph_edge_index = convert_hypergraph_matrix_to_edge_index(hyper_graph_matrix)
    # jaccard_matrix = get_jaccard_matrix(hypergraph_edge_index)
    # print('hyper_graph_index:',hypergraph_edge_index)
    # exit()
    ## angle_feature
    # descriptor = mordred
    angle_feature = torch.from_numpy(data['bond_angle'].astype(np.float32))
    
    # logp = torch.ones([angle_feature.shape[0]]) * descriptor[1766]
    angle_feature = angle_feature[valid_angles_in_all].reshape(-1, 1)
    # print(angle_feature.shape)
    
    # angle_feature = torch.cat([angle_feature.reshape(-1, 1), logp.reshape(-1, 1)], dim=1)

    ## bond_angle
    bond_bond_link = data['BondAngleGraph_edges']
    # print('angle:',bond_bond_link)
    # exit()
    num_bond = np.max(bond_bond_link)
    num_angle = bond_bond_link.shape[0]
    num_bond_num_angle = np.zeros((num_bond+1, num_angle), dtype=int)
    for i in range(bond_bond_link.shape[0]):
        k1, k2 = bond_bond_link[i]
        num_bond_num_angle[k1, i] = 1
        num_bond_num_angle[k2, i] = 1
    num_bond_num_angle = torch.tensor(num_bond_num_angle)

    ## atom_bond
    atom_atom_link = data['edges']
    num_atom = np.max(atom_atom_link)
    num_bond = atom_atom_link.shape[0]
    num_atom_num_bond = np.zeros((num_atom+1, num_bond), dtype=int)
    for i in range(atom_atom_link.shape[0]):
        k1, k2 = atom_atom_link[i]
        num_atom_num_bond[k1, i] = 1
        num_atom_num_bond[k2, i] = 1
    num_atom_num_bond = torch.tensor(num_atom_num_bond)
    atom_atom_link = torch.tensor(atom_atom_link, dtype=torch.long).transpose(0, 1)
    

    atom_feature = base_graph.x
    node_pos_emb=get_rw_landing_probs_and_edge_features(atom_atom_link, range(1,atom_feature.shape[1]+1))[0]
    atom_poses=torch.tensor(atom_poses, dtype=torch.float32)
    bond_feature = base_graph.edge_attr
    label = base_graph.y
    
    smiles = getattr(base_graph, 'smiles', None)
    atom_bond_index = convert_hypergraph_matrix_to_edge_index(num_atom_num_bond)
    bond_angle_index = convert_hypergraph_matrix_to_edge_index(num_bond_num_angle)
    atom_angle_index = convert_hypergraph_matrix_to_edge_index(torch.matmul(num_atom_num_bond, num_bond_num_angle))
    
    node_batch_idx =torch.zeros((len(set(hypergraph_edge_index[0].tolist()))), dtype=torch.long)
    hyperedge_batch_idx =torch.zeros((len(set(hypergraph_edge_index[1].tolist()))), dtype=torch.long)
    atom_type_batch_index = torch.zeros((len(set(atom_type_index[1].tolist()))), dtype=torch.long)
    bond_type_batch_index = torch.zeros((len(set(bond_type_index[1].tolist()))), dtype=torch.long)
    angle_type_batch_index = torch.zeros((len(set(angle_type_index[1].tolist()))), dtype=torch.long)
    conj_type_batch_index = torch.zeros((len(set(conj_type_index[1].tolist()))), dtype=torch.long)
    
    # print('atom_type_batch_index',atom_type_batch_index.shape)
    # print(atom_type_batch_index)
    # 细粒度超边类型编码：为每个超边键分配唯一 id，并以 Tensor 形式存储
    # 构建每个超边键到 id 的词表（仅用于构造 tensor，不随图外共享）
    # _type_vocab = {k: i for i, k in enumerate(hyperedges.keys())}
    # hyperedge_type_id = torch.tensor([_type_vocab[k] for k in hyperedges.keys()], dtype=torch.long)

    # 节点类型：原子序数张量
    node_type = torch.tensor(data['atomic_num'], dtype=torch.long)
   
    if not multi_task:
        
        graph = HData(
                x=atom_feature,
                batch=node_batch_idx,
                coord=atom_poses,
                node_pos_emb=node_pos_emb,
                edge_index=hypergraph_edge_index,
                hyperedge_batch_index=hyperedge_batch_idx,
                bond_feature=bond_feature,
                angle_feature=angle_feature,
                conj_feature=conj_feature,  
                atom_conj_index=atom_conj_index,  
                atom_atom_index=atom_atom_link,
                atom_bond_index=atom_bond_index,
                bond_angle_index=bond_angle_index,
                atom_type_batch_index=atom_type_batch_index,
                bond_type_batch_index=bond_type_batch_index,
                angle_type_batch_index=angle_type_batch_index,
                conj_type_batch_index=conj_type_batch_index,
                y=label,
                # jaccard_index=jaccard_matrix[0],
                # jaccard_weight=jaccard_matrix[1],
                hyperedge_type=hyperedges,
                node_type=node_type,
                atom_angle_index=atom_angle_index,
                smiles=smiles  ,
                num_hyperedges=num_hyperedges,
                atom_type_index=atom_type_index,
                bond_type_index=bond_type_index,
                angle_type_index=angle_type_index,
                conj_type_index=conj_type_index,
                num_bonds=bond_feature.shape[0],
                num_angles=angle_feature.shape[0],
                num_conj=conj_feature.shape[0],
                num_atom_types=len(set(atom_type_index[1].tolist())),
                num_bond_types=len(set(bond_type_index[1].tolist())),
                num_angle_types=len(set(angle_type_index[1].tolist())),
                num_conj_types=len(set(conj_type_index[1].tolist())),
                )
        return graph    
            
            
    else:
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.int8)  # 使用int8节省内存
        DataStructs.ConvertToNumpyArray(fp, arr)
        morgan=torch.tensor(arr).reshape(1,-1)
        logp=torch.tensor(logp,dtype=torch.float).reshape(1,-1)
        tpsa=torch.tensor(tpsa,dtype=torch.float).reshape(1,-1)
        
        graph = HData(
                x=atom_feature,
                batch=node_batch_idx,
                coord=atom_poses,
                node_pos_emb=node_pos_emb,
                edge_index=hypergraph_edge_index,
                hyperedge_batch_index=hyperedge_batch_idx,
                bond_feature=bond_feature,
                angle_feature=angle_feature,
                conj_feature=conj_feature,  
                atom_conj_index=atom_conj_index,  
                atom_atom_index=atom_atom_link,
                atom_bond_index=atom_bond_index,
                bond_angle_index=bond_angle_index,
                atom_type_batch_index=atom_type_batch_index,
                bond_type_batch_index=bond_type_batch_index,
                angle_type_batch_index=angle_type_batch_index,
                conj_type_batch_index=conj_type_batch_index,
                # jaccard_index=jaccard_matrix[0],
                # jaccard_weight=jaccard_matrix[1],
                smiles=smiles,

                logp=logp,
                tpsa=tpsa,
                morgan=morgan,
                conj_type_label=conj_type_label,

                hyperedge_type=hyperedges,
                node_type=node_type,
                atom_angle_index=atom_angle_index,
                # smiles=smiles,
                num_hyperedges=num_hyperedges,
                atom_type_index=atom_type_index,
                bond_type_index=bond_type_index,
                angle_type_index=angle_type_index,
                conj_type_index=conj_type_index,
                num_bonds=bond_feature.shape[0],
                num_angles=angle_feature.shape[0],
                num_conj=conj_feature.shape[0],
                num_atom_types=len(set(atom_type_index[1].tolist())),
                num_bond_types=len(set(bond_type_index[1].tolist())),
                num_angle_types=len(set(angle_type_index[1].tolist())),
                num_conj_types=len(set(conj_type_index[1].tolist())))
            
    return graph,atom_type_label,bond_type_label,angle_type_label
def show_example(mol_3d_info,mol):
    same_atom_dict,same_bond_dict,same_angle_dict = get_edges(mol_3d_info)
    same_conj_dict, conj_feature, atom_conj_index = conj(mol)
    # print(mol_3d_info)
    print("=== 原子类型分组 ===")
    for atom_type, atoms in same_atom_dict.items():
        element_name = get_atom_type_name(atom_type)
        print(f"{element_name} (原子序数{atom_type}): {atoms}")
    print("\n=== 详细键类型分组（区分单键双键）===")
    for bond_key, bonds in same_bond_dict.items():
        atom1_type, atom2_type, bond_type = bond_key
        atom1_name = get_atom_type_name(atom1_type)
        atom2_name = get_atom_type_name(atom2_type)
        bond_name = get_bond_type_name(bond_type)
        print(f"{atom1_name}{bond_name.replace('SINGLE', '-').replace('DOUBLE', '=').replace('TRIPLE', '≡')}{atom2_name}: {bonds}")
    print("\n=== 键角形式分组 ===")
    for pattern, angles in same_angle_dict.items():
        print(f"{pattern}: {angles}")
    print("\n=== 共轭组分组 ===")
    for conj_key, conj_atoms in same_conj_dict.items():
        print(f"{conj_key}: {conj_atoms}")

if __name__ == '__main__':
    smiles='CC(=O)OC1=CC=CC=C1C(=O)O'
    mol = Chem.MolFromSmiles(smiles)
    new_mol = Chem.AddHs(mol)
    res = AllChem.EmbedMultipleConfs(new_mol)
    res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
    new_mol = Chem.RemoveHs(new_mol)

    mol_3d_info = mol_to_geognn_graph_data_MMFF3d(new_mol)
    show_example(mol_3d_info,mol)
    
