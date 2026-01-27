from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import rdchem
from compound_tools import mol_to_geognn_graph_data_MMFF3d,mord
from collections import defaultdict
import itertools
import numpy as np  # 添加缺少的导入
import torch
# 阿司匹林分子
smile='CC(=O)OC1=CC=CC=C1C(=O)O'

mol = AllChem.MolFromSmiles(smile)
new_mol = Chem.AddHs(mol)
res = AllChem.EmbedMultipleConfs(new_mol)
res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
new_mol = Chem.RemoveHs(new_mol)

mol_3d_info = mol_to_geognn_graph_data_MMFF3d(new_mol)
mordred=mord(new_mol)

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

# 添加缺少的map_nested_list函数
def map_nested_list(nested_list, mapping_dict):
    result = []
    for item in nested_list:
        if isinstance(item, list): 
            result.append(tuple(map_nested_list(item, mapping_dict)))
        else:  
            result.append(mapping_dict.get(item, item))  
    return result

# 新增：根据键类型和原子类型进行分组的函数
def create_bond_type_groups(mol_3d_info,num_atom_type=0):
    """根据键类型和连接的原子类型创建键分组"""
    data = mol_3d_info
    atom_number_list = data["atomic_num"]
    bond_type_list = data["bond_type"]  # 键类型信息
    atom_atom_link = data['edges']
    
    # 创建原子索引到原子类型的映射
    id_atom_num = {}
    for id, atom_num in enumerate(atom_number_list):
        id_atom_num[id] = atom_num
    
    # 创建键的详细分类：(原子1类型, 原子2类型, 键类型)
    bond_detailed_dict = {}
    for bond_index, (atom1_idx, atom2_idx) in enumerate(atom_atom_link):
        atom1_type = id_atom_num[atom1_idx]
        atom2_type = id_atom_num[atom2_idx]
        bond_type = bond_type_list[bond_index]
        
        # 标准化原子对顺序（较小的原子类型在前）
        if atom1_type > atom2_type:
            atom1_type, atom2_type = atom2_type, atom1_type
            
        # 创建键的详细标识：(原子类型1, 原子类型2, 键类型)
        bond_key = (atom1_type, atom2_type, bond_type)
        bond_detailed_dict[bond_index] = bond_key
    # print(bond_detailed_dict)
    
    # 按详细键类型分组
    detailed_bond_groups = group_keys_by_value(bond_detailed_dict)
    # print('-------------------------------------')
    
    # 将索引映射回实际的边
    same_bond_detailed_dict = map_indices_to_values(detailed_bond_groups, atom_atom_link.tolist())
    # print(same_bond_detailed_dict)
    same_bond_detailed_dict = remove_duplicate_tuple_keys(same_bond_detailed_dict)
    # print('-------------------------------------')
    # print(same_bond_detailed_dict)
    rename_edge_type={}
    for edge_type_id,key in enumerate(same_bond_detailed_dict.keys()):
        rename_edge_type[key]=edge_type_id+num_atom_type
    edge_type_index=torch.zeros((2,len(bond_detailed_dict)),dtype=torch.long)
    for edge_id,edge_type in bond_detailed_dict.items():
        edge_type_index[0,edge_id]=edge_id
        edge_type_index[1,edge_id]=rename_edge_type[edge_type]
    # print(edge_type_index.shape,edge_type_index)    
        
    bond_type_label=list(bond_detailed_dict.values())
    # exit()
    return same_bond_detailed_dict,edge_type_index,max(edge_type_index[1])+1,bond_type_label

# 获取键类型的含义
def get_bond_type_name(bond_type_value):
    """将键类型数值转换为可读名称"""
    # RDKit中的键类型映射（减1是因为代码中+1了）
    bond_type_names = {
        1: "SINGLE",    # 单键
        2: "DOUBLE",    # 双键  
        3: "TRIPLE",    # 三键
        4: "AROMATIC"   # 芳香键
    }
    return bond_type_names.get(bond_type_value, f"UNKNOWN({bond_type_value})")

# 获取原子类型名称
def get_atom_type_name(atomic_num):
    """将原子序数转换为元素符号"""
    element_names = {
        1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"
    }
    return element_names.get(atomic_num, f"Element({atomic_num})")



    
def create_bond_angle_pattern_groups(mol_3d_info,base_type_id=0):
    """根据键角形式（如 O-C=C）进行分组"""
    data = mol_3d_info
    atom_number_list = data["atomic_num"]
    bond_type_list = data["bond_type"]  # 键类型信息
    atom_atom_link = data['edges']
    bond_bond_link = data['BondAngleGraph_edges']
    
    # 创建原子索引到原子类型的映射
    id_atom_num = {}
    for id, atom_num in enumerate(atom_number_list):
        id_atom_num[id] = atom_num
    
    # 创建键索引到键类型的映射
    bond_type_dict = {}
    for bond_idx, bond_type in enumerate(bond_type_list):
        bond_type_dict[bond_idx] = bond_type
    # print('bond_type_dict',bond_type_dict)
    
    # 处理键角形式
    edges_to_nodes = {i: (atom_atom_link[i, 0], atom_atom_link[i, 1]) for i in range(atom_atom_link.shape[0])}
    
    angle_pattern_dict = {}
    valid_angles = []
    valid_angles_in_all=[]
    valid_angle_idx = 0  # 用于跟踪有效角度的索引
    # print('bond_bond_link',len(bond_bond_link))
    for angle_idx, (bond1, bond2) in enumerate(bond_bond_link):
        node1_bond1, node2_bond1 = edges_to_nodes[bond1]
        node1_bond2, node2_bond2 = edges_to_nodes[bond2]
        
        # 找到共同的原子（键角的顶点）
        common_atoms = set([node1_bond1, node2_bond1]) & set([node1_bond2, node2_bond2])
        if len(common_atoms) == 1:
            center_atom = list(common_atoms)[0]
            
            # 找到键角的三个原子
            atoms_in_angle = list(set([node1_bond1, node2_bond1, node1_bond2, node2_bond2]))
            if len(atoms_in_angle) == 3:
                # 获取中心原子类型
                center_atom_type = id_atom_num[center_atom]
                center_atom_name = get_atom_type_name(center_atom_type)
                
                # 获取两个键的类型
                bond1_type = bond_type_dict[bond1]
                bond2_type = bond_type_dict[bond2]
                
                # 找到另外两个原子
                if center_atom == node1_bond1:
                    atom1 = node2_bond1
                else:
                    atom1 = node1_bond1
                    
                if center_atom == node1_bond2:
                    atom2 = node2_bond2
                else:
                    atom2 = node1_bond2
                
                # 获取原子类型和名称
                atom1_type = id_atom_num[atom1]
                atom2_type = id_atom_num[atom2]
                atom1_name = get_atom_type_name(atom1_type)
                atom2_name = get_atom_type_name(atom2_type)
                
                # 获取键符号
                bond1_symbol = get_bond_type_name(bond1_type).replace('SINGLE', '-').replace('DOUBLE', '=').replace('TRIPLE', '≡').replace('AROMATIC', ':')
                bond2_symbol = get_bond_type_name(bond2_type).replace('SINGLE', '-').replace('DOUBLE', '=').replace('TRIPLE', '≡').replace('AROMATIC', ':')
                
                # 创建键角形式：原子1-键1-中心原子-键2-原子2
                pattern1 = f"{atom1_name}{bond1_symbol}{center_atom_name}{bond2_symbol}{atom2_name}"
                pattern2 = f"{atom2_name}{bond2_symbol}{center_atom_name}{bond1_symbol}{atom1_name}"
                
                # 选择字典序较小的作为标准形式
                angle_pattern = pattern1 if pattern1 <= pattern2 else pattern2
                
                # 使用有效角度的索引而不是原始的angle_idx
                angle_pattern_dict[valid_angle_idx] = angle_pattern
                valid_angles.append(atoms_in_angle)
                valid_angle_idx += 1
                valid_angles_in_all.append(angle_idx)
    # print('valid_angles_in_all',len(valid_angles_in_all))
    # print('angle_pattern_dict',angle_pattern_dict)
    # 按键角形式分组
    angle_type_label=angle_pattern_dict.values()
    pattern_groups = group_keys_by_value(angle_pattern_dict)
    # print('pattern_groups',pattern_groups)
    # 将索引映射回实际的角度
    same_angle_pattern_dict = map_indices_to_values(pattern_groups, valid_angles)
    # print('same_angle_pattern_dict',same_angle_pattern_dict)
    # exit()
    rename_angle_type={}
    for angle_type_id,key in enumerate(same_angle_pattern_dict.keys()):
        rename_angle_type[key]=angle_type_id+base_type_id
    angle_type_index=torch.zeros((2,len(angle_pattern_dict)),dtype=torch.long)
    for angle_id,angle_type in angle_pattern_dict.items():
        angle_type_index[0,angle_id]=angle_id
        angle_type_index[1,angle_id]=rename_angle_type[angle_type]
    # print(angle_type_index)
    num_angle_type=max(angle_type_index[1])+1
    # print('angle_type_index',angle_type_index)
    return same_angle_pattern_dict,angle_type_index,num_angle_type,valid_angles_in_all,angle_type_label

def get_edges(mol_3d_info,show_detail=False):
# 执行分析
    data = mol_3d_info
    # print(data[0])
    ## 原子类型分组（原有功能）
    atom_number_list = data["atomic_num"]
    id_atom_num = {}
    for id, atom_num in enumerate(atom_number_list):
        id_atom_num[id] = atom_num
    # print(id_atom_num)
    same_atom_dict = group_keys_by_value(id_atom_num)
    # print(same_atom_dict)
    rename_node_type={}
    for node_type_id,key in enumerate(same_atom_dict.keys()):
        rename_node_type[key]=node_type_id
    node_type_index=torch.zeros((2,len(id_atom_num)),dtype=torch.long)
    for node_id,node_type in id_atom_num.items():
        node_type_index[0,node_id]=node_id
        node_type_index[1,node_id]=rename_node_type[node_type]
    # print(node_type_index)
    num_node_type=max(node_type_index[1])+1
    # exit()

    ## 新增：详细的键类型分组（区分单键和双键）
    same_bond_detailed_dict,edge_type_index,base_type_id,bond_type_label = create_bond_type_groups(mol_3d_info,num_node_type)

    ## 新增：键角形式分组（如 O-C=C）
    same_angle_pattern_dict,angle_type_index,num_node_bond_angle_type,valid_angles_in_all,angle_type_label = create_bond_angle_pattern_groups(mol_3d_info,base_type_id)
    
    # print('node_type_index',node_type_index)
    # print('edge_type_index',edge_type_index)
    # print('angle_type_index',angle_type_index)
    atom_type_label=mol_3d_info["atomic_num"].tolist() 
    # bond_type_label=mol_3d_info["bond_type"] 
    if show_detail:
        print("=== 原子类型分组 ===")
        for atom_type, atoms in same_atom_dict.items():
            element_name = get_atom_type_name(atom_type)
            print(f"{element_name} (原子序数{atom_type}): {atoms}")
        print("\n=== 详细键类型分组（区分单键双键）===")
        for bond_key, bonds in same_bond_detailed_dict.items():
            atom1_type, atom2_type, bond_type = bond_key
            atom1_name = get_atom_type_name(atom1_type)
            atom2_name = get_atom_type_name(atom2_type)
            bond_name = get_bond_type_name(bond_type)
            print(f"{atom1_name}{bond_name.replace('SINGLE', '-').replace('DOUBLE', '=').replace('TRIPLE', '≡')}{atom2_name}: {bonds}")
        print("\n=== 键角形式分组 ===")
        for pattern, angles in same_angle_pattern_dict.items():
            print(f"{pattern}: {angles}")
    return same_atom_dict,same_bond_detailed_dict,same_angle_pattern_dict, node_type_index,edge_type_index,angle_type_index,\
    num_node_bond_angle_type,valid_angles_in_all,atom_type_label,bond_type_label,angle_type_label
if __name__ == '__main__':
    smiles='CC(=O)OC1=CC=CC=C1C(=O)O'
    mol = Chem.MolFromSmiles(smile)
    new_mol = Chem.AddHs(mol)
    res = AllChem.EmbedMultipleConfs(new_mol)
    res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
    new_mol = Chem.RemoveHs(new_mol)

    mol_3d_info = mol_to_geognn_graph_data_MMFF3d(new_mol)
    graph=get_edges(mol_3d_info,show_detail=True)
