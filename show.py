import torch
import hypernetx as hnx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_scatter import scatter_mean
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
dataname='esol'
model_dict=torch.load('./results/chhtrans_experiment_20251228_175227/models/esol_best_model_epoch_92.pth')
data=torch.load('dataset/esol_final.pt')

aphabet=data[2]
dataset=data[1]
l1=[]

for id,data in enumerate(dataset):
    num_atom=data.x.shape[0]
    # print(num_atom)
    l1.append(num_atom)

# for id,num in enumerate(l1):
#     if num==24:
#         choice_id=id
#         break
choice_id=705
atom_dim = dataset[0].x.shape[1]
bond_dim = dataset[0].bond_feature.shape[1]
angle_dim = dataset[0].angle_feature.shape[1]
num_blocks=3
from CHHTrans_hete import HyperGrpahTransformer
model = HyperGrpahTransformer(
            atom_dim=atom_dim, bond_dim=bond_dim, angle_dim=angle_dim,
            emb_dim=128, dropout=0.6, conj_dim=10,
            batch_size=128,
            num_blocks=3, 
            num_classes=1,
        
        ).to('cpu')


model.load_state_dict(model_dict['model_state_dict'])


NUMBER_TO_SYMBOL = {
        1: "H",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        19: "K",
        20: "Ca",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        34: "Se",
        35: "Br",
        53: "I",
    }
from typing import List, Union, Dict

def encode_atoms(seq: List[Union[int, str]], number_to_symbol: Dict[int, str] = None) -> List[str]:
    if number_to_symbol is None:
        number_to_symbol = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}
    counts: Dict[str, int] = {}
    out: List[str] = []
    for item in seq:
        if isinstance(item, int):
            symbol = number_to_symbol.get(item)
            if symbol is None:
                raise ValueError(f"未知原子序数: {item}")
        else:
            symbol = str(item)
        counts[symbol] = counts.get(symbol, 0) + 1
        out.append(f"{symbol}{counts[symbol]}")
    return out
def convert_hyper_edge_index_to_string(hyper_edge_index,formal_name):
        return np.stack([hyper_edge_index[0], formal_name(hyper_edge_index[1])], axis=0)
    # m1 = convert_hyper_edge_index_to_string(mapped_hyper_edge_index,formal_name)
        

def formal_name(hyper_edge_index,int_to_symbol_dict=NUMBER_TO_SYMBOL,):
        for id,item in enumerate(hyper_edge_index):
            if type(item)==np.int64:
                hyper_edge_index[id]=int_to_symbol_dict[item]
            if type(item)==str:
                if 'UNKNOWN(13)' in item:
                    hyper_edge_index[id]=item.replace('UNKNOWN(13)','(in_ring)')
                if 'conj_type_0' in item:
                    hyper_edge_index[id]=item.replace('conj_type_0','Aromatic_conj')
                if 'conj_type_1' in item:
                    hyper_edge_index[id]=item.replace('conj_type_1','Olefin_conj')
                if 'conj_type_2' in item:
                    hyper_edge_index[id]=item.replace('conj_type_2','Carbonyl_conj')
                if 'conj_type_3' in item:
                    hyper_edge_index[id]=item.replace('conj_type_3','Imine_conj')
            if type(item)==tuple:
                hyper_edge_index[id]=''
                for i in item[:2]:
                    hyper_edge_index[id]+=int_to_symbol_dict[i]
                bond_type=item[2]
                if bond_type==1:
                    hyper_edge_index[id]=hyper_edge_index[id][0]+'-'+hyper_edge_index[id][1]
                if bond_type==2:
                    hyper_edge_index[id]=hyper_edge_index[id][0]+'='+hyper_edge_index[id][1]
                if bond_type==3:
                    hyper_edge_index[id]=hyper_edge_index[id][0]+'≡'+hyper_edge_index[id][1]
                if bond_type==13:
                    hyper_edge_index[id]=hyper_edge_index[id][0]+'^'+hyper_edge_index[id][1]
                if bond_type>3 and bond_type!=13:
                    hyper_edge_index[id]=hyper_edge_index[id]+'('+str(bond_type)+')'
        return hyper_edge_index
 

# try:
_,_,attnetion_list=model(dataset[choice_id].cpu())

node_type=dataset[choice_id].node_type
hyperedge_type=dataset[choice_id].hyperedge_type
node_id_type_dict={node_id:one_node_type for node_id,one_node_type in enumerate(encode_atoms(node_type.tolist(),NUMBER_TO_SYMBOL))}
hyperedge_type_dict={hyperedge_id:one_hyperedge_type.item() for hyperedge_id,one_hyperedge_type in enumerate(hyperedge_type)}

# print('node_id_type_dict',node_id_type_dict)
# print('hyperedge_type_dict',hyperedge_type_dict)
reverse_alphabet={v:k for k,v in aphabet.items()}
mapped_hyper_edge_index_list=[]
hyper_edge_index=dataset[choice_id].edge_index

new_node_list=[]
new_hyperedge_list=[]
for orignal_node_id in hyper_edge_index[0].tolist():
    new_node_id=node_id_type_dict[orignal_node_id]
    new_node_list.append(new_node_id)
for orignal_hyperedge_id in hyper_edge_index[1].tolist():
    new_hyperedge_id=reverse_alphabet[hyperedge_type_dict[orignal_hyperedge_id]]
    new_hyperedge_list.append(new_hyperedge_id)
# new_node_id_list=encode_atoms(new_node_id_list,NUMBER_TO_SYMBOL)
mapped_hyper_edge_index=[new_node_list,new_hyperedge_list]
n_0=convert_hyper_edge_index_to_string(mapped_hyper_edge_index,formal_name)
attention_block_list=[]
for block in range(num_blocks):
    each_type_list=[]
    for edge_type in attnetion_list:
        each_type_list+=edge_type[block].tolist()
    attention_block_list.append(each_type_list)

smiles=dataset[choice_id].smiles
mol=Chem.MolFromSmiles(smiles)
AllChem.Compute2DCoords(mol)
doubel_d_pos=torch.tensor(mol.GetConformer().GetPositions()[:,0:2])
# print(type(doubel_d_pos))
node_to_edge=dataset[choice_id].edge_index
doubel_d_edge_pos=scatter_mean(doubel_d_pos[node_to_edge[0]],node_to_edge[1].unsqueeze(-1).expand(-1,doubel_d_pos.size(1)),dim=0)
# print(edge_pos)
# exit()
unique_node_id_list=[]
for node_id in n_0[0]:
    if node_id not in unique_node_id_list:
        unique_node_id_list.append(node_id)
unique_edge_id_list=[]
for edge_id in n_0[1]:
    if edge_id not in unique_edge_id_list:
        unique_edge_id_list.append(edge_id)
pos_dict={}
for pos_id,pos in enumerate(doubel_d_pos):
    pos_dict[unique_node_id_list[pos_id]]=(pos[0].item(),pos[1].item())
for edge_id,edge_pos in enumerate(doubel_d_edge_pos):
    pos_dict[unique_edge_id_list[edge_id]]=(edge_pos[0].item(),edge_pos[1].item())
# print(pos_dict)
# print('*'*60)
# print(n_0[1])
# print(unique_edge_id_list)
# print(doubel_d_edge_pos.shape)
# print(len(attention_block_list[1]))
# print(n_0)
H_0 = hnx.Hypergraph(n_0.T)
# H_2 = hnx.Hypergraph(n_2.T)
fig = plt.figure(figsize=(100,70))
gs = fig.add_gridspec(2, num_blocks, height_ratios=[1, 0.05], wspace=0.1, hspace=0.02)
axes = [fig.add_subplot(gs[0, i]) for i in range(num_blocks)]
cax = fig.add_axes([0.2, 0.3, 0.6, 0.02])

# all_scores=[]
# for blk in range(num_blocks):
#     for edge_score_list in attention_block_list[blk]:
#         all_scores.append(edge_score_list[0])
# all_scores=np.array(all_scores,dtype=float)
# if all_scores.size>0:
#     g_min=float(all_scores.min())
#     g_max=float(all_scores.max())
# else:
#     g_min=g_max=0.0
cmap=plt.cm.get_cmap('YlGnBu')
node_cmap=plt.cm.get_cmap('Greys')

def transform_scores(scores, lower_pct=2.0, upper_pct=98.0, scale=None, method='asinh', gamma=0.6):
    s = np.asarray(scores, dtype=float)
    lo, hi = np.percentile(s, [lower_pct, upper_pct])
    s = np.clip(s, lo, hi)
    if scale is None:
        scale = np.median(np.abs(s)) + 1e-8
    if method == 'asinh':
        y = np.arcsinh(s/scale)
    elif method == 'log1p':
        y = np.log1p(np.maximum(s, 0.0)/scale)
    elif method == 'signed_log':
        y = np.sign(s) * np.log1p(np.abs(s)/scale)
    else:
        y = s
    ymin = y.min()
    ymax = y.max()
    y = (y - ymin) / (ymax - ymin + 1e-12)
    y = np.power(y, gamma)
    return y
# print('all_scores:',all_scores)

def adjust_label_positions(edge_positions, node_positions, min_dist=0.7, iterations=60, step=0.04):
    P = np.array(edge_positions, dtype=float).copy()
    N = np.array(node_positions, dtype=float)
    for _ in range(iterations):
        disp = np.zeros_like(P)
        for i in range(P.shape[0]):
            for j in range(P.shape[0]):
                if i == j:
                    continue
                d = P[i] - P[j]
                dist = np.linalg.norm(d)
                if dist < min_dist and dist > 1e-12:
                    disp[i] += (d / (dist + 1e-12)) * (min_dist - dist) * 0.5
        for i in range(P.shape[0]):
            for k in range(N.shape[0]):
                d = P[i] - N[k]
                dist = np.linalg.norm(d)
                if dist < min_dist and dist > 1e-12:
                    disp[i] += (d / (dist + 1e-12)) * (min_dist - dist) * 0.3
        P += np.clip(disp, -step, step)
    return P

for block in range(num_blocks):
    H_0 = hnx.Hypergraph(n_0.T)
    # H_2 = hnx.Hypergraph(n_2.T)

    example_attention_block_list = attention_block_list[block]
    edge_attention_score_list = []
    for edge_score_list in example_attention_block_list:
        edge_attention_score_list.append(float(np.mean(edge_score_list)))
    scores = np.array(edge_attention_score_list, dtype=float)
    print(f'block {block}_scores:',scores)
    if scores.size > 0:
        n_edges = len(unique_edge_id_list)
        if scores.shape[0] != n_edges:
            if scores.shape[0] > n_edges:
                scores = scores[:n_edges]
            else:
                scores = np.pad(scores, (0, n_edges - scores.shape[0]), mode='edge')
        scores = transform_scores(scores, lower_pct=2.0, upper_pct=98.0, method='asinh', gamma=0.6)
        s_min = float(scores.min())
        s_max = float(scores.max())
        if s_max == s_min:
            normed = np.ones(scores.shape, dtype=float)
        else:
            normed = (scores - s_min) / (s_max - s_min)
        bnorm = mcolors.Normalize(vmin=s_min, vmax=s_max)
        edge_color_map = {unique_edge_id_list[i]: cmap(bnorm(scores[i])) for i in range(n_edges)}
        linewidth_map = {unique_edge_id_list[i]: float(2.0 + 6.0 * normed[i]) for i in range(n_edges)}
        edge_score_map = {unique_edge_id_list[i]: float(scores[i]) for i in range(n_edges)}
        pair_scores = torch.tensor([edge_score_map[e] for e in n_0[1]], dtype=torch.float32)
        node_scores = scatter_mean(pair_scores, node_to_edge[0], dim=0)
        ns_min = float(node_scores.min())
        ns_max = float(node_scores.max())
        if ns_max == ns_min:
            node_norm = torch.ones_like(node_scores)
        else:
            node_norm = (node_scores - ns_min) / (ns_max - ns_min)
    else:
        edge_color_map = {}
        linewidth_map = {}
        node_norm = torch.zeros(doubel_d_pos.size(0))
    ax = axes[block]
    hnx.draw(
        H_0.dual(),
        ax=ax,
        pos=pos_dict,
        edges_kwargs={'edgecolors': edge_color_map, 'linewidths': linewidth_map},
        node_labels_kwargs={'fontsize':40,'fontweight':'bold'}
    )
    edge_positions = np.array([pos_dict[e] for e in unique_edge_id_list], dtype=float)
    node_positions = np.array([pos_dict[n] for n in unique_node_id_list], dtype=float)
    # adjusted = adjust_label_positions(edge_positions, node_positions, min_dist=0.4, iterations=60, step=0.04)
    for i,e in enumerate(unique_edge_id_list):
        x,y = edge_positions[i]
        ax.text(x, y, str(e), fontsize=40, fontweight='bold', ha='center', va='center', zorder=10)
    xy = doubel_d_pos.numpy()
    ax.scatter(xy[:,0], xy[:,1], c=node_norm.numpy(), cmap=node_cmap, vmin=0.0, vmax=1.0, s=3000, edgecolors='black', linewidths=6, zorder=5)
    ax.set_title(f"block {block}", fontsize=60, fontweight='bold', pad=0, y=0.7)

plt.suptitle(f'Smile:{smiles}', fontsize=100, fontweight='bold', y=0.78)
sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=mcolors.Normalize(vmin=0.0, vmax=1.0))
sm.set_array(np.linspace(0.0,1.0,256))
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_label('Node Attention Score', fontsize=40, fontweight='bold')
cbar.ax.tick_params(labelsize=30)
edge_cmap_sample_ax = fig.add_axes([0.2, 0.24, 0.6, 0.02])
edge_gradient = np.linspace(0.0, 1.0, 512, dtype=float).reshape(1, -1)
edge_cmap_sample_ax.imshow(edge_gradient, aspect='auto', cmap=cmap, extent=[0.0,1.0,0.0,1.0])
edge_cmap_sample_ax.set_yticks([])
edge_cmap_sample_ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
edge_cmap_sample_ax.set_xticklabels(['0.0','0.2','0.4','0.6','0.8','1.0'])
edge_cmap_sample_ax.set_xlabel('Edge Attention Score', fontsize=40, fontweight='bold')
edge_cmap_sample_ax.tick_params(axis='x', labelsize=30)
plt.tight_layout(rect=[0,0,1,0.95], pad=0.2)
plt.savefig(f'example_{choice_id}.png')
    # except:
    #         pass
    # hnx.draw(H_2.dual(),ax=plt.subplot(3,1,3))





    
