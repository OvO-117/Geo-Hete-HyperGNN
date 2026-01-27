import torch
from torch_geometric.datasets import MoleculeNet
from dataset_utils import FrozenOneHotTransform, create_hypergraph_dataset

dataset_name='BACE'
onehot_spec = torch.load("./700k/onehot_spec.pt", map_location="cpu")
transform = FrozenOneHotTransform(onehot_spec)
dataset_untransformed = MoleculeNet('./original_data', dataset_name)
dataset_transformed = [transform(data) for data in dataset_untransformed]
# print(dataset_transformed[0])
print("创建超图数据集...")
hypergraph_list = create_hypergraph_dataset(dataset_transformed,num_workers=10)

print(f"超图数据集创建完成，包含{len(hypergraph_list[1])}个样本")
torch.save(hypergraph_list, f'./processed_data/{dataset_name}.pt')

dataset=hypergraph_list[1]
new_data=[]
for data in dataset:
    if data.x.shape==data.node_pos_emb.shape:
        new_data.append(data)
    else:
        pass
print(len(new_data))
torch.save(new_data, f'./processed_data/{dataset_name}_final.pt')
print(f"位置编码对其筛选，包含{len(new_data)}个样本")

id_list=[]
for id,data in enumerate(dataset):
    if len(set(data.x.flatten().tolist()))>2:
        id_list.append(id)
    else:
        pass

edge_list=[]
for id,data in enumerate(dataset):
    if len(set(data.bond_feature.flatten().tolist()))>2:
        edge_list.append(id)
    else:
        pass

total_fliter=edge_list+id_list
fliter_list=[]
for id,data in enumerate(dataset):
    if id in total_fliter:
        pass
    else:
        fliter_list.append(data)
print(f"x和bond独热编码过滤筛选，包含{len(fliter_list)}个样本")
torch.save(fliter_list,f'./processed_data/{dataset_name}_fliter.pt')
