import pandas as pd
from torch_geometric.utils.smiles import from_smiles
from tqdm import tqdm
from molecular_dataset import PretrainDataset
from dataset_utils import OneHotTransform, FrozenOneHotTransform
import torch

filtered_smiles_list=pd.read_csv('pretrain-5m-filter-10.csv')['smiles'].tolist()

final_data=[]
for i in tqdm(filtered_smiles_list):
    data=from_smiles(i)
    final_data.append(data)

dataset = PretrainDataset(final_data,root='700k')
transform = OneHotTransform(dataset)
dataset_transformed = [transform(data) for data in dataset]
onehot_spec = FrozenOneHotTransform.build_spec(dataset)
torch.save(onehot_spec,'./700k/onehot_spec.pt')