import torch
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold

def generate_scaffold(smiles, include_chirality=True):
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)

def scaffold_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    N = len(dataset)
    scaffolds = {}
    for i in range(N):
        s = getattr(dataset[i], 'smiles', None)
        if s is None:
            continue
        scaf = generate_scaffold(s, include_chirality=True)
        scaffolds.setdefault(scaf, []).append(i)
    scaffold_sets = sorted(scaffolds.values(), key=lambda x: (len(x), x[0]), reverse=True)
    train_cutoff = frac_train * N
    valid_cutoff = (frac_train + frac_valid) * N
    train_idx, valid_idx, test_idx = [], [], []
    for ss in scaffold_sets:
        if len(train_idx) + len(ss) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(ss) > valid_cutoff:
                test_idx.extend(ss)
            else:
                valid_idx.extend(ss)
        else:
            train_idx.extend(ss)
    return train_idx, valid_idx, test_idx

def extract_labels(items):
    vals = []
    for g in items:
        y = getattr(g, 'y', None)
        if y is None:
            continue
        t = y.view(-1).float()
        if t.numel() == 0:
            continue
        vals.append(float(t[0].item()))
    return vals

def print_stats(name, vals):
    a = np.array(vals, dtype=float)
    if a.size == 0:
        print(f'{name}: 无有效标签')
        return
    print(f'{name}: count={a.size} min={a.min():.6f} max={a.max():.6f} mean={a.mean():.6f} std={a.std():.6f}')

if __name__ == '__main__':
    data = torch.load('./dataset/lipo_final.pt')
    dataset = data[1]
    train_idx, valid_idx, test_idx = scaffold_split(dataset, 0.8, 0.1, 0.1)
    train_vals = extract_labels([dataset[i] for i in train_idx])
    valid_vals = extract_labels([dataset[i] for i in valid_idx])
    test_vals = extract_labels([dataset[i] for i in test_idx])
    all_vals = extract_labels(dataset)
    print_stats('Train', train_vals)
    print_stats('Valid', valid_vals)
    print_stats('Test', test_vals)
    print_stats('All', all_vals)