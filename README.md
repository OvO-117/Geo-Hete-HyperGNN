## Geo-Hete-HyperGNN
Geo-Hete-HyperGNN: A prior-free and universal Molecular Hypergraph Representation Learning strategy


### 🛠️ Environment requirement

- **Python**:3.10.19
- **CUDA**: 11.8 
- **PyTorch**:2.1.1+cu118
- **networkx**: 3.4.2
- **rdkit**: 2025.3.6
- **torch_geometric**:2.5.1
- detailed in requirement.txt

### Download pretrain_model.pt
通过网盘分享的文件：pretrain_model.pth
链接: https://pan.baidu.com/s/1sW6Uem52xMpi4VZIjqlHcQ?pwd=v9xe 提取码: v9xe 
--来自百度网盘超级会员v4的分享

### Download tunning dataset
通过网盘分享的文件：TUNING dataset
链接: https://pan.baidu.com/s/1LVYfsF6Uz2tLFQIVP1qO-w?pwd=g45i 提取码: g45i 
--来自百度网盘超级会员v4的分享

### Run class task(clintox,bbbp,tox21,toxcast,bace) 
```bash
python main_class.py

### Run reg task(esol,lipo,freesolv) 
```bash
python main_regression.py

