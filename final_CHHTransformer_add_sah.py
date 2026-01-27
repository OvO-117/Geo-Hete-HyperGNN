import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, global_mean_pool,global_max_pool
from SAHPooling import HypergraphPooling
from CHHLayer import HypergraphConv_modif
from hyper_utils import AttentionModel,Atom_AttentionModel,GEANet,change_GEANet, hypermol_GEANet
from hypergraph_utils_data import get_jaccard_matrix_tensor_optimized  # 添加导入
from torch_scatter import scatter_add
import math
import torch.nn.functional as F

class RBF(nn.Module):
    def __init__(self, in_dim, out_dim, num_centers=2):
        super(RBF, self).__init__()
        self.num_centers = num_centers
        self.centers = nn.Parameter(torch.linspace(0, math.pi, num_centers))
        self.gamma = nn.Parameter(torch.ones(1))
        self.linear = nn.Linear(num_centers * in_dim, out_dim)

    def forward(self, x):
        dist = (x.unsqueeze(1) - self.centers.unsqueeze(0).unsqueeze(2)).abs()
        rbf = torch.exp(-self.gamma * dist.pow(2)).view(x.size(0), -1)
        return self.linear(rbf)

class HyperGCNBlock(nn.Module):
    """超图卷积块，包含HyperGCN + 激活 + Dropout + Pooling + Jaccard矩阵更新"""
    def __init__(self, emb_dim, dropout=0.4, use_pooling=True, ratio=0.8):
        super(HyperGCNBlock, self).__init__()
        self.hypergcn = HypergraphConv_modif(emb_dim, emb_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.use_pooling = use_pooling
        if use_pooling:
            self.pool = HypergraphPooling(emb_dim, ratio=ratio)
    
    def update_jaccard_matrix(self, hyperedge_index):
        """根据新的超图边索引重新计算jaccard矩阵"""
        jaccard_edge_index, jaccard_edge_weight = get_jaccard_matrix_tensor_optimized(hyperedge_index)
        return jaccard_edge_index.to(hyperedge_index.device), jaccard_edge_weight.to(hyperedge_index.device)

    def forward(self, x, hyperedge_index, jaccard_edge_index, jaccard_edge_weight, node_batch_idx):
        # HyperGCN卷积
        z = self.hypergcn(x=x, hyperedge_index=hyperedge_index, 
                         jaccard_edge_index=jaccard_edge_index, 
                         jaccard_edge_weight=jaccard_edge_weight)
        z = self.gelu(z)
        # print(self.dropout)
        z = self.dropout(z)
        
        # 计算图级别特征
        z_graph = global_mean_pool(z, node_batch_idx)
        # print(self.pool.ratio)
        # 如果使用pooling
        if self.use_pooling:
            z_pool, edge_index_pool, _, batch_pool, perm, score = self.pool(z, hyperedge_index, batch=node_batch_idx)
            # 重新计算jaccard矩阵
            jaccard_index_pool, jaccard_attr_pool = self.update_jaccard_matrix(edge_index_pool)
            return z_pool, edge_index_pool, jaccard_index_pool, jaccard_attr_pool, batch_pool, z_graph, perm, score
        else:
            return z, hyperedge_index, jaccard_edge_index, jaccard_edge_weight, node_batch_idx, z_graph, None, None

class HyperGrpahTrasnformer(torch.nn.Module):
    def __init__(self, atom_dim, bond_dim, angle_dim, conj_dim=10,emb_dim=128, dropout=0.4, num_blocks=2, heads=1, temperature=0.5, ratio=0.5, num_classes=2,self_supervised=False):
        super(HyperGrpahTrasnformer, self).__init__()
        
        self.num_blocks = num_blocks
        self.temperature = temperature  # 存储temperature参数
        self.num_classes = num_classes
        
        # 原有的编码器
        self.atom_embedding_list = nn.ModuleList()
        for _ in range(atom_dim):
            emb = nn.Embedding(2, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        self.bond_embedding_list = nn.ModuleList()
        for _ in range(bond_dim):
            emb = nn.Embedding(2, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)
       
        self.angle_encoder = RBF(angle_dim, emb_dim)
        self.conj_encoder = nn.Linear(conj_dim, emb_dim)
        nn.init.xavier_uniform_(self.conj_encoder.weight.data)
        # 创建多个HyperGCN块
        self.hypergcn_blocks = nn.ModuleList()
        for i in range(num_blocks):
            # 最后一个块不使用pooling
            use_pooling = (i < num_blocks - 1)
            self.hypergcn_blocks.append(HyperGCNBlock(emb_dim, dropout, use_pooling, ratio))
        
        # 将 decoder 输出维度从 1 改为 num_classes，支持分类 logits
        if num_classes is not None:
            self.decoder = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.Linear(emb_dim//2, self.num_classes))
        else:
            self.decoder = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.Linear(emb_dim//2, 1))
        self.drop3 = nn.Dropout(dropout)
        # self.GEA_layer = change_GEANet(emb_dim, node_dim=emb_dim, heads=1)
        self.hypermol_geaet=hypermol_GEANet(emb_dim,node_dim=emb_dim,heads=heads)  # 使用传入的heads参数
        # 保存开关，决定是否启用多任务头
        self.self_supervised = self_supervised
        if self_supervised:
            self.morgan_predictor=nn.Linear(emb_dim,2048)
            self.logp_predictor=nn.Linear(emb_dim,1)
            self.tpsa_predictor=nn.Linear(emb_dim,1)
        self.expert_mlp_x = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.expert_mlp_bond = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.expert_mlp_angle = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.expert_mlp_conj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))

        # 单任务分类路由器（始终创建）
        self.router_cls = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.GELU(), nn.Linear(emb_dim//2, 4))

        # 仅在自监督（预训练多任务）时创建多任务路由器与预测头
        if self.self_supervised:
            self.router_morgan = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.GELU(), nn.Linear(emb_dim//2, 4))
            self.router_logp = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.GELU(), nn.Linear(emb_dim//2, 4))
            self.router_tpsa = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.GELU(), nn.Linear(emb_dim//2, 4))

            self.morgan_predictor = nn.Linear(emb_dim, 2048)
            self.logp_predictor = nn.Linear(emb_dim, 1)
            self.tpsa_predictor = nn.Linear(emb_dim, 1)
    def _remap_conj_indices(self, atom_conj_index, conj_embedding):
        """
        重新编码atom_conj_index的第1行，使其与conj_embedding的维度一致
        
        Args:
            atom_conj_index: [2, num_conj_atoms] 原子共轭索引
            conj_embedding: [num_conj_groups, emb_dim] 共轭特征嵌入
        
        Returns:
            remapped_atom_conj_index: 重新编码后的原子共轭索引
        """
        if atom_conj_index.size(1) == 0:
            return atom_conj_index
        
        # 获取原始的共轭组id
        original_conj_ids = atom_conj_index[1]
        
        # 获取唯一的共轭组id并排序
        unique_conj_ids = torch.unique(original_conj_ids, sorted=True)
        
        # 创建从原始id到新id的映射
        # 新id从0开始连续编号
        id_mapping = {}
        for new_id, original_id in enumerate(unique_conj_ids.tolist()):
            id_mapping[original_id] = new_id
        
        # 应用映射
        remapped_conj_ids = torch.zeros_like(original_conj_ids)
        for i, original_id in enumerate(original_conj_ids.tolist()):
            remapped_conj_ids[i] = id_mapping[original_id]
        
        # 构建重新编码后的atom_conj_index
        remapped_atom_conj_index = torch.stack([
            atom_conj_index[0],  # 原子id保持不变
            remapped_conj_ids    # 共轭组id重新编码
        ])
        
        return remapped_atom_conj_index
    
    def _map_conj_to_atom(self, conj_embedding, atom_conj_index, num_atoms):
        # 初始化原子级别的共轭特征为零
        conj_to_atom_feature = torch.zeros(num_atoms, conj_embedding.size(1), 
                                          device=conj_embedding.device, 
                                          dtype=conj_embedding.dtype)
        
        # 如果有参与共轭的原子
        if atom_conj_index.size(1) > 0:
            # 重新编码共轭组索引
            remapped_atom_conj_index = self._remap_conj_indices(atom_conj_index, conj_embedding)
            
            atom_indices = remapped_atom_conj_index[0]  # 参与共轭的原子索引
            conj_indices = remapped_atom_conj_index[1]  # 重新编码后的共轭组索引
            # print(torch.max(remapped_atom_conj_index[1]),torch.max(remapped_atom_conj_index[0]))
            # 验证索引范围
            max_conj_idx = torch.max(conj_indices).item()
            if max_conj_idx >= conj_embedding.size(0):
                raise ValueError(f"共轭组索引超出范围: max_idx={max_conj_idx}, conj_embedding_size={conj_embedding.size(0)}")
            
            # 使用scatter_add将共轭特征分配给对应的原子
            conj_to_atom_feature = scatter_add(conj_embedding[conj_indices], 
                                              atom_indices, 
                                              dim=0, 
                                              dim_size=num_atoms)
        
        return conj_to_atom_feature
    def forward(self, graph,multi_task_learning=False):
        ## get data 
        node_batch_idx = graph.batch
        atom_feature = graph.x
        bond_feature = graph.bond_feature
        angle_feature = graph.angle_feature
        hyper_edge_index = graph.edge_index
        conj_feature=graph.conj_feature  # 添加共轭特征
        atom_conj_index=graph.atom_conj_index  # 添加原子共轭索引
        bond_angle, atom_bond, atom_angle = graph.bond_angle_index, graph.atom_bond_index, graph.atom_angle_index
        jaccard_index, jaccard_attr = graph.jaccard_index, graph.jaccard_weight

        ## get embedding
        atom_embedding = 0
        for i in range(atom_feature.shape[1]):
            atom_embedding += self.atom_embedding_list[i](atom_feature[:, i].long())

        bond_embedding = 0
        for i in range(bond_feature.shape[1]):
            bond_embedding += self.bond_embedding_list[i](bond_feature[:, i].long())
        angle_embedding = self.angle_encoder(angle_feature)
        
        conj_embedding = self.conj_encoder(conj_feature)

        conj_to_atom_feature = self._map_conj_to_atom(conj_embedding, atom_conj_index, atom_feature.size(0))
        ##fusion node feature
        atom_indices, angle_indices = atom_angle[0], atom_angle[1]
        angle_to_atom_feature = scatter_add(angle_embedding[angle_indices], atom_indices, dim=0, dim_size=atom_feature.size(0))
        
        atom_indices, bond_indices = atom_bond[0], atom_bond[1]
        bond_to_atom_feature = scatter_add(bond_embedding[bond_indices], atom_indices, dim=0, dim_size=atom_feature.size(0))
        # print(len(conj_to_atom_feature))
        ## external GEA layer
        ex_fusion = self.hypermol_geaet(atom_embedding, angle_to_atom_feature , bond_to_atom_feature, conj_to_atom_feature )
        ex_graph = global_mean_pool(ex_fusion, node_batch_idx)
        
        ## 多个HyperGCN块的前向传播
        current_x = ex_fusion
        current_hyperedge_index = hyper_edge_index
        current_jaccard_index = jaccard_index
        current_jaccard_attr = jaccard_attr
        current_batch = node_batch_idx
        
        graph_features = []  # 存储每个块的图级别特征
        total_kl_loss = torch.tensor(0.0, device=ex_fusion.device)  # 初始化总KL loss
        last_perm = None  # 记录最后一次 pooling 的节点选择映射
        
        for i, block in enumerate(self.hypergcn_blocks):
            current_x, current_hyperedge_index, current_jaccard_index, current_jaccard_attr, current_batch, z_graph, perm, score = block(
                current_x, current_hyperedge_index, current_jaccard_index, current_jaccard_attr, current_batch
            )
            graph_features.append(z_graph)
            if perm is not None:
                last_perm = perm  # 仅在使用 pooling 的 block 返回有效的映射
            
            # 计算当前block的z_graph与ex_graph的KL散度并累加
            kl_loss = self.compute_entropy_loss(z_graph, ex_graph)
            total_kl_loss = total_kl_loss + kl_loss

        # === pooling 后更新三类索引矩阵并得到当前节点空间的 bond/angle/conj ===
        if last_perm is None:
            # 未发生 pooling，则沿用原子编号顺序
            last_perm = torch.arange(atom_feature.size(0), device=atom_feature.device)
        updated_atom_bond, updated_atom_angle, updated_atom_conj_index = self._update_indices_after_pool(
            atom_bond, atom_angle, atom_conj_index, last_perm, num_nodes=atom_feature.size(0)
        )
        
        # scatter 到当前节点空间
        cur_angle = scatter_add(angle_embedding[updated_atom_angle[1]], updated_atom_angle[0], dim=0, dim_size=current_x.size(0))
        cur_bond = scatter_add(bond_embedding[updated_atom_bond[1]], updated_atom_bond[0], dim=0, dim_size=current_x.size(0))
        cur_conj = self._map_conj_to_atom(conj_embedding, updated_atom_conj_index, current_x.size(0))

        # === MoE：四个专家对应 current_x / bond / angle / conj，每个一个 MLP ===
        expert_x = self.expert_mlp_x(current_x)
        expert_b = self.expert_mlp_bond(cur_bond)
        expert_a = self.expert_mlp_angle(cur_angle)
        expert_c = self.expert_mlp_conj(cur_conj)

        # 图级别聚合各专家输出
        ex_x_graph = global_mean_pool(expert_x, current_batch)
        ex_b_graph = global_mean_pool(expert_b, current_batch)
        ex_a_graph = global_mean_pool(expert_a, current_batch)
        ex_c_graph = global_mean_pool(expert_c, current_batch)

        # 融合所有块的图级别特征（保持原有设计）
        final_graph_feature = ex_graph
        for graph_feat in graph_features:
            final_graph_feature = final_graph_feature + graph_feat

        # === 路由并输出（依据 self_supervised 控制多/单任务分支）===
        if not self.self_supervised:
            gate = torch.softmax(self.router_cls(final_graph_feature), dim=-1)  # [B, 4]
            moe_feature = (gate[:, 0:1] * ex_x_graph +
                           gate[:, 1:2] * ex_b_graph +
                           gate[:, 2:3] * ex_a_graph +
                           gate[:, 3:4] * ex_c_graph)
            predict = self.decoder(moe_feature)
            return predict, total_kl_loss
        else:
            gate_morgan = torch.softmax(self.router_morgan(final_graph_feature), dim=-1)
            gate_logp = torch.softmax(self.router_logp(final_graph_feature), dim=-1)
            gate_tpsa = torch.softmax(self.router_tpsa(final_graph_feature), dim=-1)

            moe_morgan = (gate_morgan[:, 0:1] * ex_x_graph +
                          gate_morgan[:, 1:2] * ex_b_graph +
                          gate_morgan[:, 2:3] * ex_a_graph +
                          gate_morgan[:, 3:4] * ex_c_graph)
            moe_logp = (gate_logp[:, 0:1] * ex_x_graph +
                        gate_logp[:, 1:2] * ex_b_graph +
                        gate_logp[:, 2:3] * ex_a_graph +
                        gate_logp[:, 3:4] * ex_c_graph)
            moe_tpsa = (gate_tpsa[:, 0:1] * ex_x_graph +
                        gate_tpsa[:, 1:2] * ex_b_graph +
                        gate_tpsa[:, 2:3] * ex_a_graph +
                        gate_tpsa[:, 3:4] * ex_c_graph)

            morgan = self.morgan_predictor(moe_morgan)
            logp = self.logp_predictor(moe_logp)
            tpsa = self.tpsa_predictor(moe_tpsa)
            predict = {'morgan': morgan, 'logp': logp, 'tpsa': tpsa}
            return predict, total_kl_loss

    def compute_entropy_loss(self, z1, z2, temperature=None):
        """计算信息熵损失，确保pooling后信息分布相似"""
        # 使用传入的temperature或默认的self.temperature
        temp = temperature if temperature is not None else self.temperature
        
        # 归一化特征
        z1_scaled = z1 / temp
        z2_scaled = z2 / temp
        
        # 计算softmax分布
        p = F.softmax(z1_scaled, dim=-1)
        q = F.softmax(z2_scaled, dim=-1)
        
        # 计算KL散度
        kl_div = F.kl_div(q.log(), p, reduction='batchmean')
        return kl_div
        # === MoE experts & routers ===
       

    def _update_indices_after_pool(self, atom_bond, atom_angle, atom_conj_index, node_perm, num_nodes):
        """
        根据 pooling 的 node_perm 更新三类索引矩阵到新节点空间：
        - 仅保留被选中的节点；
        - 将原始 atom 索引映射为 [0, new_num_nodes) 的连续编号。
        """
        device = node_perm.device
        node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        node_map[node_perm] = torch.arange(node_perm.size(0), device=device)

        # atom_bond: [2, E], row0=atom_idx, row1=bond_idx
        ab_atom = atom_bond[0]
        ab_bond = atom_bond[1]
        ab_mask = node_map[ab_atom] != -1
        updated_atom_bond = torch.stack([node_map[ab_atom[ab_mask]], ab_bond[ab_mask]], dim=0)

        # atom_angle: [2, E], row0=atom_idx, row1=angle_idx
        aa_atom = atom_angle[0]
        aa_angle = atom_angle[1]
        aa_mask = node_map[aa_atom] != -1
        updated_atom_angle = torch.stack([node_map[aa_atom[aa_mask]], aa_angle[aa_mask]], dim=0)

        # atom_conj_index: [2, E], row0=atom_idx, row1=conj_group_idx（组ID保持不变，后续 remap 内部处理）
        ac_atom = atom_conj_index[0]
        ac_conj = atom_conj_index[1]
        ac_mask = node_map[ac_atom] != -1
        updated_atom_conj_index = torch.stack([node_map[ac_atom[ac_mask]], ac_conj[ac_mask]], dim=0)

        return updated_atom_bond, updated_atom_angle, updated_atom_conj_index
