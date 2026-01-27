# from knowledge_distall import batch_idx
import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, global_mean_pool,global_max_pool,GCNConv
# from SAHPooling import E_HypergraphPooling
# from CHHLayer import HypergraphConv_modif
from ECHHLayer import E_HypergraphConv

# from hyper_utils import AttentionModel,Atom_AttentionModel,GEANet,change_GEANet, hypermol_GEANet
# from hypergraph_utils_data import get_jaccard_matrix_tensor_optimized  # 添加导入
from torch_scatter import scatter_add,scatter_mean
from torch_sparse import SparseTensor
import math
import torch.nn.functional as F
from torch_geometric.utils import to_edge_index, softmax

class Attention_Scatter(nn.Module):
    def __init__(self, emb_dim):
        super(Attention_Scatter, self).__init__()
        self.emb_dim = emb_dim
        self.w_omega = nn.Parameter(torch.Tensor(self.emb_dim,self.emb_dim))
        self.b_omega = nn.Parameter(torch.Tensor(self.emb_dim))
        self.u_omega = nn.Parameter(torch.Tensor(self.emb_dim,1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()
   
    def reset_parameters(self):
       nn.init.xavier_uniform_(self.w_omega)
       nn.init.zeros_(self.b_omega)
       nn.init.xavier_uniform_(self.u_omega)

    def forward(self,x,type_batch_idx=None):
        v = self.tanh(torch.matmul(x, self.w_omega) + self.b_omega)
        vu = torch.matmul(v, self.u_omega)
        if type_batch_idx is None:
            alphas = self.softmax(vu)
            output = torch.sum(x * alphas.reshape(alphas.shape[0],-1,1),dim=1)
            return output,alphas
        alphas = softmax(vu.view(-1), type_batch_idx).view(-1,1)
        output = scatter_add(x * alphas, type_batch_idx, dim=0)
        return output,alphas

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

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss(nn.Module):
    def __init__(self,batch_size,embed_size,lambd=0.005):
        super(BarlowTwinsLoss,self).__init__()
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.lambd = lambd 
        self.bn = nn.BatchNorm1d(self.embed_size, affine=False)

    def forward(self, z1, z2):
        z1 = (z1 - torch.mean(z1, axis = 0))/torch.std(z1, axis = 0)
        z2 = (z2 - torch.mean(z2, axis = 0))/torch.std(z2, axis = 0)
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

class HyperGCNBlock(nn.Module):
    def __init__(self, emb_dim,batch_size, dropout=0.4, ratio=0.8):
        super(HyperGCNBlock, self).__init__()
        self.hypergcn = E_HypergraphConv(emb_dim, emb_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(emb_dim)
        self.X_GNN=GCNConv(emb_dim,emb_dim)
        self.E_GNN=GCNConv(emb_dim,emb_dim)
        self.BarlowTwinsLoss=BarlowTwinsLoss(batch_size,emb_dim)
        self.graph_fusion=nn.Linear(2*emb_dim,emb_dim)
    
    def forward(self, x, hyperedge_feature,hyperedge_index,node_coord,node_batch_idx,hyperedge_batch_idx,X_X,E_E):
        
        z,e,update_coord = self.hypergcn(x=x,hyperedge_feature=hyperedge_feature, hyperedge_index=hyperedge_index,node_coord=node_coord)
        z_implict=self.X_GNN(x,X_X)
        e_implict=self.E_GNN(hyperedge_feature,E_E)
        z_loss=self.BarlowTwinsLoss(z_implict,z)
        e_loss=self.BarlowTwinsLoss(e_implict,e)
        bt_loss=z_loss+e_loss

        z = self.layernorm(z)
        e=self.layernorm(e)

        z = self.gelu(z)
        e=self.gelu(e)

        z = self.dropout(z)
        e=self.dropout(e)

        z=z+x
        e=e+hyperedge_feature

        z_graph = global_mean_pool(z, node_batch_idx)
        e_graph = global_mean_pool(e, hyperedge_batch_idx)
        graph=torch.cat([z_graph,e_graph],dim=1)
        graph=self.graph_fusion(graph)
    
        return z, e,hyperedge_index, update_coord,node_batch_idx,hyperedge_batch_idx, graph,bt_loss

class HyperGrpahTransformer(torch.nn.Module):
    def __init__(self, atom_dim, bond_dim, angle_dim,batch_size, conj_dim=10,emb_dim=128, dropout=0.4, num_blocks=2, heads=1, ratio=0.5, num_classes=2,self_supervised=False, num_atom_classes=None, num_bond_classes=None, num_angle_classes=None, num_conj_classes=None):
        super(HyperGrpahTransformer, self).__init__()
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.emb_dim=emb_dim
    ## init_atom_embedding 
        self.atom_embedding_list = nn.ModuleList()
        for _ in range(atom_dim):
            emb = nn.Embedding(2, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

     ## init_bond_embedding 
        self.bond_embedding_list = nn.ModuleList()
        for _ in range(bond_dim):
            emb = nn.Embedding(2, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    ## init_angle_embedding
        self.angle_encoder = RBF(angle_dim, emb_dim)
        self.conj_encoder = nn.Linear(conj_dim, emb_dim)
        nn.init.xavier_uniform_(self.conj_encoder.weight.data)
        self.coord_encoder = RBF(3, emb_dim)
    
    ## position_embedding_transform
        self.pos_transform=nn.Linear(atom_dim,emb_dim)
    ## fusion node feature
        self.node_fusion=nn.Linear(2*emb_dim,emb_dim)
    ## fusion graph feature
        self.graph_fusion=nn.Linear(2*emb_dim,emb_dim)
    ## create_hypergcn_blocks
        self.hypergcn_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.hypergcn_blocks.append(HyperGCNBlock(emb_dim,batch_size, dropout,ratio))
    
    ## Decoder
        if num_classes is not None:
            self.decoder = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.Linear(emb_dim//2, self.num_classes))
        else:
            self.decoder = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.Linear(emb_dim//2, 1))
        self.drop3 = nn.Dropout(dropout)

        ## mixture_of_experts
        self.expert_mlp_x = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.expert_mlp_atom = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.expert_mlp_bond = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.expert_mlp_angle = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.expert_mlp_conj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))

        # router
        self.router_cls = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.GELU(), nn.Linear(emb_dim//2, 5))
        
        ## semantic_hyperedge
        self.heads=heads
        self.linear_atom_H = nn.Linear(emb_dim, emb_dim)
        self.q_atom_H = nn.Parameter(torch.rand(emb_dim, heads))
        self.linear_bond_H = nn.Linear(emb_dim, emb_dim)
        self.q_bond_H = nn.Parameter(torch.rand(emb_dim, heads))
        self.linear_angle_H = nn.Linear(emb_dim, emb_dim)
        self.q_angle_H = nn.Parameter(torch.rand(emb_dim, heads))
        self.linear_conj_H = nn.Linear(emb_dim, emb_dim)
        self.q_conj_H = nn.Parameter(torch.rand(emb_dim, heads))
        self.att_fusion = nn.Linear(self.heads * emb_dim, emb_dim)
        self.attention_scatter = Attention_Scatter(emb_dim)

        ## self_supervised_decoder
        self.self_supervised = self_supervised
        if self.self_supervised:
            self.router_morgan = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.GELU(), nn.Linear(emb_dim//2, 4))
            self.router_logp = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.GELU(), nn.Linear(emb_dim//2, 4))
            self.router_tpsa = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.GELU(), nn.Linear(emb_dim//2, 4))
            self.morgan_predictor = nn.Linear(emb_dim, 2048)
            self.logp_predictor = nn.Linear(emb_dim, 1)
            self.tpsa_predictor = nn.Linear(emb_dim, 1)
        self.num_atom_classes = num_atom_classes
        self.num_bond_classes = num_bond_classes
        self.num_angle_classes = num_angle_classes
        self.num_conj_classes = num_conj_classes
        if self.num_atom_classes is not None:
            self.atom_classifier = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.GELU(), nn.Linear(emb_dim//2, self.num_atom_classes))
        if self.num_bond_classes is not None:
            self.bond_classifier = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.GELU(), nn.Linear(emb_dim//2, self.num_bond_classes))
        if self.num_angle_classes is not None:
            self.angle_classifier = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.GELU(), nn.Linear(emb_dim//2, self.num_angle_classes))
        if self.num_conj_classes is not None:
            self.conj_classifier = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.GELU(), nn.Linear(emb_dim//2, self.num_conj_classes))

        # self.out_act=nn.Sigmoid()
        self.norm = nn.LayerNorm(emb_dim)
    def get_EE_XX(self,hyperedge_index):
        ## create implict realtion between nodes and hyperedges
        num_nodes = int(hyperedge_index[0].max().item()) + 1
        num_hyperedges = int(hyperedge_index[1].max().item()) + 1
        X_E = SparseTensor(row=hyperedge_index[0], col=hyperedge_index[1], value=torch.ones(hyperedge_index.size(1), device=hyperedge_index.device), sparse_sizes=(num_nodes, num_hyperedges))
        X_X=X_E@X_E.t()
        E_E=X_E.t()@X_E
        X_edge_index,X_edge_attr=to_edge_index(X_X)
        E_edge_index,E_edge_attr=to_edge_index(E_E)
        return  X_edge_index,X_edge_attr,E_edge_index,E_edge_attr

    def hyperedge_semantic_encoder(self,hyperedge_feature,type_atom_hyperedge_id,type_bond_hyperedge_id,type_angle_hyperedge_id,type_conj_hyperedge_id,atom_type_batch_idx,bond_type_batch_idx,angle_type_batch_idx,conj_type_batch_idx):
        atom_hyperedge=hyperedge_feature[type_atom_hyperedge_id]
        bond_hyperedge=hyperedge_feature[type_bond_hyperedge_id]
        angle_hyperedge=hyperedge_feature[type_angle_hyperedge_id]
        conj_hyperedge=hyperedge_feature[type_conj_hyperedge_id]

        attention_atom_hyperedge,each_atom_alpha_in_hyperedge=self.attention_scatter(atom_hyperedge,atom_type_batch_idx)
        attention_bond_hyperedge,each_bond_alpha_in_hyperedge=self.attention_scatter(bond_hyperedge,bond_type_batch_idx)
        attention_angle_hyperedge,each_angle_alpha_in_hyperedge=self.attention_scatter(angle_hyperedge,angle_type_batch_idx)
        attention_conj_hyperedge,each_conj_alpha_in_hyperedge=self.attention_scatter(conj_hyperedge,conj_type_batch_idx)
        padded_attention_conj_hyperedge=torch.zeros_like(attention_atom_hyperedge)
        padded_attention_conj_hyperedge[:attention_conj_hyperedge.shape[0]]=attention_conj_hyperedge
        attention_conj_hyperedge=padded_attention_conj_hyperedge
  
        atom_H_feats = torch.sigmoid(self.linear_atom_H(attention_atom_hyperedge))
        atom_H_alpha = torch.matmul(atom_H_feats, self.q_atom_H)
        bond_H_feats = torch.sigmoid(self.linear_bond_H(attention_bond_hyperedge))
        bond_H_alpha = torch.matmul(bond_H_feats, self.q_bond_H)
        angle_H_feats = torch.sigmoid(self.linear_angle_H(attention_angle_hyperedge))
        angle_H_alpha = torch.matmul(angle_H_feats, self.q_angle_H)
        conj_H_feats = torch.sigmoid(self.linear_conj_H(attention_conj_hyperedge))
        conj_H_alpha = torch.matmul(conj_H_feats, self.q_conj_H)    

        alpha = torch.exp(atom_H_alpha) + torch.exp(bond_H_alpha) + torch.exp(angle_H_alpha) + torch.exp(conj_H_alpha)
        atom_H_alpha = torch.exp(atom_H_alpha) / alpha
        bond_H_alpha = torch.exp(bond_H_alpha) / alpha
        angle_H_alpha = torch.exp(angle_H_alpha) / alpha
        conj_H_alpha = torch.exp(conj_H_alpha) / alpha
        
        # print(atom_H_alpha)
        # print(atom_type_batch_idx)
        each_atom_hyperedge_alpha=torch.tensor([atom_H_alpha[id,:].cpu().detach().numpy() for id in atom_type_batch_idx]).to(atom_hyperedge.device)
        each_bond_hyperedge_alpha=torch.tensor([bond_H_alpha[id,:].cpu().detach().numpy() for id in bond_type_batch_idx]).to(bond_hyperedge.device)
        each_angle_hyperedge_alpha=torch.tensor([angle_H_alpha[id,:].cpu().detach().numpy() for id in angle_type_batch_idx]).to(angle_hyperedge.device)
        if len(conj_type_batch_idx) > 0:
            each_conj_hyperedge_alpha = torch.stack([conj_H_alpha[id, :].detach() for id in conj_type_batch_idx], dim=0)
        else:
            each_conj_hyperedge_alpha = conj_H_alpha[:0]
        each_conj_hyperedge_alpha = each_conj_hyperedge_alpha.to(conj_hyperedge.device)
        # print(each_atom_hyperedge_alpha.shape,each_atom_alpha_in_hyperedge.shape,atom_hyperedge.shape)
        semantic_hyperedge=hyperedge_feature.clone()
        semantic_hyperedge[type_atom_hyperedge_id]=self.att_fusion(torch.cat([each_atom_hyperedge_alpha[:,i].unsqueeze(1)*each_atom_alpha_in_hyperedge*atom_hyperedge for i in range(self.heads)],dim=1))
        semantic_hyperedge[type_bond_hyperedge_id]=self.att_fusion(torch.cat([each_bond_hyperedge_alpha[:,i].unsqueeze(1)*each_bond_alpha_in_hyperedge*bond_hyperedge for i in range(self.heads)],dim=1))
        semantic_hyperedge[type_angle_hyperedge_id]=self.att_fusion(torch.cat([each_angle_hyperedge_alpha[:,i].unsqueeze(1)*each_angle_alpha_in_hyperedge*angle_hyperedge for i in range(self.heads)],dim=1))
        semantic_hyperedge[type_conj_hyperedge_id]=self.att_fusion(torch.cat([each_conj_hyperedge_alpha[:,i].unsqueeze(1)*each_conj_alpha_in_hyperedge*conj_hyperedge for i in range(self.heads)],dim=1))
        atom_type_attention=each_atom_hyperedge_alpha.unsqueeze(1)*each_atom_alpha_in_hyperedge
        bond_type_attention=each_bond_hyperedge_alpha.unsqueeze(1)*each_bond_alpha_in_hyperedge
        angle_type_attention=each_angle_hyperedge_alpha.unsqueeze(1)*each_angle_alpha_in_hyperedge
        conj_type_attention=each_conj_hyperedge_alpha.unsqueeze(1)*each_conj_alpha_in_hyperedge
        return semantic_hyperedge,atom_type_attention,bond_type_attention,angle_type_attention,conj_type_attention
    
    def forward(self, graph):
        ## get data 
        node_batch_idx = graph.batch
        hyperedge_batch_idx=graph.hyperedge_batch_index
        # print(node_batch_idx)
        atom_type_batch_idx = graph.atom_type_batch_index
        bond_type_batch_idx = graph.bond_type_batch_index
        angle_type_batch_idx = graph.angle_type_batch_index
        conj_type_batch_idx = graph.conj_type_batch_index
        atom_feature = graph.x
        bond_feature = graph.bond_feature
        angle_feature = graph.angle_feature
        hyper_edge_index = graph.edge_index
        conj_feature=graph.conj_feature  
        atom_type_index, bond_type_index, angle_type_index, conj_type_index = graph.atom_type_index, graph.bond_type_index, graph.angle_type_index, graph.conj_type_index
        node_coord=graph.coord
        try:
            num_type_hyperedge=torch.sum(torch.tensor([graph.num_atom_types,graph.num_bond_types,graph.num_angle_types,graph.num_conj_types]))
        except:
            num_type_hyperedge=torch.sum(graph.num_atom_types+graph.num_bond_types+graph.num_angle_types+graph.num_conj_types)
        
        ## node_feature(position+coord)
        node_coord_emb=self.coord_encoder(node_coord)
        pos=graph.node_pos_emb
        pos_embedding=self.pos_transform(pos)
        # print(pos_embedding.shape)
        # print(node_coord_emb.shape)
        node_feature=torch.cat([node_coord_emb,pos_embedding],dim=1)
        node_feature=self.node_fusion(node_feature)
        
        ## hyperedge_feature
        type_hyperedge_embedding = torch.zeros([num_type_hyperedge,self.emb_dim])
    
        ## atom_type_hyperedge_feature
        atom_embedding = 0
        for i in range(atom_feature.shape[1]):
            atom_embedding += self.atom_embedding_list[i](atom_feature[:, i].long())
        atom_type_embedding=scatter_add(atom_embedding[atom_type_index[0]], atom_type_index[1], dim=0)
        padding_atom_size=num_type_hyperedge-max(atom_type_index[1])-1
        padded_atom_type_embedding=torch.cat([atom_type_embedding,torch.zeros([padding_atom_size,self.emb_dim]).to(atom_type_embedding.device)],dim=0)

        ## bond_type_hyperedge_feature
        bond_embedding = 0
        for i in range(bond_feature.shape[1]):
            bond_embedding += self.bond_embedding_list[i](bond_feature[:, i].long())
        bond_id,bond_type=bond_type_index[0],bond_type_index[1]
        bond_type_embedding=scatter_add(bond_embedding[bond_id], bond_type, dim=0)
        padding_bond_size=num_type_hyperedge-max(bond_type_index[1])-1
        padded_bond_type_embedding=torch.cat([bond_type_embedding,torch.zeros([padding_bond_size,self.emb_dim]).to(bond_type_embedding.device)],dim=0)
        
        ## angle_type_hyperedge_feature
        angle_embedding = self.angle_encoder(angle_feature)
        angle_type_embedding=scatter_add(angle_embedding[angle_type_index[0]], angle_type_index[1], dim=0)
        padding_angle_size=num_type_hyperedge-max(angle_type_index[1])-1
        padded_angle_type_embedding=torch.cat([angle_type_embedding,torch.zeros([padding_angle_size,self.emb_dim]).to(angle_type_embedding.device)],dim=0)
        
        ## conj_type_hyperedge_feature
        conj_embedding = self.conj_encoder(conj_feature)
        conj_type_embedding=scatter_add(conj_embedding[conj_type_index[0]], conj_type_index[1], dim=0)
        if conj_type_embedding.shape[0]==0:
            padded_conj_type_embedding=torch.zeros_like(type_hyperedge_embedding).to(conj_type_embedding.device)
        else:
            padding_conj_size=num_type_hyperedge-max(conj_type_index[1])-1
            padded_conj_type_embedding=torch.cat([conj_type_embedding,torch.zeros([padding_conj_size,self.emb_dim]).to(conj_type_embedding.device)],dim=0)
        
        ## final_hyperedge_feature
        type_hyperedge_embedding=padded_atom_type_embedding+padded_bond_type_embedding+padded_angle_type_embedding+padded_conj_type_embedding
        type_atom_hyperedge_id=list(sorted(set(graph.atom_type_index[1].tolist())))
        type_bond_hyperedge_id=list(sorted(set(graph.bond_type_index[1].tolist())))
        type_angle_hyperedge_id=list(sorted(set(graph.angle_type_index[1].tolist())))
        type_conj_hyperedge_id=list(sorted(set(graph.conj_type_index[1].tolist())))
        # semantic_hyperedge=self.hyperedge_semantic_encoder(type_hyperedge_embedding,type_atom_hyperedge_id,type_bond_hyperedge_id,type_angle_hyperedge_id,type_conj_hyperedge_id,atom_type_batch_idx\
        # ,bond_type_batch_idx,angle_type_batch_idx,conj_type_batch_idx)
        # semantic_hyperedge=type_hyperedge_embedding

        ## ECHH_HyperGCN
        
        current_x = node_feature
        current_node_coord=node_coord
        current_hyperedge_index = hyper_edge_index
        current_XX,current_X_attr,current_EE,current_E_attr=self.get_EE_XX(current_hyperedge_index)
        current_batch = node_batch_idx
        currrent_hyperedge_batch=hyperedge_batch_idx
        current_hyperedge_feature=type_hyperedge_embedding

        x_graph=global_mean_pool(node_feature, node_batch_idx)
        hyperedge_graph=global_mean_pool(type_hyperedge_embedding, hyperedge_batch_idx)
        graph_feature=torch.cat([x_graph,hyperedge_graph],dim=1)
        init_graph_feature=self.graph_fusion(graph_feature)

        graph_features = []
        atom_type_attention_list=[]
        bond_type_attention_list=[]
        angle_type_attention_list=[]
        conj_type_attention_list=[]
        bt_loss=0
        for i, block in enumerate(self.hypergcn_blocks):
            current_hyperedge_feature,atom_type_attention,bond_type_attention,angle_type_attention,conj_type_attention=self.hyperedge_semantic_encoder(current_hyperedge_feature,type_atom_hyperedge_id,type_bond_hyperedge_id,type_angle_hyperedge_id,type_conj_hyperedge_id,atom_type_batch_idx\
        ,bond_type_batch_idx,angle_type_batch_idx,conj_type_batch_idx)
            current_x, current_hyperedge_feature, current_hyperedge_index, current_node_coord,current_batch,currrent_hyperedge_batch, z_graph,bt_loss = block(
                current_x, current_hyperedge_feature,current_hyperedge_index,current_node_coord,current_batch,currrent_hyperedge_batch,current_XX,current_EE
            )
            graph_features.append(z_graph)
            atom_type_attention_list.append(atom_type_attention)
            bond_type_attention_list.append(bond_type_attention)
            angle_type_attention_list.append(angle_type_attention)
            conj_type_attention_list.append(conj_type_attention)
            bt_loss+=bt_loss
        
    ## get_updated_hyperedge_feature
        cur_atom=current_hyperedge_feature[type_atom_hyperedge_id]
        cur_bond=current_hyperedge_feature[type_bond_hyperedge_id]
        cur_angle=current_hyperedge_feature[type_angle_hyperedge_id]
        cur_conj=current_hyperedge_feature[type_conj_hyperedge_id]

    ## Decoder(mixture of experts)
        expert_x = self.expert_mlp_x(current_x)
        expert_atom=self.expert_mlp_atom(cur_atom)
        expert_bond = self.expert_mlp_bond(cur_bond)
        expert_angle = self.expert_mlp_angle(cur_angle)
        expert_conj = self.expert_mlp_conj(cur_conj)

    ## expert_level_graph_feature
        ex_x_graph = global_mean_pool(expert_x,current_batch)
        ex_atom_graph = global_mean_pool(expert_atom,atom_type_batch_idx)
        ex_b_graph = global_mean_pool(expert_bond,bond_type_batch_idx)
        ex_a_graph = global_mean_pool(expert_angle,angle_type_batch_idx)
        ex_c_graph=torch.zeros_like(ex_b_graph)
        if conj_type_embedding.shape[0]!=0:
            ex_c_graph[:max(conj_type_batch_idx)+1]=global_mean_pool(expert_conj,conj_type_batch_idx)
        
    ## final_graph_feature
        final_graph_feature = init_graph_feature
        for graph_feat in graph_features:
            final_graph_feature = final_graph_feature + graph_feat

    ## router score based on final_graph_feature
        if not self.self_supervised:
            gate = torch.softmax(self.router_cls(final_graph_feature), dim=-1)  # [B, 5]
            moe_feature = (gate[:, 0:1] * ex_x_graph +
                           gate[:, 1:2] * ex_atom_graph +
                           gate[:, 2:3] * ex_b_graph +
                           gate[:, 3:4] * ex_a_graph +
                           gate[:, 4:5] * ex_c_graph)
            moe_feature = self.norm(moe_feature)
            predict = self.decoder(moe_feature)
            # predict =self.out_act(predict)
            return predict,bt_loss,(atom_type_attention_list,bond_type_attention_list,angle_type_attention_list,conj_type_attention_list)
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
            expert_logits = {}
            if hasattr(self, 'atom_classifier') and expert_atom.shape[0] > 0:
                expert_logits['atom'] = self.atom_classifier(expert_atom)
            if hasattr(self, 'bond_classifier') and expert_bond.shape[0] > 0:
                expert_logits['bond'] = self.bond_classifier(expert_bond)
            if hasattr(self, 'angle_classifier') and expert_angle.shape[0] > 0:
                expert_logits['angle'] = self.angle_classifier(expert_angle)
            if hasattr(self, 'conj_classifier') and expert_conj.shape[0] > 0:
                expert_logits['conj'] = self.conj_classifier(expert_conj)
            return (predict, expert_logits), bt_total

class PretrainedCHHTransformer(HyperGrpahTransformer):
    def __init__(self, atom_dim, bond_dim, angle_dim, batch_size, conj_dim=10, emb_dim=128, dropout=0.4, num_blocks=2, heads=1, ratio=0.5, num_classes=2, self_supervised=False, num_atom_classes=None, num_bond_classes=None, num_angle_classes=None, num_conj_classes=None, pretrained_path=None, map_location='cpu'):
        super().__init__(atom_dim, bond_dim, angle_dim, batch_size, conj_dim, emb_dim, dropout, num_blocks, heads, ratio, num_classes, self_supervised, num_atom_classes, num_bond_classes, num_angle_classes, num_conj_classes)
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path, map_location)

    def _load_pretrained(self, ckpt_path, map_location='cpu'):
        if ckpt_path is None:
            return
        device = torch.device(map_location) if isinstance(map_location, str) else map_location
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                sd = ckpt['state_dict']
            elif 'model_state_dict' in ckpt:
                sd = ckpt['model_state_dict']
            elif 'net' in ckpt and isinstance(ckpt['net'], dict):
                sd = ckpt['net']
            else:
                sd = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
                if len(sd) == 0:
                    sd = ckpt
        else:
            sd = ckpt
        cleaned_sd = {}
        for k, v in sd.items():
            nk = k.replace('module.', '').replace('model.', '')
            cleaned_sd[nk] = v
        with torch.no_grad():
            for emb in self.atom_embedding_list:
                nn.init.xavier_uniform_(emb.weight)
            for emb in self.bond_embedding_list:
                nn.init.xavier_uniform_(emb.weight)
        filtered_sd = {k: v for k, v in cleaned_sd.items() if not (k.startswith('atom_embedding_list.') or k.startswith('bond_embedding_list.'))}
        own = self.state_dict()
        to_load = {}
        missing = []
        mismatch = []
        for k in own.keys():
            if k in filtered_sd:
                if own[k].shape == filtered_sd[k].shape:
                    to_load[k] = filtered_sd[k]
                else:
                    mismatch.append((k, tuple(own[k].shape), tuple(filtered_sd[k].shape)))
            else:
                missing.append(k)
        unexpected = [k for k in filtered_sd.keys() if k not in own]
        self.load_state_dict(to_load, strict=False)
        for k, _, _ in mismatch:
            self._reinit_by_key(k)
        for k in missing:
            if k.startswith('atom_embedding_list.') or k.startswith('bond_embedding_list.'):
                continue
            self._reinit_by_key(k)
        if len(unexpected) > 0:
            print('忽略的预训练权重键:', unexpected)
        if len(mismatch) > 0:
            print('形状不匹配并已重新初始化的键:', [m[0] for m in mismatch])
        print('已完全重新初始化的嵌入模块: atom_embedding_list, bond_embedding_list')
        if len(missing) > 0:
            print('在预训练权重中缺失并已重新初始化的键:', [k for k in missing if not (k.startswith('atom_embedding_list.') or k.startswith('bond_embedding_list.'))])

    def _reinit_by_key(self, key):
        module, param_name = self._resolve_module_and_param(key)
        if module is None:
            return
        with torch.no_grad():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight'):
                    nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                return
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                return
            if isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight'):
                    module.weight.fill_(1.0)
                if hasattr(module, 'bias'):
                    module.bias.zero_()
                return
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
                return
            p = getattr(module, param_name, None)
            if isinstance(p, torch.nn.Parameter):
                nn.init.uniform_(p, -0.1, 0.1)

    def _resolve_module_and_param(self, key):
        parts = key.split('.')
        module = self
        for i, p in enumerate(parts):
            if i == len(parts) - 1:
                return module, p
            if p.isdigit():
                if isinstance(module, (nn.Sequential, nn.ModuleList, list, tuple)):
                    module = module[int(p)]
                else:
                    return None, None
            else:
                if hasattr(module, p):
                    module = getattr(module, p)
                else:
                    return None, None
        return None, None

