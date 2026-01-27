import torch
import torch.nn as nn
from torch_scatter import scatter_add,scatter_mean
import torch.nn.functional as F
from torch_scatter import scatter_add,scatter_mean
from torch import Tensor
import math
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

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

class E_HypergraphConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(E_HypergraphConv, self).__init__()
        self.epsilon = 1e-8
        self.in_channels = in_channels
        self.out_channels = out_channels
## edge_mlp ###        
        self.edge_mlp = nn.Sequential(
            nn.Linear(3*out_channels , out_channels),
            nn.SiLU(),
            nn.Linear(out_channels , out_channels ),
            nn.SiLU())
        layer = nn.Linear(out_channels, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
## coord_mlp ###
        coord_mlp = []
        coord_mlp.append(nn.Linear(out_channels, out_channels))
        coord_mlp.append(nn.SiLU())
        coord_mlp.append(layer)
        coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)
        self.lin = nn.Linear(in_channels, out_channels)
        self.node_mlp = nn.Sequential(
            nn.Linear(out_channels*2, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels))
        self.radial_trans=RBF(1,out_channels)

    def coord2radial(self, hyperedge_index, coord):
        row, col = hyperedge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        norm = torch.sqrt(radial).detach() + self.epsilon
        coord_diff = coord_diff / norm

        return self.radial_trans(radial), coord_diff

    def edge_model(self, source, target, radial, edge_attr=None):
        if edge_attr is None:  
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat,coords_agg='mean'):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % coords_agg)
        coord = coord + agg
        return coord
    
    def node_model(self, x, edge_index, edge_attr, node_attr=None):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        out = x + out
        return out, agg

    def forward(self, x: Tensor, hyperedge_feature: Tensor,hyperedge_index: Tensor,node_coord: Tensor,
                ) -> Tensor:
        reindex_hyperedge=hyperedge_index.clone()
        reindex_hyperedge[1]=max(reindex_hyperedge[0])+1+reindex_hyperedge[1]
        node_indices, hyperedge_indices = reindex_hyperedge[0], reindex_hyperedge[1]
        hyperedge_coord = scatter_mean(node_coord[node_indices], hyperedge_indices , dim=0)

        x = self.lin(x)
        h=torch.cat([x,hyperedge_feature],dim=0)
        coord=torch.cat([node_coord,hyperedge_coord],dim=0)
        reverse_index=torch.cat([reindex_hyperedge[1][None,:],reindex_hyperedge[0][None,:]],dim=0)
        edge_index=torch.cat([reindex_hyperedge,reverse_index],dim=1)
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, _ = self.node_model(h, edge_index, edge_feat)
        node_h,edge_h=h[:x.size(0)],h[x.size(0):]
        return node_h,edge_h,coord