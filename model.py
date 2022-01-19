import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
import torch.nn as nn

#GCN models with 2 layers
class Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,device='cpu'):
        super(Model, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        #self.conv2_bn = nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1,
        #        affine=True, track_running_stats=True)
        #self.fc_block1 = nn.Linear(20, 10)
        #self.fc_block2 = nn.Linear(10, 1)

    def forward(self, x: Tensor, edge_index:Tensor):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]

        #x = x.to(device)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def forward1(self, x: Tensor, edge_index:Tensor, device,
            return_graph_embedding=False):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]

        x.to(device)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        sumpool = SumPooling()
        out = sumpool(g, x)
        if return_graph_embedding:
            return out

        #out = function.dropout(out, p=0.2, training=self.training)
        out = self.fc_block1(out)
        out = function.leaky_relu(out)
        out = self.fc_block2(out)
        return out
