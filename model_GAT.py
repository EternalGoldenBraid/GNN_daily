import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.nn as nn

#GCN models with 2 layers
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
        super(GATModel, self).__init__()
        self.attention= GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)

        self.dropout = dropout
        self.out_att = GATConv(
                hidden_channels * heads, out_channels, heads=heads, dropout=dropout, concat=False)

    def forward(self, x: Tensor, edge_index:Tensor):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.attention(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.out_att(x, edge_index)
        x = F.elu(x)
        return F.log_softmax(x, dim=1)

    def reset_weights():
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer {layer}')
                layer.reset_parameters()
