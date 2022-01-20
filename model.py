import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

#GCN models with 2 layers
class ModelDeep(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,device='cpu'):
        super(ModelDeep, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.conv2_bn = nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1,
                affine=True, track_running_stats=True)
        self.fc_block1 = nn.Linear(out_channels, 10)
        self.fc_block2 = nn.Linear(10, 5)

    def forward(self, x: Tensor, edge_index:Tensor):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        x = F.leaky_relu(x)
        x = self.conv2_bn(x)

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.leaky_relu(self.fc_block1(x))
        x = self.fc_block2(x)
        return x
    def reset_weights():
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer {layer}')
                layer.reset_parameters()

class ModelShallow(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,device='cpu'):
        super(ModelShallow, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index:Tensor):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return F.log_softmax(x,dim=1)

    def reset_weights():
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer {layer}')
                layer.reset_parameters()

#hclass Model(nn.Module):
#h    def __init__(self, out_channels, k=20, aggr='max'):
#h        super().__init__()
#h
#h        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
#h        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
#h        self.lin1 = Linear(128 + 64, 1024)
#h
#h        self.mlp = MLP([1024, 512, 256, out_channels], dropout=0.5,
#h        batch_norm=False)
#h
#h    def forward(self, data):
#h        pos, batch = data.pos, data.batch
#h        x1 = self.conv1(pos, batch)
#h        x2 = self.conv2(x1, batch)
#h        out = self.lin1(torch.cat([x1, x2], dim=1))
#h        out = global_max_pool(out, batch)
#h        out = self.mlp(out)
#h
#h        return F.log_softmax(out, dim=1)
