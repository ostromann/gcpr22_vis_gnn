from re import X
import torch
import torchvision
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class MyFCModule(torch.nn.Module):
    def __init__(self, in_features, hidden_units, out_features):
        super().__init__()

        self.fc1 = torch.nn.Linear(in_features, hidden_units)
        self.fc2 = torch.nn.Linear(hidden_units, out_features)

    def forward(self, data):
        x = data.x
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class MyGraphSAGE_WithImage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, embed_dim, n_classes, dropout_rate=0.5):
        super().__init__()

        self.gcn1 = SAGEConv(in_channels, hidden_channels)
        self.gcn2 = SAGEConv(hidden_channels, embed_dim)
        self.dropout_rate = dropout_rate
        self.fc = torch.nn.Linear(embed_dim, n_classes)

    def forward(self, data):
        x, img, edge_index = data.x, data.img, data.edge_index

        x = torch.cat([img, x], 1)

        x = self.gcn1(x, edge_index)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(x)

        x = self.gcn2(x, edge_index)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(x)

        x = self.fc(x)

        return F.log_softmax(x, dim=1)