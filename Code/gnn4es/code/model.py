# construct the model
import torch
from torch import nn
import numpy as np
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_
from torch_geometric.utils import negative_sampling, add_self_loops
from torch_geometric.data import Data
from torch_geometric.nn import Sequential
from disen_conv import DisenConv
from torch_geometric.utils import to_undirected


class MLP(nn.Module):
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def __init__(
        self,
        input_size,
        output_size,
        layer_sizes,
        num_layers,
        negative_slope=0.2,
        bias=False,
        is_leaky_relu=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        # 添加隐藏层
        last_size = input_size
        for i in range(num_layers):
            layer = nn.Linear(last_size, layer_sizes[i], bias)
            self.layers.append(layer)
            last_size = layer_sizes[i]
        # 添加输出层
        self.output_layer = nn.Linear(last_size, output_size, bias)
        self.negative_slope = negative_slope
        self.is_leaky_relu = is_leaky_relu
        self.apply(self._init_weights)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if self.is_leaky_relu:
                x = nn.LeakyReLU(negative_slope=self.negative_slope)(x)
            else:
                x = nn.ReLU()(x)
        output = self.output_layer(x)
        return output


class DCLModel(torch.nn.Module):
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def __init__(self, args, in_channels, out_channels, training):
        super().__init__()
        self.args = args
        if self.args.backbone == "GCN":
            backbone = GCNConv(in_channels * 2, out_channels)
        else:
            backbone = DisenConv(in_channels * 2, out_channels, K=5)
        self.communication_layer = Sequential(
            "x, edge_index",
            [
                (nn.Linear(in_channels, in_channels * 2), "x -> x"),
                nn.Dropout(p=0.2),
                (backbone, "x, edge_index -> x"),
            ],
        )
        # self.communication_layer = DisenConv(in_channels,out_channels,K=5)
        output_layers = [
            2 ** (i + 6) for i in range(self.args.dcl_output_layers, 0, -1)
        ]  # 2**9,2**8,2**7
        self.output_layer = MLP(
            out_channels * 2, out_channels, output_layers, len(output_layers)
        )
        self.apply(self._init_weights)

    def encode(self, desc_graph):
        desc_x, desc_edge_index = desc_graph.x, desc_graph.edge_index
        desc_x = self.communication_layer(desc_x, desc_edge_index)
        return desc_x

    def decode(self, z, edge_label_index):
        edge_feature = torch.cat(
            (z[edge_label_index[0]], z[edge_label_index[1]]), dim=-1
        )
        link_score = (self.output_layer(edge_feature)).sum(-1)
        return link_score


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor, postive, negative, margin):
        # Compute Euclidean distances
        pos_dist = F.pairwise_distance(anchor, postive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        # Compute triplet loss
        loss = torch.mean(F.relu(pos_dist - neg_dist + margin))
        return loss


class CatEmbedding(nn.Module):
    def __init__(self, args, hidden_size):
        super(CatEmbedding, self).__init__()
        self.args = args
        input_layer_size = [256, 128, 64]
        self.trans_layer = MLP(
            hidden_size * 2,
            hidden_size,
            layer_sizes=input_layer_size,
            num_layers=len(input_layer_size),
        )

    def forward(self, x):
        label_embed = x[-1].clone()
        to_cat = x[:-1].clone()
        to_cat = torch.cat((to_cat, label_embed.expand(to_cat.shape[0], -1)), dim=-1)
        x[:-1] = self.trans_layer(to_cat)
        return x


class NormalizeLayer(nn.Module):
    def __init__(self, p=2, dim=1):
        super(NormalizeLayer, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)


class Self_Calibration_Loss(nn.Module):
    def __init__(self, hidden_size):
        super(Self_Calibration_Loss, self).__init__()
        self.linear_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_embedding, sample_centroids_dist, K=5):
        hidden_embedding = F.relu(self.linear_layer(hidden_embedding))
        loss = (hidden_embedding.view(-1) - sample_centroids_dist).square().mean() / (
            2 * K
        )
        return loss


class GAE4ES(nn.Module):
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1.0 / self.input_channel)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def __init__(self, args, input_size, hidden_size, out_size=64, training=True):
        super(GAE4ES, self).__init__()
        input_layer_size = [512, 256, 128]
        self.training = training
        self.dcl_model = DCLModel(args, hidden_size, out_size, training)
        self.embedding_transfer = nn.Sequential(
            MLP(
                input_size,
                hidden_size,
                layer_sizes=input_layer_size,
                num_layers=len(input_layer_size),
            ),
            NormalizeLayer(),
        )
        self.args = args
        self.triplet_loss = TripletLoss()
        self.calibration_loss = Self_Calibration_Loss(hidden_size)
        self.apply(self._init_weights)

    def forward(self, graph):
        x = graph.x
        x = self.embedding_transfer(x)
        if self.training:
            anchor = x[-1]
            triplet_loss = self.triplet_loss(
                anchor,
                x[graph.postive_index],
                x[graph.negative_index],
                self.args.triplet_margin,
            )
            index_to_cluster_id = graph.index_to_cluster_id
            centroids = graph.centroids
            link_predict_loss = 0
            calibration_loss = 0
            for link_graph in graph.link_graph_list:
                desc_graph = Data(x=x, edge_index=link_graph.edge_index)
                desc_x = self.dcl_model.encode(desc_graph)
                desc_index = link_graph.desc_index
                centroids_embedding = torch.stack(
                    [centroids[index_to_cluster_id[index]] for index in desc_index]
                ).to(self.args.device)
                sample_centroids_dist = torch.norm(
                    centroids_embedding - graph.x[desc_index], dim=1
                )
                calibration_loss += self.calibration_loss(
                    desc_x[desc_index], sample_centroids_dist
                )
                eval_edge = link_graph.eval_edge
                link_predict_loss += F.binary_cross_entropy_with_logits(
                    self.dcl_model.decode(desc_x, eval_edge), link_graph.eval_label
                )
            link_predict_loss = link_predict_loss / len(graph.link_graph_list)
            calibration_loss = calibration_loss / len(graph.link_graph_list)
            total_loss = (
                triplet_loss
                + (1 + self.args.alpha) * link_predict_loss
                + calibration_loss
            )
            return total_loss, (
                triplet_loss,
                (1 + self.args.alpha) * link_predict_loss,
                calibration_loss,
            )
        else:
            label_embedding = x[-1]
            sim_score = torch.matmul(x[:-1], label_embedding)
            edge_index = graph.edge_index
            eval_edge = graph.eval_edge
            desc_graph = Data(x=x, edge_index=edge_index)
            desc_x = self.dcl_model.encode(desc_graph)
            link_score = self.dcl_model.decode(desc_x, graph.eval_edge)
            return sim_score, link_score
