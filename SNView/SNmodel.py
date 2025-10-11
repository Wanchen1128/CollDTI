import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv

def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size).apply(init),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False).apply(init)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class HANLayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):

        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GraphConv(in_size, out_size, activation=F.relu).apply(init))
        self.semantic_attention = Attention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                    g, meta_path)
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            new_g = dgl.add_self_loop(new_g)
            semantic_embeddings.append(self.gat_layers[0](new_g, h).flatten(1))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)

class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, dropout, num_heads=1):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.predict = nn.Linear(hidden_size * num_heads, out_size, bias=False).apply(init)
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads, dropout)
        )

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)


class NView(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(NView, self).__init__()
        self.sum_layers = nn.ModuleList()

        for i in range(0, len(all_meta_paths)):
            self.sum_layers.append(
                HAN(all_meta_paths[i], in_size[i], hidden_size[i], out_size[i], dropout))


    def forward(self, s_g, s_h_1, s_h_2):
        h1  = self.sum_layers[0](s_g[0], s_h_1)
        h2  = self.sum_layers[1](s_g[1], s_h_2)
        return h1, h2


class SNView(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, hidden_size1, out_size, dropout):
        super(SNView, self).__init__()
        self.NView = NView(all_meta_paths, in_size, hidden_size, out_size, dropout)
        #加入的
        hidden_size2 = int(hidden_size1/2)
        hidden_size3 = int(hidden_size1*2)
        self.a = nn.Parameter(torch.zeros(size=(hidden_size1, 1)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, graph, h, d_sv, p_sv):
        # Neighbor View
        d_nv, p_nv = self.NView(graph, h[0], h[1])
        d_nv = d_nv.to(torch.float32)
        p_nv = p_nv.to(torch.float32)
        # Similarity View
        d_sv = d_sv.to(dtype=torch.float32, device=d_nv.device)
        p_sv = p_sv.to(dtype=torch.float32, device=p_nv.device)

        d_snv =  torch.cat([d_nv, d_sv], dim=1)
        p_snv = torch.cat([p_nv, p_sv], dim=1)

        return d_snv, p_snv