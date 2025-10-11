import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn import GCN
from einops import rearrange, repeat

class HyMV(nn.Module):
    def __init__(self, **config):
        super(HyMV, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        self_attention_heads = config["SELF_ATTENTION"]["HEADS"]
        self_attention_h_dim = config["SELF_ATTENTION"]["HIDDEN_DIM"]
        self_attention_h_out = config["SELF_ATTENTION"]["OUT_DIM"]

        # intrinsic view
        self.d_iv = IVDrug(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.p_iv = IVProtein(protein_emb_dim, num_filters, kernel_size, protein_padding)

        # WRX-Attn
        self._dim = 356
        drug_avg_len = 256
        num_classes = 1
        self.smiles_attn1 = PreNorm_KQV(self._dim, nn.MultiheadAttention(self._dim, 2, batch_first=True))
        self.smiles_attn1_ln = PreNorm(self._dim, FeedForward(self._dim, self._dim))
        self.smiles_alpha1 = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
        self.prot_attn1 = PreNorm_KQV(self._dim, nn.MultiheadAttention(self._dim, 2, batch_first=True))
        self.prot_attn1_ln = PreNorm(self._dim, FeedForward(self._dim, self._dim))
        self.prot_alpha1 = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
        self.smiles_attn2 = PreNorm_KQV(self._dim, nn.MultiheadAttention(self._dim, 2, batch_first=True))
        self.smiles_attn2_ln = PreNorm(self._dim, FeedForward(self._dim, self._dim))
        self.smiles_alpha2 = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
        self.prot_attn2 = PreNorm_KQV(self._dim, nn.MultiheadAttention(self._dim, 2, batch_first=True))
        self.prot_attn2_ln = PreNorm(self._dim, FeedForward(self._dim, self._dim))
        self.prot_alpha2 = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
        self.prot_pool = nn.AdaptiveAvgPool2d((drug_avg_len, self._dim))
        self.linear3 = FFLayer(self._dim, self._dim)
        self.linear4 = FFLayer(self._dim, self._dim)
        self.mlt = MultiLayerTransformer(self._dim, 2 * drug_avg_len, self._dim, 2, 2, self._dim)
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)
        self.linear5 = FFLayer(self._dim, self._dim)
        self.linear6 = nn.Linear(self._dim, num_classes if num_classes > 2 else 1)
        # mlp_in_dim = self_attention_h_dim
        # self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, d_iv, p_iv, d_snv, p_snv, mode="train"):
        d_iv = self.d_iv(d_iv)
        p_iv = self.p_iv(p_iv)
        d_snv = d_snv.unsqueeze(1).repeat(1, d_iv.shape[1], 1)
        p_snv = p_snv.unsqueeze(1).repeat(1, p_iv.shape[1], 1)
        d = torch.cat([d_iv, d_snv],dim=2)
        p = torch.cat([p_iv, p_snv], dim=2)

        x3 = self.smiles_attn1(d, p, p)[0] + self.smiles_alpha1 * d
        x3 = self.smiles_attn1_ln(x3) + x3
        x4 = self.prot_attn1(p, d, d)[0] + self.prot_alpha1 * p
        x4 = self.prot_attn1_ln(x4) + x4

        x1 = self.smiles_attn2(x3, x4, x4)[0] + self.smiles_alpha2 * x3
        x1 = self.smiles_attn2_ln(x1) + x1
        x2 = self.prot_attn2(x4, x3, x3)[0] + self.prot_alpha2 * x4
        x4 = self.prot_attn2_ln(x4) + x4
        x2 = self.prot_pool(x2)
        x2 = self.linear3(x2)
        x = torch.concat([x1, x2], dim=1)

        x = self.linear4(x)
        x = self.mlt(x)
        x = self.linear5(x)

        score = self.linear6(x)
        #score = torch.nn.functional.sigmoid(score)

        if mode == "train":
            return d, p, 0, score
        elif mode == "eval":
            return d, p, x, score


class IVDrug(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(IVDrug, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.05):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.match_dim = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if x.shape[-1] != out.shape[-1]:
            residual = F.pad(residual, (0, out.shape[-1] - x.shape[-1]))

        residual = self.match_dim(residual)
        out += residual
        out = self.relu(out)
        return out

class IVProtein(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(IVProtein, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.res_block1 = ResidualBlock(in_ch[0], in_ch[1], kernels[0])
        self.res_block2 = ResidualBlock(in_ch[1], in_ch[2], kernels[1])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.res_block1(v)
        v = self.res_block2(v)
        v = v.view(v.size(0), v.size(2), -1)
        return v

class PreNorm_KQV(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm1(q), self.norm2(k), self.norm3(v), **kwargs)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FFLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, act: str = "ReLU", bn: str = "LayerNorm"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = getattr(nn, act)()
        self.bn = getattr(nn, bn)(out_features)

    def forward(self, x):
        return self.bn(self.act(self.linear(x)))

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(F.sigmoid(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MultiLayerTransformer(nn.Module):
    def __init__(self, patch_dim, num_patches, dim, depth, heads, mlp_dim, pool="mean", dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        return x
