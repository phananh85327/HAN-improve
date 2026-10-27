#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HAN (Heterogeneous Graph Attention Network, WWW-2019) for link prediction.

Node-level attention follows the reference TF1 implementation in ../HAN
(models/gat.py HeteGAT_multi): one multi-head GAT layer per meta-path neighbor
graph (paper eq. 1-5). Semantic-level attention follows the PAPER's eq. 7-9
(a shared tanh MLP scores every node's per-path embedding, scores are averaged
over all nodes, softmax over meta-paths, weighted sum). The classification head
is replaced by returning the fused embedding, exactly as the LMSPS models do for
--task link_prediction, so the same BPR / ranking-eval pipeline applies.

Repo additions that are NOT in the paper and are OFF by default:
  - SemanticAttention.tau (softmax temperature; 1.0 = paper),
  - SemanticAttention.force_uniform (train_han.py --beta-warmup; 0 = paper).
Initialisation (ON by default, train_han.py --independent-init turns it off): every
path's GAT starts from the same weights, so the model treats the paths alike until their
graphs separate them (the semantic attention applies one function to every path anyway).
With an independent random start per GAT, the path weights beta followed each path's
POSITION in the list rather than the path: reversing the list dropped a different path,
and without the warm-up the path at a favoured position took all the weight (HAN review
part H3).
Other deliberate departures (pruned/symmetrized meta-path neighbor graphs, a
separate user tower, the repo's default hidden/dropout/lr) are documented in
README.md "Deviations from the paper"; train_han.py --paper-config restores the
paper's Sec. 5.3 hyperparameters.

Implemented in PURE PyTorch -- no dgl at runtime (the dgl wheel in this env is
CPU-only). Two memory traps this file deliberately avoids at ~3M edges/graph:

  1. gathering per-edge messages h[src] * alpha materializes [E, heads, dim]
     (~7 GB per graph, retained for backward) -> aggregation runs as per-head
     sparse matmul instead;
  2. torch.sparse.mm's OWN backward for the sparse argument materializes the
     values-gradient as a DENSE [N, N] matrix (grad_out @ h^T: 90,659^2 floats
     = 30.6 GiB, the exact OOM observed) -> SpmmEdgeValues below keeps the
     sparse.mm forward but hand-writes the backward: grad wrt each edge value
     is just <grad_out[dst], h[src]>, computed in edge chunks (SDDMM), and
     grad wrt h is A^T @ grad_out, another sparse.mm.

Graphs are passed as (src, dst) LongTensor edge lists WITH self-loops already
added (train_han.py does this), which guarantees every node has at least one
incoming edge and the edge softmax is well-defined.

The user side reuses ShallowNodeTower from model.py -- identical to the LMSPS
single-tower design, which isolates the item-encoder comparison (HAN vs LMSPS).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ShallowNodeTower(nn.Module):
    """Simple linear projection of a node type's raw features into the shared
    embedding space -- the user side of the link-prediction pair (inlined from
    the LMSPS repo's model.py so this repo stands alone)."""

    def __init__(self, attr_dim, hidden):
        super(ShallowNodeTower, self).__init__()
        self.proj = nn.Linear(attr_dim, hidden)
        nn.init.xavier_uniform_(self.proj.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.proj.bias)

    def forward(self, attr):
        return self.proj(attr)


class UserEmbeddingTower(nn.Module):
    """Free per-user trainable vectors (one hidden-dim parameter row per user),
    trained end-to-end with the same BPR loss -- the per-user capacity that
    metapath2vec and GAT get implicitly (m2v: its embedding table; GAT: attention
    over the user's own training purchases). Using this on the HAN/LMSPS user
    side levels the user representation across all compared models, so the
    ranking comparison isolates the ITEM encoder (the meta-path question).
    Trade-off vs ShallowNodeTower: transductive -- a user unseen at training
    time has an untrained row, whereas the feature tower at least generalizes
    through the (hash) feature space."""

    def __init__(self, num_users, hidden):
        super().__init__()
        self.emb = nn.Embedding(num_users, hidden)
        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, idx):
        return self.emb(idx)

    def all_users(self):
        return self.emb.weight


SDDMM_CHUNK = 1_000_000  # edges per chunk in the values-gradient computation


class SpmmEdgeValues(torch.autograd.Function):
    """out = A @ h where A is sparse [n, n] built from (indices, values).
    Forward is cuSPARSE spmm; backward avoids torch's dense [n, n] values-grad."""

    @staticmethod
    def forward(ctx, indices, values, n, h):
        ctx.save_for_backward(indices, values, h)
        ctx.n = n
        A = torch.sparse_coo_tensor(indices, values, (n, n))
        return torch.sparse.mm(A, h)

    @staticmethod
    def backward(ctx, grad_out):
        indices, values, h = ctx.saved_tensors
        n = ctx.n
        dst, src = indices[0], indices[1]
        grad_out = grad_out.contiguous()

        # d out / d values[e] = <grad_out[dst_e], h[src_e]> -- chunked SDDMM,
        # transient [chunk, dim] buffers instead of a dense [n, n] matrix
        grad_values = torch.empty_like(values)
        for s in range(0, dst.numel(), SDDMM_CHUNK):
            e = min(s + SDDMM_CHUNK, dst.numel())
            grad_values[s:e] = (grad_out[dst[s:e]] * h[src[s:e]]).sum(-1)

        # d out / d h = A^T @ grad_out (values treated as constants here)
        At = torch.sparse_coo_tensor(torch.stack([src, dst]), values, (n, n))
        grad_h = torch.sparse.mm(At, grad_out)

        return None, grad_values, None, grad_h


class GATLayer(nn.Module):
    """Single-layer multi-head graph attention over one homogeneous graph given as
    an edge list (src -> dst message direction). Matches dgl.nn.GATConv default
    semantics: feat_drop on input, shared projection, LeakyReLU(0.2) edge scores,
    softmax over incoming edges, attn_drop on coefficients, bias, ELU."""

    def __init__(self, in_dim, head_dim, num_heads, feat_drop=0.0, attn_drop=0.0,
                 negative_slope=0.2, activation=True):
        super().__init__()
        # activation=False -> raw concatenated heads + bias (an OUTPUT layer: the
        # original GAT applies no nonlinearity to its last layer; train_gat.py uses
        # this so embeddings are not clamped at -1 by ELU before dot-product scoring)
        self.activation = activation
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.fc = nn.Linear(in_dim, head_dim * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.empty(num_heads, head_dim))
        self.attn_r = nn.Parameter(torch.empty(num_heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads * head_dim))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, edges, feat):
        src, dst = edges
        n = feat.shape[0]
        h = self.fc(self.feat_drop(feat)).view(n, self.num_heads, self.head_dim)
        el = (h * self.attn_l).sum(-1)                     # [N, H]
        er = (h * self.attn_r).sum(-1)                     # [N, H]
        e = self.leaky_relu(el[src] + er[dst])             # [E, H]

        # numerically stable softmax over each dst node's incoming edges
        dst_idx = dst.unsqueeze(1).expand_as(e)
        m = torch.full_like(el, float('-inf'))
        m = m.scatter_reduce(0, dst_idx, e, reduce='amax', include_self=True)
        e = torch.exp(e - m[dst].detach())  # exact: softmax is shift-invariant
        denom = torch.zeros_like(el).index_add_(0, dst, e)
        alpha = self.attn_drop(e / denom[dst].clamp(min=1e-9))  # [E, H]

        indices = torch.stack([dst, src])
        out = torch.stack(
            [SpmmEdgeValues.apply(indices, alpha[:, k].contiguous(), n,
                                  h[:, k].contiguous())
             for k in range(self.num_heads)], dim=1)       # [N, H, head_dim]
        out = out.reshape(n, -1) + self.bias                 # [N, H * head_dim]
        return F.elu(out) if self.activation else out


class SemanticAttention(nn.Module):
    """Semantic-level attention over per-meta-path embeddings (HAN eq. 7-9):
    a shared MLP scores each meta-path, scores are averaged over nodes and
    softmax-normalized into meta-path weights.

    tau: softmax temperature. 1.0 reproduces the paper exactly. The path scores
    are unbounded (they scale with the GAT output magnitudes), so logit gaps of
    ~20 saturate the softmax into a one-hot beta -- after which the losing
    paths' GATs receive (near-)zero gradient and can never recover. A tau > 1
    keeps all paths trainable without changing the argmax preference."""

    def __init__(self, in_dim, hidden_dim=128, tau=1.0):
        super().__init__()
        self.tau = tau
        # set True (train_han.py --beta-warmup) to fuse with a FROZEN uniform beta:
        # every path's GAT then receives equal gradient regardless of its current
        # score, so no path can be starved before it has learned anything. The
        # projection MLP receives no gradient while frozen -- it judges the
        # already-trained GATs fresh once unfrozen.
        self.force_uniform = False
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, z):
        # z: [N, M, D] node embeddings per meta-path
        w = self.project(z).mean(0)            # [M, 1] path importance
        if self.force_uniform:
            beta = torch.full_like(w, 1.0 / z.shape[1])
        else:
            beta = torch.softmax(w / self.tau, dim=0)  # [M, 1]
        return (beta.unsqueeze(0) * z).sum(1), beta.squeeze(-1)


class HAN_LP(nn.Module):
    """One GATLayer per meta-path graph + semantic attention. Returns the fused
    target-node embedding [N, hidden] (hidden = num_heads * per-head dim)."""

    def __init__(self, num_metapaths, in_dim, hidden, num_heads=8,
                 feat_drop=0.2, attn_drop=0.2, semantic_tau=1.0, grad_ckpt=False,
                 symmetric_init=True):
        super().__init__()
        assert hidden % num_heads == 0, 'hidden must be divisible by num-heads'
        self.grad_ckpt = grad_ckpt
        self.gats = nn.ModuleList([
            GATLayer(in_dim, hidden // num_heads, num_heads,
                     feat_drop=feat_drop, attn_drop=attn_drop)
            for _ in range(num_metapaths)
        ])
        self.semantic = SemanticAttention(hidden, tau=semantic_tau)
        if symmetric_init:
            # no path favoured by its place in the list (module docstring): the GATs
            # differ only in the graphs they run on
            for gat in self.gats[1:]:
                gat.load_state_dict(self.gats[0].state_dict())

    def forward(self, graphs, feat):
        # graphs: list of (src, dst) LongTensor pairs (self-loops included);
        # feat: [N, in_dim] raw target-type features
        zs = []
        for gat, g in zip(self.gats, graphs):
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                # recompute this GAT's internals during backward instead of keeping
                # every path's [E, heads] activation tensors alive simultaneously:
                # peak activation memory drops from M GATs to ~1 at ~+30% step time.
                # use_reentrant=False is REQUIRED (feat has no requires_grad; the
                # reentrant variant would silently drop the parameter gradients)
                # and preserves dropout RNG, so results are bit-identical.
                zs.append(checkpoint(gat, g, feat, use_reentrant=False))
            else:
                zs.append(gat(g, feat))
        z = torch.stack(zs, dim=1)                          # [N, M, hidden]
        fused, beta = self.semantic(z)
        return fused, beta
