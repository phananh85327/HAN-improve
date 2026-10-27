"""The HAN model (HAN review part H3): the hand-written sparse backward, the GAT layer against
DGL, the semantic attention against the paper, and the order-invariant initialisation.

    python tests/test_han_model.py      (or: python -m pytest tests)
"""
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import model_han  # noqa: E402
from model_han import GATLayer, HAN_LP, SemanticAttention, SpmmEdgeValues  # noqa: E402


def _graph(n, e, seed):
    g = torch.Generator().manual_seed(seed)
    src, dst = torch.randint(0, n, (e,), generator=g), torch.randint(0, n, (e,), generator=g)
    keep = src != dst
    pairs = torch.unique(torch.stack([src[keep], dst[keep]], 1), dim=0)
    loop = torch.arange(n)
    return torch.cat([pairs[:, 0], loop]), torch.cat([pairs[:, 1], loop])


def test_sparse_backward_gradcheck():
    torch.manual_seed(0)
    n = 7
    src, dst = _graph(n, 25, 0)
    src, dst = torch.cat([src, src[:3]]), torch.cat([dst, dst[:3]])          # duplicate edges too
    idx = torch.stack([dst, src])
    vals = torch.rand(idx.shape[1], dtype=torch.double, requires_grad=True)
    h = torch.randn(n, 3, dtype=torch.double, requires_grad=True)
    chunk = model_han.SDDMM_CHUNK
    try:
        for model_han.SDDMM_CHUNK in (chunk, 4):                             # one chunk, many chunks
            assert torch.autograd.gradcheck(lambda v, x: SpmmEdgeValues.apply(idx, v, n, x), (vals, h))
    finally:
        model_han.SDDMM_CHUNK = chunk


def test_gat_layer_matches_dgl():
    try:
        import dgl
        from dgl.nn import GATConv
    except ImportError:
        print('  (skipped: dgl not installed)')
        return
    torch.manual_seed(1)
    n, heads, hd, ind = 50, 4, 5, 6
    src, dst = _graph(n, 300, 1)
    x = torch.randn(n, ind)
    mine = GATLayer(ind, hd, heads)
    ref = GATConv(ind, hd, heads, feat_drop=0., attn_drop=0., negative_slope=0.2, activation=None, bias=True)
    with torch.no_grad():
        ref.fc.weight.copy_(mine.fc.weight)
        ref.attn_l.copy_(mine.attn_l.view(1, heads, hd))
        ref.attn_r.copy_(mine.attn_r.view(1, heads, hd))
        torch.nn.init.normal_(mine.bias)
        ref.bias.copy_(mine.bias)
    mine.eval(); ref.eval()
    out_ref = F.elu(ref(dgl.graph((src, dst), num_nodes=n), x).reshape(n, -1))
    assert torch.allclose(out_ref, mine((src, dst), x), atol=1e-5)


def test_semantic_attention_is_the_papers():
    torch.manual_seed(2)
    z = torch.randn(20, 3, 8)
    sa = SemanticAttention(8, hidden_dim=5)
    torch.nn.init.normal_(sa.project[2].weight)
    fused, beta = sa(z)
    w = torch.stack([(sa.project[2].weight @ torch.tanh(sa.project[0](z[:, m])).T).mean() for m in range(3)])
    assert torch.allclose(beta, torch.softmax(w, 0), atol=1e-6)
    assert torch.allclose(fused, (beta[None, :, None] * z).sum(1), atol=1e-6)


def test_initialisation_ignores_the_order_of_the_paths():
    n, ind = 40, 6
    graphs = [_graph(n, 150, s) for s in (3, 4, 5)]
    x = torch.randn(n, ind)
    for symmetric in (True, False):
        torch.manual_seed(7)
        model = HAN_LP(3, ind, 16, num_heads=4, feat_drop=0., attn_drop=0., symmetric_init=symmetric).eval()
        fused, beta = model(graphs, x)
        fused_rot, beta_rot = model(graphs[1:] + graphs[:1], x)
        same = torch.allclose(fused, fused_rot, atol=1e-5) and torch.allclose(beta, beta_rot[[2, 0, 1]], atol=1e-6)
        assert same == symmetric, symmetric


def test_symmetric_gats_separate_once_trained():
    n, ind = 40, 6
    graphs = [_graph(n, 150, s) for s in (3, 4)]
    torch.manual_seed(8)
    model = HAN_LP(2, ind, 16, num_heads=4)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    x, target = torch.randn(n, ind), torch.randn(n, 16)
    for _ in range(3):
        opt.zero_grad()
        fused, _ = model(graphs, x)
        F.mse_loss(fused, target).backward()
        opt.step()
    a, b = model.gats[0].fc.weight, model.gats[1].fc.weight
    assert not torch.allclose(a, b)                     # different graphs, different gradients


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
