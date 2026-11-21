"""Models for the leakage-audited deep cross-sectional benchmark.

All models consume one day's cross-section x of shape (B, N, T, C) plus a
validity mask (B, N) and output scores (B, N).  B = number of days per batch,
N = fixed fold universe, T = lookback window, C = feature count.

Implemented:
  RidgeMLP   : per-stock MLP on flattened window (no cross-stock information)
  GRU        : per-stock GRU over the window
  MarketGRU  : GRU with MASTER-style market-conditioned feature gating
               alpha_t = C * softmax(W m_t + b),  x~ = x * alpha  (same gate for all stocks on day t)
               gate='feat'      -> MASTER paper's gate only
               gate='feat_hidden' -> + an extra hidden-state gate (this project's addition)
  StockMixer : faithful re-implementation of the AAAI-24 architecture
               (TriU time mixing + multi-scale conv + indicator mixing + NoGraph stock mixing)
  SMNoMix    : StockMixer with the cross-stock mixing block removed (ablation)
  SMMarketPool: StockMixer with stock mixing replaced by permutation-invariant
               market mean-pooling (variable-universe-safe variant)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

GELU = nn.GELU()


# --------------------------------------------------------------------- losses
def sm_loss(pred, y, mask, alpha=0.1):
    """Masked MSE + pairwise ranking hinge, as in StockMixer's get_loss."""
    m = mask.float()
    denom = m.sum().clamp(min=1.0)
    reg = (((pred - y) ** 2) * m).sum() / denom
    pre = pred.unsqueeze(2) - pred.unsqueeze(1)
    gt = y.unsqueeze(1) - y.unsqueeze(2)          # gt_j - gt_i
    mp = m.unsqueeze(2) * m.unsqueeze(1)
    rank = (F.relu(pre * gt) * mp).sum() / mp.sum().clamp(min=1.0)
    return reg + alpha * rank


def mse_loss(pred, y, mask):
    m = mask.float()
    return (((pred - y) ** 2) * m).sum() / m.sum().clamp(min=1.0)


# ------------------------------------------------------------------- baselines
class MLP(nn.Module):
    def __init__(self, T, C, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(T * C, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, mask):
        B, N, T, C = x.shape
        z = self.net(x.reshape(B * N, T * C))
        return z.reshape(B, N)


class GRU(nn.Module):
    def __init__(self, C, hidden=64):
        super().__init__()
        self.gru = nn.GRU(C, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        B, N, T, C = x.shape
        out, _ = self.gru(x.reshape(B * N, T, C))
        h = out[:, -1]
        return self.head(h).reshape(B, N)


class MarketGRU(nn.Module):
    """GRU whose input features are gated by a market-state vector (MASTER-style).

    alpha(m_t) = C * softmax_beta(W m_t + b)   (shared by all stocks on day t)
    x~_n,t = x_n,t * alpha
    Optionally an output gate  g = sigmoid(W_g [m_t; h_T]), h <- h * (1+g).
    """

    def __init__(self, C, mdim, hidden=64, gate="feat", beta=2.0):
        super().__init__()
        self.C, self.gate, self.beta = C, gate, beta
        self.gate_fc = nn.Linear(mdim, C)
        self.gru = nn.GRU(C, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.out_gate = nn.Linear(2 * hidden, hidden) if gate in ("feat_hidden", "hidden") else None
        self.mproj = nn.Linear(mdim, hidden) if gate in ("feat_hidden", "hidden") else None

    def forward(self, x, mask, mkt):
        B, N, T, C = x.shape
        a = self.C * torch.softmax(self.beta * self.gate_fc(mkt), dim=-1)   # (B,C)
        xg = x * a.reshape(B, 1, 1, C)
        out, _ = self.gru(xg.reshape(B * N, T, C))
        h = out[:, -1].reshape(B, N, -1)
        if self.gate in ("feat_hidden", "hidden"):
            mm = self.mproj(mkt).unsqueeze(1).expand(-1, N, -1)             # (B,N,H)
            g = torch.sigmoid(self.out_gate(torch.cat([mm, h], dim=-1)))
            h = h * (1.0 + g)
        return self.head(h).squeeze(-1)


# ----------------------------------------------------------------- StockMixer
class TriU(nn.Module):
    """Causal time mixing: output position t is a linear function of positions 0..t."""

    def __init__(self, T):
        super().__init__()
        self.T = T
        self.triU = nn.ParameterList([nn.Linear(i + 1, 1) for i in range(T)])

    def forward(self, x):                      # (M, C, T)
        out = [self.triU[0](x[:, :, 0:1])]
        for i in range(1, self.T):
            out.append(self.triU[i](x[:, :, : i + 1]))
        return torch.cat(out, dim=-1)          # (M, C, T)


class MixerBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.d1, self.d2 = nn.Linear(d, h), nn.Linear(h, d)

    def forward(self, x):
        return self.d2(GELU(self.d1(x)))


class Mixer2dTriU(nn.Module):
    def __init__(self, T, C):
        super().__init__()
        self.ln1 = nn.LayerNorm([T, C])
        self.ln2 = nn.LayerNorm([T, C])
        self.time = TriU(T)
        self.chan = MixerBlock(C, C)

    def forward(self, x):                      # (M, T, C)
        z = self.ln1(x).permute(0, 2, 1)       # (M, C, T)
        z = self.time(z).permute(0, 2, 1)      # (M, T, C)
        z = self.ln2(z + x)
        return z + self.chan(z)


class MultTime2dMixer(nn.Module):
    def __init__(self, T, C):
        super().__init__()
        self.main = Mixer2dTriU(T, C)
        self.scale = Mixer2dTriU(T // 2, C)

    def forward(self, x, conv_x):
        return torch.cat([x, self.main(x), self.scale(conv_x)], dim=1)   # (M, 1.5T, C)


class NoGraphMixer(nn.Module):
    """LayerNorm over stock axis + Linear(N->m) -> Hardswish -> Linear(m->N), masked."""

    def __init__(self, stocks, hidden, masked=True):
        super().__init__()
        self.d1 = nn.Linear(stocks, hidden)
        self.d2 = nn.Linear(hidden, stocks)
        self.masked = masked

    def forward(self, x, mask):                # x (B, N, L), mask (B, N)
        m = mask.float().unsqueeze(-1)         # (B,N,1)
        if self.masked:
            xm = x * m
            cnt = m.sum(1).clamp(min=1.0)      # (B,1)
            mu = xm.sum(1) / cnt               # (B,L)
            var = (((x - mu.unsqueeze(1)) ** 2) * m).sum(1) / cnt
            xn = ((x - mu.unsqueeze(1)) / torch.sqrt(var.unsqueeze(1) + 1e-5)) * m
        else:
            # 官方代码原样: 统计量用**全部** N 只(无效个股填 0 也参与统计)
            cnt = float(x.shape[1])
            mu = x.sum(1) / cnt
            var = ((x - mu.unsqueeze(1)) ** 2).sum(1) / cnt
            xn = (x - mu.unsqueeze(1)) / torch.sqrt(var.unsqueeze(1) + 1e-5)
        z = xn.permute(0, 2, 1)                # (B, L, N)
        z = F.hardswish(self.d1(z))            # (B, L, m)
        z = self.d2(z)                         # (B, L, N)
        return z.permute(0, 2, 1) * m          # (B, N, L)


class MarketPoolMixer(nn.Module):
    """Permutation-invariant replacement for NoGraphMixer.

    h_n' = h_n + W2(MLP(mean_n h_n)) ;  works with a variable stock count.
    """

    def __init__(self, L, hidden=20):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(L, hidden), nn.Hardswish(), nn.Linear(hidden, L))

    def forward(self, x, mask):                # x (B,N,L)
        m = mask.float().unsqueeze(-1)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)   # (B,L)
        return x + self.mlp(pooled).unsqueeze(1)


class StockMixer(nn.Module):
    def __init__(self, stocks, T, C, market=20, mix="graph"):
        super().__init__()
        self.C, self.T = C, T
        self.mixer = MultTime2dMixer(T, C)
        self.conv = nn.Conv1d(C, C, kernel_size=2, stride=2)
        self.channel_fc = nn.Linear(C, 1)
        self.time_fc = nn.Linear(T + T + T // 2, 1)
        self.time_fc_ = nn.Linear(T + T + T // 2, 1)
        if mix in ("graph", "graph_nomask"):
            self.stock_mixer = NoGraphMixer(stocks, market,
                                            masked=(mix != "graph_nomask"))
        elif mix == "pool":
            self.stock_mixer = MarketPoolMixer(T + T + T // 2, market)
        else:                                  # "none": ablation
            self.stock_mixer = None

    def forward(self, x, mask):                # x (B,N,T,C)
        B, N, T, C = x.shape
        xf = x.reshape(B * N, T, C)
        cx = self.conv(xf.permute(0, 2, 1)).permute(0, 2, 1)     # (B*N, T/2, C)
        z = self.mixer(xf, cx)                                   # (B*N, 1.5T, C)
        z = self.channel_fc(z).squeeze(-1)                       # (B*N, 1.5T)
        z = z.reshape(B, N, -1)
        if self.stock_mixer is None:
            h = z
        else:
            h = self.stock_mixer(z, mask)
        return (self.time_fc(z) + self.time_fc_(h)).squeeze(-1)  # (B,N)


class HIST(nn.Module):
    """HIST (概念图横截面模型), 按 lit spec §4 复现.

    结构:
      1) 2 层 GRU 个股编码器 -> 个股表示 x (B,N,H)
      2) 预定义概念模块: 概念嵌入 = 成员个股聚合 (LeakyReLU FFN),
         再用余弦相似度 softmax 在**全体个股**上做修正 (软成员关系),
         最后按硬成员关系回写到个股
      3) 隐藏概念模块: 概念初值取自个股嵌入, 每个个股连到最相似的隐藏概念
         (余弦, 排除自身), 概念再聚合其成员, LeakyReLU 后回写
      4) 个股信息模块 + 双重残差(backcast/forecast)堆叠 -> 预测

    mem: (N, M) 二值 股票×概念 成员矩阵 (预定义概念), 固定股票池下按折构造。
    """

    def __init__(self, C, T, mem, H=64, hidden=64):
        super().__init__()
        H = hidden
        self.gru = nn.GRU(C, H, num_layers=2, batch_first=True)
        self.register_buffer("mem", torch.as_tensor(mem, dtype=torch.float32))
        self.M = mem.shape[1]
        # 预定义概念
        self.con_emb = nn.Linear(H, H)
        self.con_corr = nn.Linear(H, H)
        self.con_msg = nn.Linear(H, H)
        self.con_back = nn.Linear(H, H)
        # 隐藏概念
        self.hid_init = nn.Linear(H, H)
        self.hid_agg = nn.Linear(H, H)
        self.hid_msg = nn.Linear(H, H)
        self.hid_back = nn.Linear(H, H)
        # 个股信息模块
        self.ind_fc = nn.Linear(H, H)
        self.ind_back = nn.Linear(H, H)
        self.head = nn.Linear(H, 1)

    def forward(self, x, mask):
        B, N, T, C = x.shape
        h, _ = self.gru(x.reshape(B * N, T, C))
        r = h[:, -1].reshape(B, N, -1)
        m = mask.float().unsqueeze(-1)
        r = r * m

        # --- 个股信息模块 (先做, 其余模块在其残差上工作) ---
        f_ind = self.ind_fc(r)
        r = r - self.ind_back(r)

        # --- 预定义概念模块 ---
        Mem = self.mem.unsqueeze(0).expand(B, -1, -1)                  # (B,N,M)
        cnt = (Mem * m).sum(1).clamp(min=1.0)                        # (B,M)
        agg = torch.einsum("bnm,bnh->bmh", Mem, r) / cnt.unsqueeze(-1)
        e = F.leaky_relu(self.con_emb(agg))                           # (B,M,H)
        cos = F.normalize(r, dim=-1) @ F.normalize(e, dim=-1).transpose(1, 2)
        alpha = torch.softmax(cos, dim=1)                             # 沿个股维 softmax
        e2 = torch.einsum("bnm,bnh->bmh", alpha, r)                   # 修正后的概念嵌入
        r = r + torch.einsum("bnm,bmh->bnh", Mem, self.con_corr(e2))
        f_con = self.con_msg(r)
        r = r - self.con_back(r)

        # --- 隐藏概念模块 ---
        hid = self.hid_init(r)                                        # (B,N,H)
        cos2 = F.normalize(r, dim=-1) @ F.normalize(hid, dim=-1).transpose(1, 2)
        eye = torch.eye(N, device=r.device, dtype=r.dtype)
        cos2 = cos2 - 1e4 * eye                                      # 排除自身
        j = cos2.argmax(dim=2)                                       # (B,N)
        onehot = F.one_hot(j, N).to(r.dtype)                          # (B,N,N)
        cnt2 = onehot.sum(1).clamp(min=1.0)
        hagg = torch.einsum("bnk,bnh->bkh", onehot, r) / cnt2.unsqueeze(-1)
        hagg = F.leaky_relu(self.hid_agg(hagg))
        r = r + torch.einsum("bnk,bkh->bnh", onehot, hagg)
        f_hid = self.hid_msg(r)
        r = r - self.hid_back(r)

        out = (f_ind + f_con + f_hid) * m
        return self.head(out).squeeze(-1)                            # (B,N)


def build_model(name, stocks, T, C, mdim, hidden=64, market=20, extra=None):
    extra = extra or {}
    if name == "mlp":
        return MLP(T, C)
    if name == "gru":
        return GRU(C, hidden)
    if name == "mgru":
        return MarketGRU(C, mdim, hidden, gate="feat_hidden")
    if name == "mgru_feat":
        # pure MASTER: feature gate only (no extra hidden gate), so that the
        # mgru-vs-gru gap can be attributed to the market conditioning itself
        return MarketGRU(C, mdim, hidden, gate="feat")
    if name == "hist":
        return HIST(C, T, extra["mem"], hidden=extra.get("H", 64))
    if name == "sm":
        return StockMixer(stocks, T, C, market, mix="graph")
    if name == "sm_nomix":
        return StockMixer(stocks, T, C, market, mix="none")
    if name == "sm_pool":
        return StockMixer(stocks, T, C, market, mix="pool")
    if name == "sm_nomask":
        # 官方代码原样: 股票混合层 LayerNorm 的统计量用全部 N 只(含无效个股)
        return StockMixer(stocks, T, C, market, mix="graph_nomask")
    raise ValueError(name)
