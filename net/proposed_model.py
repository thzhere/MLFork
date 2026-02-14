class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x
    
class FreeConvSS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        #self.conv2d = nn.Conv2d(
           # in_channels=self.d_inner,
            #out_channels=self.d_inner,
            #groups=self.d_inner,
           # bias=conv_bias,
           # kernel_size=d_conv,
           # padding=(d_conv - 1) // 2,
           # **factory_kwargs,
        #)
        #self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        #x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class FreeVSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = FreeConvSS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x
    
class ResMambaBlock(nn.Module):
    def __init__(self, in_c, k_size = 3):
      super().__init__()
      self.in_c = in_c
      self.conv = nn.Conv2d(in_c, in_c, k_size, stride=1, padding='same', dilation=1, groups=in_c, bias=True, padding_mode='zeros')
      self.ins_norm = nn.InstanceNorm2d(in_c, affine=True)
      self.act = nn.LeakyReLU(negative_slope=0.01)
      self.block = FreeVSSBlock(hidden_dim = in_c)
      # self.block = nn.Conv2d(in_c, in_c, k_size, stride=1, padding='same')
      self.scale = nn.Parameter(torch.ones(1))
    def forward(self, x):

      skip = x

      x = self.conv(x)
      x = x.permute(0, 2, 3, 1)
      x = self.block(x)
      x = x.permute(0, 3, 1, 2)
      x = self.act(self.ins_norm(x))
      return x + skip * self.scale
    
class EncoderBlock_Mamba(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, mixer_kernel=(3,3)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = mixer_kernel )
        self.bn = nn.BatchNorm2d(in_c)
        self.act = nn.ReLU()
        self.resmamba = ResMambaBlock(in_c)

    def forward(self, x):
        x = self.resmamba(x)
        x = self.dw(x)
        x = self.act(self.bn(x))
        return x
    
class SB(nn.Module):
    def __init__(self, features, G=64, d = 64):

        super(SB, self).__init__()
        self.features = features
        self.ln1 = nn.LayerNorm([features,16,16])
        self.ln2 = nn.LayerNorm([features,16,16])

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=True),
                                nn.BatchNorm2d(d).eval(),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(
                 nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
        self.conv_end1 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn = nn.BatchNorm2d(features)
        self.act = nn.ReLU()
    def forward(self, f1, f2):
        batch_size = f1.shape[0]

        # f1 = f1.permute(0,2,3,1)
        # f2 = f2.permute(0,2,3,1)
        # f1 = self.ln1(f1)
        # f2 = self.ln2(f2)
        # f1 = f1.permute(0,3,1,2)
        # f2 = f2.permute(0,3,1,2)

        feats = torch.cat((f1,f2), dim=1)
        feats = feats.view(batch_size, 2, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, 2, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats*attention_vectors, dim=1)

        feats_V = self.conv_end1(feats_V)
        # feats_V = self.act(self.bn(feats_V))
        return feats_V

class Feature_Extractor(nn.Module):
    def __init__(self, features, M=3, stride=1):
        super(Feature_Extractor, self).__init__()
        d = features
        norm_layer = nn.BatchNorm2d
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv1 = nn.Sequential(
              nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
              nn.LeakyReLU(0.2, True),
              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
              nn.LeakyReLU(0.2, True),
              nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(0.2, True),
              nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
              nn.LeakyReLU(0.2, True),
          )

        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i,
                          groups=features, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.LayerNorm([features, 1, 1]),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
        # self.e_rot = RotEncoderBlock_Mamba(64, mixer_kernel=(7,7))
        self.e1 = EncoderBlock_Mamba(64, mixer_kernel=(7,7))
        self.e2 = EncoderBlock_Mamba(64, mixer_kernel=(7,7))
        self.e3 = EncoderBlock_Mamba(64, mixer_kernel=(7,7))
        self.conv_end1 = nn.Conv2d(64, 64, kernel_size=1)
        # self.conv_end2 = nn.Conv2d(128, 64, kernel_size=1)

        self.select1 = SB(64)
        # self.select2 = SB(64)

    def forward(self, x):
        x = self.conv1(x)
        x2 = x.clone()

        x = self.e1(x)
        x2= self.e2(x2)
        x = self.select1(x, x2)

        batch_size = x.shape[0]

        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        mask = [fc(feats_Z) for fc in self.fcs]
        mask = torch.cat(mask, dim=1)
        mask = mask.view(batch_size, self.M, self.features, 1, 1)
        mask = self.softmax(mask)
        feats_V = torch.sum(feats * mask, dim=1)

        return feats_V
    
# Priority Channel Attention (PCA)
class PCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Conv1d(1, 1, kernel_size=64, stride=64)
        self.dw = nn.Conv2d(dim, dim, kernel_size=9, groups=int(dim) if isinstance(dim, torch.Tensor) else dim, padding=4)
        self.prob = nn.Softmax(dim=1)

    def cal_covariance(self, input):
        B, C, H, W = input.size()
        CovaMatrix_list = []

        for i in range(B):
            local_feature = input[i]  # C, H, W
            reshaped_tensor = local_feature.view(C, H*W)  # H*W, C
            covariance_matrix = torch.cov(reshaped_tensor)  # C, C
            CovaMatrix_list.append(covariance_matrix)

        CovaMatrix_list = torch.stack(CovaMatrix_list, dim=0)  # B, C, C
        CovaMatrix_list = CovaMatrix_list.view(B, C * C)  # Reshape for Conv1d
        CovaMatrix_list = self.linear(CovaMatrix_list.view(B,1,-1)).squeeze(1)  # Ensure correct shape
        return CovaMatrix_list  # B, C

    def forward(self, x):
        c = self.cal_covariance(x)
        x = self.dw(x)
        c_ = self.cal_covariance(x)
        raise_ch = self.prob(c_ - c)
        att_score = torch.sigmoid(c_ * (1 + raise_ch))
        att_score = att_score.unsqueeze(2).unsqueeze(3)
        return x * att_score
    
# Priority Spatial Attention (PSA)
class PSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Conv1d(1, 1, kernel_size = 256, stride = 256)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.prob = nn.Softmax2d()

    def cosine_attention(self, x):
        B, C, H, W = x.size()
        score_vector = []
        for i in range(B):
            local_feature = x[i] #C,H,W
            local_feature = local_feature.permute(1,2,0)
            local_feature = local_feature.view(H * W, C) #H*W, C
            score = torch.matmul(local_feature, local_feature.t()) #H*W,H*W
            score = score.sum(dim=0)
            score = F.normalize(score, p=2, dim=0)
            score_vector.append(score)
        score_vector = torch.stack(score_vector)
        score_vector = score_vector.view(B,H,W)
        return score_vector
    def forward(self, x):
        s = self.cosine_attention(x)
        x = self.pw(x)
        s_ = self.cosine_attention(x)
        raise_sp = self.prob(s_ - s)
        att_score = torch.sigmoid(s_ * (1 + raise_sp))
        att_score = att_score.unsqueeze(1)
        return x * att_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class MahalanobisBlock(nn.Module):
    def __init__(self):
        super(MahalanobisBlock, self).__init__()

    def cal_covariance(self, input):
        CovaMatrix_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, h, w = support_set_sam.size()
            local_feature_list = []

            for local_feature in support_set_sam:
                local_feature_np = local_feature.detach().cpu().numpy()
                transposed_tensor = np.transpose(local_feature_np, (1, 2, 0))
                reshaped_tensor = np.reshape(transposed_tensor, (h * w, C))

                for line in reshaped_tensor:
                    local_feature_list.append(line)

            local_feature_np = np.array(local_feature_list)
            # mean = np.mean(local_feature_np, axis=0)
            # local_feature_list = [x - mean for x in local_feature_list]

            covariance_matrix = np.cov(local_feature_np, rowvar=False)
            covariance_matrix = torch.from_numpy(covariance_matrix)
            CovaMatrix_list.append(covariance_matrix)

        return CovaMatrix_list



    def mahalanobis_similarity(self, input, CovaMatrix_list):
        B, C, h, w = input.size()
        mahalanobis = []

        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm
            mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w).cuda()
            for j in range(len(CovaMatrix_list)):
                covariance_matrix = CovaMatrix_list[j].float().cuda()
                diff = query_sam - torch.mean(query_sam, dim=1, keepdim=True)
                temp_dis = torch.matmul(torch.matmul(diff.T, covariance_matrix), diff)
                mea_sim[0, j * h * w:(j + 1) * h * w] = temp_dis.diag()

            mahalanobis.append(mea_sim.view(1, -1))

        mahalanobis = torch.cat(mahalanobis, 0)

        return mahalanobis


    def forward(self, x1, x2):

        CovaMatrix_list = self.cal_covariance(x2)
        maha_sim = self.mahalanobis_similarity(x1, CovaMatrix_list)

        return maha_sim
    
class MultiLevel_Model(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes = 13):
        super(MultiLevel_Model, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = Feature_Extractor(64)
        self.w = nn.Linear(10, 10)
        self.Spatial_Level = MahalanobisBlock()
        self.classifier_S = nn.Sequential(
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size = 256, stride = 256, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        )
        self.classifier_C = nn.Sequential(
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size = 64, stride = 64, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        )
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.pca = PSA(64)
        self.psa = PCA(64)
        # self.activation = nn.LeakyReLU(0.2, True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input1, input2):
        q = self.features(input1)
        q_channel = self.pca(q)
        q_spatial = self.psa(q)
        B, C, H, W = q.size()
        S_channel = []
        S_spatial = []
        for i in range(len(input2)):
            s_safe = self.features(input2[i])
            S_channel.append(self.pca(s_safe))
            S_spatial.append(self.psa(s_safe))

        m_c = self.Spatial_Level(q_spatial, S_spatial)
        m_c = self.classifier_S(m_c.view(m_c.size(0), 1, -1))
        m_c = m_c.squeeze(1)
        # m_c = self.softmax(m_c)

        scores = []
        scores_batch = []

        for q_batch in q_channel:
            q_batch = q_batch.view(C, H * W)
            for s_class in S_channel:
                for s_batch in s_class:
                    s_batch = s_batch.view(C, H * W)
                    # R_ = torch.matmul(q_batch.t(), s_batch)
                    R_ = torch.matmul(s_batch, q_batch.t())
                    mask = torch.sum(R_, dim=0)
                    mask = F.normalize(mask, p=2, dim=0)
                    # score = torch.matmul(R_, mask)
                    # top1_value, _ = torch.topk(score, 1)
                    # scores_batch.append(score)
                    scores_batch.append(mask)

                scores.append(sum(scores_batch)/len(scores_batch))
        m_s = torch.stack(scores, dim=0)
        m_s = m_s.view(B,1,-1)
        m_s = self.classifier_C(m_s)
        m_s = m_s.squeeze(1)
        # m_s = self.softmax(m_s)

        total = self.w1 * m_s + self.w2 * m_c

        # total = self.softmax(total)
        return total