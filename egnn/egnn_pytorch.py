import torch
from torch import nn, einsum, broadcast_tensors
from einops import rearrange, repeat
import numpy as np
import torch.nn.functional as F

def exists(val):
    return val is not None


def safe_div(num, den, eps=1e-8):
    res = num.div(den.clamp(min=eps))
    res.masked_fill_(den == 0, 0.)
    return res


def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


def embedd_token(x, dims, layers):
    stop_concat = -len(dims)
    to_embedd = x[:, stop_concat:].long()
    for i, emb_layer in enumerate(layers):
        # the portion corresponding to `to_embedd` part gets dropped
        x = torch.cat([x[:, :stop_concat], emb_layer(to_embedd[:, i])], dim=-1)
        stop_concat = x.shape[-1]
    return x


# swish activation fallback

class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_
# this follows the same strategy for normalization as done in SE3 Transformers
# https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/se3_transformer_pytorch.py#L95

class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


# global linear attention有用
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, mask=None):
        h = self.heads

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask):
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b () () n')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class GlobalLinearAttention(nn.Module):
    def __init__(self, *, dim, heads=8, dim_head=64):
        super().__init__()
        self.norm_seq = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, heads, dim_head)
        self.attn2 = Attention(dim, heads, dim_head)

        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, queries, mask=None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x), self.norm_queries(queries)

        induced = self.attn1(queries, x, mask=mask)
        out = self.attn2(x, induced)

        x = out + res_x
        queries = induced + res_queries

        x = self.ff(x) + x
        return x, queries


class EGNN(nn.Module):
    def __init__(self, dim, edge_dim=0, m_dim=16, fourier_features=0, num_nearest_neighbors=0, dropout=0.0,
                 init_eps=1e-3, norm_feats=False, norm_coors=False, norm_coors_scale_init=1e-2, update_feats=True,
                 update_coors=True, only_sparse_neighbors=False, valid_radius=float('inf'), m_pool_method='sum',
                 soft_edges=False, coor_weights_clamp_value=None):
        super().__init__()
        assert m_pool_method in {'sum', 'mean'}, 'pool method must be either sum or mean'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'

        self.fourier_features = fourier_features

        edge_input_dim = (fourier_features * 2) + (dim * 2) + edge_dim + 1
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = nn.Sequential(nn.Linear(edge_input_dim, edge_input_dim * 2), dropout, SiLU(),
                                      nn.Linear(edge_input_dim * 2, m_dim), SiLU())
        self.edge_gate = nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid()) if soft_edges else None
        self.node_norm = nn.LayerNorm(dim) if norm_feats else nn.Identity()
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()
        self.m_pool_method = m_pool_method
        self.node_mlp = nn.Sequential(nn.Linear(dim + m_dim, dim * 2), dropout, SiLU(),
                                      nn.Linear(dim * 2, dim), ) if update_feats else None
        self.coors_mlp = nn.Sequential(nn.Linear(m_dim, m_dim * 4), dropout, SiLU(),
                                       nn.Linear(m_dim * 4, 1)) if update_coors else None

        self.num_nearest_neighbors = num_nearest_neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_radius = valid_radius

        self.coor_weights_clamp_value = coor_weights_clamp_value

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.normal_(module.weight, std=self.init_eps)

    def forward(self, feats, coors, edges=None, mask=None, adj_mat=None):
        b, n, d, device, fourier_features, num_nearest, valid_radius, only_sparse_neighbors = *feats.shape, feats.device, self.fourier_features, self.num_nearest_neighbors, self.valid_radius, self.only_sparse_neighbors
        use_nearest = num_nearest > 0 or only_sparse_neighbors  #true

        # 计算相对位置矩阵，维度为(b, n, n, 1)
        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if use_nearest:
            ranking = rel_dist[..., 0].clone()

            if exists(mask):
                rank_mask = mask[:, :, None] * mask[:, None, :]
                ranking.masked_fill_(~rank_mask, 1e5)

            if exists(adj_mat):
                if len(adj_mat.shape) == 2:
                    adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b=b)

                if only_sparse_neighbors:
                    num_nearest = int(adj_mat.float().sum(dim=-1).max().item())
                    valid_radius = 0

                self_mask = rearrange(torch.eye(n, device=device, dtype=torch.bool), 'i j -> () i j')

                adj_mat = adj_mat.masked_fill(self_mask, False)
                ranking.masked_fill_(self_mask, -1.)
                ranking.masked_fill_(adj_mat, 0.)

            # 将num_nearest设置为n，以确保不超出邻居索引的范围
            num_nearest = min(num_nearest, n)

            # 返回num_nearest个最小的值和对应的indices，注意num_nearst应小于等于n
            nbhd_ranking, nbhd_indices = ranking.topk(num_nearest, dim=-1, largest=False)

            # 默认valid_radius为无穷大，类似于r-ball的设计
            nbhd_mask = nbhd_ranking <= valid_radius

            rel_coors = batched_index_select(rel_coors, nbhd_indices, dim=2)
            rel_dist = batched_index_select(rel_dist, nbhd_indices, dim=2)  #信息传递过程

            if exists(edges):
                edges = batched_index_select(edges, nbhd_indices, dim=2)

        if fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=fourier_features)
            rel_dist = rearrange(rel_dist, 'b i j () d -> b i j d')

        if use_nearest:
            feats_j = batched_index_select(feats, nbhd_indices, dim=1)
        else:
            feats_j = rearrange(feats, 'b j d -> b () j d')

        feats_i = rearrange(feats, 'b i d -> b i () d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)      #（8，97，32，257）

        if exists(edges):
            edge_input = torch.cat((edge_input, edges), dim=-1)

        m_ij = self.edge_mlp(edge_input)

        if exists(self.edge_gate): m_ij = m_ij * self.edge_gate(m_ij)

        if exists(mask):
            mask_i = rearrange(mask, 'b i -> b i ()')

            if use_nearest:
                mask_j = batched_index_select(mask, nbhd_indices, dim=1)
                mask = (mask_i * mask_j) & nbhd_mask
            else:
                mask_j = rearrange(mask, 'b j -> b () j')
                mask = mask_i * mask_j

        if exists(self.coors_mlp):
            coor_weights = self.coors_mlp(m_ij)
            coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

            rel_coors = self.coors_norm(rel_coors)

            if exists(mask):
                coor_weights.masked_fill_(~mask, 0.)

            if exists(self.coor_weights_clamp_value):
                clamp_value = self.coor_weights_clamp_value
                coor_weights.clamp_(min=-clamp_value, max=clamp_value)

            coors_out = einsum('b i j, b i j c -> b i c', coor_weights, rel_coors) + coors
        else:
            coors_out = coors

        # 更新特征
        if exists(self.node_mlp):
            if exists(mask):
                m_ij_mask = rearrange(mask, '... -> ... ()')
                # 不存在的节点特征mask为0
                m_ij = m_ij.masked_fill(~m_ij_mask, 0.)

            if self.m_pool_method == 'mean':
                if exists(mask):
                    # masked mean
                    mask_sum = m_ij_mask.sum(dim=-2)
                    m_i = safe_div(m_ij.sum(dim=-2), mask_sum)
                else:
                    m_i = m_ij.mean(dim=-2)

            elif self.m_pool_method == 'sum':
                m_i = m_ij.sum(dim=-2)

            normed_feats = self.node_norm(feats)
            node_mlp_input = torch.cat((normed_feats, m_i), dim=-1)
            node_out = self.node_mlp(node_mlp_input) + feats
        else:
            node_out = feats

        return node_out, coors_out


class EGNN_Network(nn.Module):
    def __init__(self, *, depth, dim, num_tokens=None, num_edge_tokens=None, num_positions=None, edge_dim=0,
                 num_adj_degrees=None, adj_dim=0, global_linear_attn_every=0, global_linear_attn_heads=8,
                 global_linear_attn_dim_head=64, num_global_tokens=4, num_prompt=9,  **kwargs):
        super().__init__()
        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than 1'
        self.prompt_embed = nn.Embedding(num_prompt, dim)
        self.num_positions = num_positions

        # 初始化embedding矩阵，num_tokens表示tokens的种类数目
        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None
        self.pos_emb = nn.Embedding(num_positions, dim) if exists(num_positions) else None
        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None
        self.has_edges = edge_dim > 0

        self.num_adj_degrees = num_adj_degrees
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None

        edge_dim = edge_dim if self.has_edges else 0
        adj_dim = adj_dim if exists(num_adj_degrees) else 0

        has_global_attn = global_linear_attn_every > 0
        self.global_tokens = None
        if has_global_attn:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, dim))

        self.layers = nn.ModuleList([])
        for ind in range(depth ):
            is_global_layer = has_global_attn and (ind % global_linear_attn_every) == 0
            self.layers.append(nn.ModuleList([GlobalLinearAttention(dim=dim, heads=global_linear_attn_heads,
                                                                    dim_head=global_linear_attn_dim_head) if is_global_layer else None,
                                              EGNN(dim=dim, edge_dim=(edge_dim + adj_dim), norm_feats=True, **kwargs), ]))

    def forward(self, feats, coors, prompt=None, adj_mat=None, edges=None, mask=None, return_coor_changes=False):
        b, device = feats.shape[0], feats.device

        if exists(self.token_emb):
            feats = self.token_emb(feats)

        if exists(self.prompt_embed):
            feats += self.prompt_embed(torch.tensor(prompt).to(device))

        # setup global attention
        global_tokens = None
        if exists(self.global_tokens):
            global_tokens = repeat(self.global_tokens, 'n d -> b n d', b=b)

        # go through layers
        coor_changes = []

        for global_attn, egnn in self.layers:
            if exists(global_attn):
                  feats, global_tokens = global_attn(feats, global_tokens, mask=mask)
            feats, coors = egnn(feats, coors, adj_mat=adj_mat, edges=edges, mask=mask)

        if return_coor_changes:
            return coor_changes, feats

        return coors, feats


class TEGN(nn.Module):
    def __init__(self, depth, dim, num_tokens=100, pretrain_MTL=True, vocab_size=13):
        super().__init__()
        self.dim = dim
        self.pretrain_MTL = pretrain_MTL


        #这里装载即可
        self.encoder_1 = EGNN_Network(num_tokens=num_tokens, dim=dim, depth=depth, num_nearest_neighbors=32,
                         dropout=0.15, global_linear_attn_every=1, norm_coors=True, coor_weights_clamp_value=2.,
                         num_prompt=9)
        self.encoder_2 = EGNN_Network(num_tokens=num_tokens, dim=dim, depth=depth, num_nearest_neighbors=32,
                                      dropout=0.15, global_linear_attn_every=1, norm_coors=True,
                                      coor_weights_clamp_value=2.,
                                      num_prompt=9)
        self.encoder_3 = EGNN_Network(num_tokens=num_tokens, dim=dim, depth=depth, num_nearest_neighbors=32,
                                      dropout=0.15, global_linear_attn_every=1, norm_coors=True,
                                      coor_weights_clamp_value=2.,
                                      num_prompt=9)
        self.encoder_4 = EGNN_Network(num_tokens=num_tokens, dim=dim, depth=depth, num_nearest_neighbors=32,
                                      dropout=0.15, global_linear_attn_every=1, norm_coors=True,
                                      coor_weights_clamp_value=2.,
                                      num_prompt=9)
        self.encoder_5 = EGNN_Network(num_tokens=num_tokens, dim=dim, depth=depth, num_nearest_neighbors=32,
                                      dropout=0.15, global_linear_attn_every=1, norm_coors=True,
                                      coor_weights_clamp_value=2.,
                                      num_prompt=9)
        self.encoder_6 = EGNN_Network(num_tokens=num_tokens, dim=dim, depth=depth, num_nearest_neighbors=32,
                                      dropout=0.15, global_linear_attn_every=1, norm_coors=True,
                                      coor_weights_clamp_value=2.,
                                      num_prompt=9)
        self.encoder_7 = EGNN_Network(num_tokens=num_tokens, dim=dim, depth=depth, num_nearest_neighbors=32,
                                      dropout=0.15, global_linear_attn_every=1, norm_coors=True,
                                      coor_weights_clamp_value=2.,
                                      num_prompt=9)
        self.encoder_8 = EGNN_Network(num_tokens=num_tokens, dim=dim, depth=depth, num_nearest_neighbors=32,
                                      dropout=0.15, global_linear_attn_every=1, norm_coors=True,
                                      coor_weights_clamp_value=2.,
                                      num_prompt=9)

        self.fc1 = nn.Linear(dim, dim * 2)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(dim * 2, vocab_size)
        self.cbam_layers = nn.ModuleList([CBAM(dim=dim) for _ in range(3)])

    def forward(self, charges, loc, mask=None):
        _, out_1 = self.encoder_1(charges, loc, prompt=1, mask=mask)
        _, out_2 = self.encoder_2(charges, loc, prompt=2, mask=mask)
        _, out_3 = self.encoder_3(charges, loc, prompt=3, mask=mask)
        _, out_4 = self.encoder_4(charges, loc, prompt=4, mask=mask)
        _, out_5 = self.encoder_5(charges, loc, prompt=5, mask=mask)
        _, out_6 = self.encoder_6(charges, loc, prompt=6, mask=mask)
        _, out_7 = self.encoder_7(charges, loc, prompt=7, mask=mask)
        _, out_8 = self.encoder_8(charges, loc, prompt=8, mask=mask)

        feats_list = [out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8]

        # 拼接8次迭代的特征向量
        feats = torch.stack(feats_list, dim=1)  # (batch_size, num_iterations * dim)

        # 经过CBAM模块处理,可以用残差连接，还可以多堆叠几层
        for layer in self.cbam_layers:
            feats = layer(feats)

        # print(feats_3.shape)
        x = feats.mean(dim=1)  #可以修改下通道注意力改8为1
        # x = torch.max(spatial_3, dim=1, keepdim=False)[0]

        if self.pretrain_MTL:
            x = self.fc1(x)
            x = self.dropout1(x)
            x = F.gelu(x)
            x = self.fc2(x)
            # print(x.shape)

        return x

#三层cbam用不同的参数，且接一个前向层
class CBAM(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(CBAM, self).__init__()
        self.ca = channel_attention(in_channel=8, ratio=2)
        self.sa = spatial_attention()
        self.LN_1 = nn.LayerNorm(dim)
        self.LN_2 = nn.LayerNorm(dim)
        self.FeedForward = nn.Sequential(nn.Linear(dim, dim * 2), nn.Dropout(p=dropout), nn.ReLU(), nn.Linear(dim * 2, dim))

    def forward(self, feats):
        channel = self.ca(feats)
        spatial = self.sa(channel)
        feats = feats + spatial
        feats = self.LN_1(feats)
        feats_new = self.FeedForward(feats)
        feats = feats_new + feats
        feats = self.LN_2(feats)

        return feats

class channel_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=2):
        # 继承父类初始化方法
        super(channel_attention, self).__init__()

        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获取输入特征图的shape
        b, c, h, w = inputs.shape

        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)
        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool
        # sigmoid函数权值归一化
        x = self.sigmoid(x)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])
        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x

        return outputs

class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x

        return outputs


# 原始预测器，输出是一个单值,直接用在任务上，而不是反向传播
class predictor(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
        self.out = nn.Sequential(nn.Linear(dim, dim*2), nn.Dropout(p=dropout),nn.ReLU(), nn.Linear(dim*2, 1))  #bbbp和bace可用nn.GELU   nn.Dropout(p=dropout),

    def forward(self, feats, mask=None):
        if exists(mask):
            feats = feats.masked_fill(mask.unsqueeze(-1) == 0, 0)
            return self.out(torch.sum(feats, dim=-2) / torch.sum(mask, dim=-1, keepdim=True))
        return self.out(torch.mean(feats, dim=-2))


