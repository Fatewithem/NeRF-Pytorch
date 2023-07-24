import torch

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# 位置编码
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dims = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dims += d

        max_freq = self.kwargs['max_freq_log2']
        N_fregs = self.kwargs['num_freqs']

        # [2^0, 2^1, ... ,2^(L-1)]
        if self.kwargs['log_sampling']:
            freq_band = 2.**torch.linspace(0., max_freq, steps=N_fregs)
        else:
            freq_band = torch.linspace(2.**0., 2.**max_freq, steps=N_fregs)

        # 进行cos sin运算
        for freq in freq_band:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq)) # x * 2^n
                out_dims += d # 维度加3

        self.embed_fns = embed_fns
        self.out_dims = out_dims

    # 将所有inputs都合并
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1) # 最后一个维度连接

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs) # 获取参数
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dims

# 模型
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4],use_viewdirs=False):
        # super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.output_ch = output_ch
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # pts MLP
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W,W) if i not in self.skips
                            else nn.Linear(W + input_ch, W)
                            for i in range(D-1)])
        # views MLP
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)


    def forward(self, x):
        # 划分[3, 3]的矩阵
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        # 进行每层的计算
        for i, l in enumerate(self.pts_linears): # l表示 enumerate(self.pts_linears) 迭代的第二个返回值
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "如果 use_viewdirs=False 则不实现"

        # 读取 pts_linears
        for i in range(self.D):
            idx_pts_linear = 2 * i # 前 8 层
            # 从numpy数组转化为Pytorch张量
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linear]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linear + 1]))

        # 读取 feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # 读取 views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # 读取 rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # 读取 alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

# 光线
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    # 进行转置操作
    i = i.t()
    j = j.t()
    # 方向向量 (H, W, 3)
    dirs = torch.stack([(i - K[0][2])/K[0][0], -(j - K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # 相机坐标系到世界坐标系
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # np.newaxis增加维度（H, W, 3, 1）
    rays_o = c2w[:3, -1].expand(rays_d.shape)  # (H, W, 3)
    return rays_d, rays_o

# 使用np得到
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # 方向向量 (H, W, 3)
    dirs = np.stack([(i - K[0][2])/K[0][0], -(j - K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # 相机坐标系到世界坐标系
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)   # np.newaxis增加维度（H, W, 3, 1）
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_d, rays_o

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # 将光线的起始点放到近平面上
    t = -(near + rays_o[...,2] / rays_d[...,2])  # 交点的距离
    rays_o = rays_o + t[..., None] * rays_d  # 平移rays_o

    # 投影(按z轴比例进行计算x,y)
    ox = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    oy = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    oz = 1. + 2. * near / rays_o[..., 2]

    dx = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    dy = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    dz= -2. * near / rays_o[..., 2]

    rays_o = torch.stack([ox, oy, oz], -1)
    rays_d = torch.stack([dx, dy, dz], -1)

    return rays_o, rays_d

# 分层采样
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # 获取 pdf
    weights = weights + 1e-5  # 避免 NaN 出现
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # 归一化概率密度函数
    cdf = torch.cumsum(pdf, -1)  # 得到累积分布函数
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # 拼接零向量

    # 均匀采样
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # 逆变换采样
    u = u.contiguous()
    index = torch.searchsorted(cdf, u, right=True)  # 在cdf中找到一个形状与 u 张量相同的索引张量
    below = torch.max(torch.zeros_like(index-1), index-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(index), index)
    index_g = torch.stack([below, above], -1)

    matched_shape = [index_g.shape[0], index_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, index_g)  # 第二个维度
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, index_g)

    # 线性插值
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

































