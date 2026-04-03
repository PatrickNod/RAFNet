import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# =========================================================================
# 1. 基础小波变换模块 (保持不变)
# =========================================================================
class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            _, C, _, _ = ctx.shape
            dx_ll, dx_lh, dx_hl, dx_hh = dx[:, :C], dx[:, C:C * 2], dx[:, C * 2:C * 3], dx[:, C * 3:]
            dx_x_ll = torch.nn.functional.conv_transpose2d(dx_ll, w_ll.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_lh = torch.nn.functional.conv_transpose2d(dx_lh, w_lh.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_hl = torch.nn.functional.conv_transpose2d(dx_hl, w_hl.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_hh = torch.nn.functional.conv_transpose2d(dx_hh, w_hh.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            return dx_x_ll + dx_x_lh + dx_x_hl + dx_x_hh, None, None, None, None
        else:
            return dx, None, None, None, None

class DWT_2D(nn.Module):
    def __init__(self):
        super(DWT_2D, self).__init__()
        w_ll = torch.tensor([[[[0.25, 0.25], [0.25, 0.25]]]], dtype=torch.float32, requires_grad=False)
        w_lh = torch.tensor([[[[0.25, 0.25], [-0.25, -0.25]]]], dtype=torch.float32, requires_grad=False)
        w_hl = torch.tensor([[[[0.25, -0.25], [0.25, -0.25]]]], dtype=torch.float32, requires_grad=False)
        w_hh = torch.tensor([[[[0.25, -0.25], [-0.25, 0.25]]]], dtype=torch.float32, requires_grad=False)
        self.register_buffer('w_ll', w_ll)
        self.register_buffer('w_lh', w_lh)
        self.register_buffer('w_hl', w_hl)
        self.register_buffer('w_hh', w_hh)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        _, C, _, _ = x.shape
        w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
        x_ll, x_lh, x_hl, x_hh = x[:, :C // 4], x[:, C // 4:C * 2 // 4], x[:, C * 2 // 4:C * 3 // 4], x[:, C * 3 // 4:]
        x_1_ll = torch.nn.functional.conv_transpose2d(x_ll, w_ll.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        x_1_lh = torch.nn.functional.conv_transpose2d(x_lh, w_lh.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        x_1_hl = torch.nn.functional.conv_transpose2d(x_hl, w_hl.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        x_1_hh = torch.nn.functional.conv_transpose2d(x_hh, w_hh.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        return x_1_ll + x_1_lh + x_1_hl + x_1_hh

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            _, C, _, _ = ctx.shape
            C //= 4
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1) / 4, stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1) / 4, stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1) / 4, stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1) / 4, stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self):
        super(IDWT_2D, self).__init__()
        w_ll = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32, requires_grad=False)
        w_lh = torch.tensor([[[[1, 1], [-1, -1]]]], dtype=torch.float32, requires_grad=False)
        w_hl = torch.tensor([[[[1, -1], [1, -1]]]], dtype=torch.float32, requires_grad=False)
        w_hh = torch.tensor([[[[1, -1], [-1, 1]]]], dtype=torch.float32, requires_grad=False)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

# =========================================================================
# 2. 可微分超像素聚类 (Differentiable Superpixel)
# =========================================================================
class DifferentiableSoftKMeans(nn.Module):
    def __init__(self, channel, n_clusters=16, hidden_dim=32, temperature=1.0, 
                 spatial_weight=10.0, n_iterations=3):
        """
        可微分超像素聚类 - 改进版
        
        Args:
            channel: 输入特征通道数
            n_clusters: 簇数量
            hidden_dim: 嵌入维度
            temperature: softmax温度
            spatial_weight: 空间距离权重（控制超像素紧凑度）
            n_iterations: 迭代更新次数
        """
        super(DifferentiableSoftKMeans, self).__init__()
        self.n_clusters = n_clusters
        self.temperature = temperature
        self.spatial_weight = spatial_weight
        self.n_iterations = n_iterations
        self.hidden_dim = hidden_dim
        
        # 1. 特征嵌入网络（更深的网络提取更好的特征）
        self.embedding = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )
        
        # 2. 可学习的簇中心（在forward中初始化）
        self.register_buffer('grid_cache', None)
        self.register_buffer('H_cache', torch.tensor(0))
        self.register_buffer('W_cache', torch.tensor(0))
        
        # 3. 初始簇中心位置（网格初始化，适合超像素）
        grid_h = int(n_clusters ** 0.5)
        grid_w = (n_clusters + grid_h - 1) // grid_h
        init_centers = []
        for i in range(grid_h):
            for j in range(grid_w):
                if len(init_centers) < n_clusters:
                    y = (i + 0.5) / grid_h * 2 - 1
                    x = (j + 0.5) / grid_w * 2 - 1
                    init_centers.append([x, y])
        self.register_buffer('init_spatial_centers', torch.tensor(init_centers).float())
        
        # 4. 可学习的特征中心
        self.feature_centers = nn.Parameter(torch.randn(n_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.feature_centers)

    def _get_grid(self, H, W, device):
        """获取或缓存空间坐标网格"""
        if self.grid_cache is None or self.H_cache.item() != H or self.W_cache.item() != W:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device), 
                torch.linspace(-1, 1, W, device=device), 
                indexing="ij"
            )
            grid = torch.stack([xx, yy], dim=0)  # [2, H, W]
            self.grid_cache = grid
            self.H_cache = torch.tensor(H)
            self.W_cache = torch.tensor(W)
        return self.grid_cache

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # 1. 提取特征嵌入
        features = self.embedding(x)  # [B, hidden_dim, H, W]
        features_flat = features.permute(0, 2, 3, 1).reshape(B, N, self.hidden_dim)  # [B, N, hidden_dim]
        
        # 2. 获取空间坐标
        spatial_grid = self._get_grid(H, W, x.device)  # [2, H, W]
        spatial_flat = spatial_grid.permute(1, 2, 0).reshape(N, 2)  # [N, 2]
        
        # 3. 初始化簇中心（特征 + 空间）
        feature_centers = self.feature_centers.unsqueeze(0).expand(B, -1, -1)  # [B, K, hidden_dim]
        spatial_centers = self.init_spatial_centers.clone()  # [K, 2]
        
        # 4. 迭代优化簇分配和中心
        for _ in range(self.n_iterations):
            # 4.1 计算特征距离 [B, N, K]
            feature_dists = torch.cdist(features_flat, feature_centers)  # [B, N, K]
            
            # 4.2 计算空间距离 [N, K]
            spatial_dists = torch.cdist(spatial_flat.unsqueeze(0), spatial_centers.unsqueeze(0)).squeeze(0)  # [N, K]
            
            # 4.3 综合距离（特征 + 空间正则化）
            total_dists = feature_dists + self.spatial_weight * spatial_dists.unsqueeze(0)  # [B, N, K]
            
            # 4.4 软分配（Softmax）
            soft_assign = F.softmax(-total_dists / self.temperature, dim=2)  # [B, N, K]
            
            # 4.5 更新特征中心（加权平均）
            weights_sum = soft_assign.sum(dim=1, keepdim=True).transpose(1, 2)  # [B, K, 1]
            feature_centers = torch.bmm(soft_assign.transpose(1, 2), features_flat) / (weights_sum + 1e-6)  # [B, K, hidden_dim]
            
            # 4.6 更新空间中心（对所有batch取平均）
            weights_spatial = soft_assign.mean(dim=0).T  # [K, N]
            spatial_centers = torch.mm(weights_spatial, spatial_flat) / (weights_spatial.sum(dim=1, keepdim=True) + 1e-6)  # [K, 2]
        
        # 5. 使用 Straight-Through Estimator (STE) 实现可微硬分配
        # 
        # 梯度流保证机制：
        # - hard_labels_forward 是 argmax 的结果 [B, N]
        # - 需要转换为 one-hot 编码 [B, N, K] 才能与 soft_assign 匹配
        # - 然后 (hard_onehot - soft_assign).detach() + soft_assign
        # - 反向传播时梯度只通过 + soft_assign 流回
        
        hard_labels_forward = soft_assign.argmax(dim=2)  # [B, N] - 不可微的硬标签索引
        
        # 转换为one-hot编码 [B, N, K]
        hard_onehot = F.one_hot(hard_labels_forward, num_classes=self.n_clusters).float()  # [B, N, K]
        
        # 关键：只有这一项有梯度
        hard_labels_ste = (hard_onehot - soft_assign).detach() + soft_assign
        #                  ↑ 无梯度(detach切断)        ↑ 有梯度(直连)
        
        # 将soft assignment转回硬标签用于后续模块
        hard_labels = hard_labels_ste.argmax(dim=2).reshape(B, H, W)
        
        return hard_labels

# =========================================================================
# 3. PWAC (低秩自适应卷积 - Low-Rank Adaptive Convolution)
# =========================================================================
class PWAC(nn.Module):
    def __init__(self, in_planes, out_planes, n_clusters=16, kernel_size=3, padding=1, 
                 rank=8, mlp_dim=32):
        """
        低秩自适应卷积模块
        
        Args:
            in_planes: 输入通道数
            out_planes: 输出通道数
            n_clusters: 簇数量
            kernel_size: 卷积核大小
            padding: 填充大小
            rank: 低秩维度（越小参数越少）
            mlp_dim: MLP隐藏维度
        """
        super(PWAC, self).__init__()
        self.n_clusters = n_clusters
        self.padding = padding
        self.kernel_size = kernel_size
        self.kernel_area = kernel_size * kernel_size
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.rank = rank
        
        # 低秩分解：不直接存储 weight，而是学习低秩映射
        # 方案：从聚类中心 → MLP → 低秩表示 → 展开为完整卷积核
        
        # 1. 将聚类中心映射到低秩表示
        self.centroid_to_lowrank = nn.Sequential(
            nn.Linear(in_planes * self.kernel_area, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, rank)
        )
        
        # 2. 基础核（rank个基核）- 展平形式便于einsum组合
        # [rank, in_planes*kernel_area, out_planes]
        self.base_kernels = nn.Parameter(
            torch.randn(rank, in_planes * self.kernel_area, out_planes)
        )
        nn.init.kaiming_uniform_(self.base_kernels, a=5**0.5)
        
        # 3. 偏置生成
        self.centroid_to_bias = nn.Sequential(
            nn.Linear(in_planes * self.kernel_area, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, out_planes)
        )
        
    def forward(self, x, labels, cluster_centers=None):
        """
        Args:
            x: 输入特征 [B, C, H, W]
            labels: 聚类标签 [B, H, W]
            cluster_centers: 聚类中心特征 [B, n_clusters, C*kernel_area] 或 None
        """
        B, C, H, W = x.shape
        
        # 如果没有提供聚类中心，从输入推导（使用unfold）
        if cluster_centers is None:
            unfold = nn.Unfold(kernel_size=self.kernel_size, padding=self.padding)
            patches = unfold(x)  # [B, C*kernel_area, H*W]
            patches = patches.permute(0, 2, 1).reshape(B, H*W, -1)  # [B, H*W, C*kernel_area]
            
            # 通过one-hot编码计算簇中心
            labels_flat = labels.reshape(B, -1)  # [B, H*W]
            labels_onehot = F.one_hot(labels_flat, num_classes=self.n_clusters).float()  # [B, H*W, K]
            weights = labels_onehot.permute(0, 2, 1)  # [B, K, H*W]
            cluster_centers = torch.bmm(weights, patches) / (weights.sum(dim=2, keepdim=True) + 1e-6)  # [B, K, C*kernel_area]
        
        # 生成每个簇的卷积核系数（低秩权重）
        lowrank_weights = self.centroid_to_lowrank(cluster_centers)  # [B, K, rank]
        
        # 生成偏置
        bias_per_cluster = self.centroid_to_bias(cluster_centers)  # [B, K, out_channels]
        
        # 低秩重构：将rank个基核与权重组合
        # 方式：kernel_weights [B,K,rank] @ base_kernels [rank, in*area, out] -> [B,K, in*area, out]
        kernels_flat = torch.einsum('bkr,rio->bkio', lowrank_weights, self.base_kernels)  # [B, K, in_planes*kernel_area, out_planes]
        
        # 应用聚类掩码进行卷积
        out = torch.zeros(B, self.out_planes, H, W, device=x.device)
        labels_flat = labels.reshape(B, H*W)
        
        for b in range(B):
            for k in range(self.n_clusters):
                # 找出属于簇k的像素
                mask = (labels_flat[b] == k).float()  # [H*W]
                mask_2d = mask.reshape(1, 1, H, W)  # [1, 1, H, W]
                
                if mask.sum() < 1e-6:
                    continue
                
                # 获取该簇的卷积核 [in_planes*kernel_area, out_planes]
                kernel_k = kernels_flat[b, k]  # [in_planes*kernel_area, out_planes]
                
                # 转换为标准卷积核形式 [out_planes, in_planes, kernel_size, kernel_size]
                kernel_conv = kernel_k.t().reshape(self.out_planes, self.in_planes, self.kernel_size, self.kernel_size)
                
                # 使用该簇的卷积核和偏置
                feat_k = F.conv2d(x[b:b+1], kernel_conv, bias=bias_per_cluster[b, k], padding=self.padding)  # [1, out_planes, H, W]
                out[b:b+1] = out[b:b+1] + feat_k * mask_2d
        
        return out

# =========================================================================
# 4. 自适应注意力模块
# =========================================================================
class AdaptiveAttention(nn.Module):
    def __init__(self, channel, head_channel, dropout, n_clusters=16):
        super(AdaptiveAttention, self).__init__()
        self.head_channel = head_channel
        self.channel = channel
        self.scale = head_channel ** 0.5  # 每个头的缩放因子
        self.num_head = channel // head_channel  # 头数
        self.n_clusters = n_clusters
        
        self.q_proj = nn.Linear(channel, channel)
        self.k_proj = nn.Linear(channel, channel)
        self.v_proj = nn.Linear(channel, channel)
        
        # 注意：像素系数在forward中动态生成，这里不定义
        
        self.mlp_1 = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, q_img, k_img_map, v_img_map, labels, pixel_coefficients=None):
        """
        多头自适应注意力（向量化优化版本）：
        - 同簇内: Q * K^T (正常计算)
        - 簇外: Q * K_center * coefficient[m] (低秩近似，m是簇外像素的系数)
        - 多头机制: 提升模型表达能力
        
        labels: 像素簇标签 [B, H, W]
        pixel_coefficients: 每个 query-key 对的可学习系数矩阵 [B, N, N] 或 None (使用全1)
        """
        B, C, H, W = q_img.shape
        N = H * W
        
        # 如果没有提供像素系数，使用全1，形状 [B, N, N] 表示每个 query-key 对的系数
        if pixel_coefficients is None:
            pixel_coefficients = torch.ones(B, N, N, device=q_img.device)
        
        # 投影 Q, K, V
        q = self.q_proj(q_img.permute(0, 2, 3, 1)).reshape(B, N, C)  # [B, N, C]
        k = self.k_proj(k_img_map.permute(0, 2, 3, 1)).reshape(B, N, C)  # [B, N, C]
        v = self.v_proj(v_img_map.permute(0, 2, 3, 1)).reshape(B, N, C)  # [B, N, C]
        
        # 计算 K 的簇内均值 centers_k [B, n_clusters, C]
        labels_flat = labels.view(B, N)  # [B, N]
        labels_onehot = F.one_hot(labels_flat, num_classes=self.n_clusters).float()  # [B, N, n_clusters]
        weights = labels_onehot.permute(0, 2, 1)  # [B, n_clusters, N]
        
        # centers_k = 投影后 K 的簇内均值
        num_k = torch.bmm(weights, k)  # [B, n_clusters, C]
        den_k = weights.sum(dim=2, keepdim=True) + 1e-6  # [B, n_clusters, 1]
        centers_k = num_k / den_k  # [B, n_clusters, C]
        
        # 重塑为多头形式 [B, num_head, N, head_channel]
        q_multi = q.reshape(B, N, self.num_head, self.head_channel).permute(0, 2, 1, 3)
        k_multi = k.reshape(B, N, self.num_head, self.head_channel).permute(0, 2, 1, 3)
        v_multi = v.reshape(B, N, self.num_head, self.head_channel).permute(0, 2, 1, 3)
        
        # 簇中心也reshape为多头形式 [B, num_head, n_clusters, head_channel]
        k_centers_multi = centers_k.reshape(B, self.n_clusters, self.num_head, self.head_channel).permute(0, 2, 1, 3)
        
        # 为每个头计算 attention scores
        # 1. 计算 Q @ K^T (同簇内使用) [B, num_head, N, N]
        attn_intra = torch.matmul(q_multi, k_multi.transpose(2, 3))  # [B, num_head, N, N]
        
        # 2. 计算 Q @ K_centers^T (簇外使用) [B, num_head, N, n_clusters]
        attn_centers = torch.matmul(q_multi, k_centers_multi.transpose(2, 3))
        
        # 4. 构建最终的 attention 矩阵 [B, num_head, N, N]
        attn_scores = torch.zeros(B, self.num_head, N, N, device=q.device)
        
        for b in range(B):
            # 创建同簇掩码：[N, N]
            labels_i = labels_flat[b].unsqueeze(1)  # [N, 1]
            labels_j = labels_flat[b].unsqueeze(0)  # [1, N]
            same_cluster_mask = (labels_i == labels_j)  # [N, N]
            
            # 扩展掩码到多头维度 [num_head, N, N]
            same_cluster_mask = same_cluster_mask.unsqueeze(0).expand(self.num_head, -1, -1)
            
            # 同簇内：使用 Q * K^T
            attn_scores[b] = torch.where(same_cluster_mask, attn_intra[b], torch.zeros_like(attn_intra[b]))
            
            # 簇外：使用 Q * K_center[cluster_j] * coefficient[j]
            # attn_centers[b]: [num_head, N, n_clusters]，取每个位置对应簇的值
            labels_j_flat = labels_flat[b]  # [N]
            attn_inter = attn_centers[b].gather (2, labels_j_flat.unsqueeze(0).unsqueeze(0).expand(self.num_head, N, -1))  # [num_head, N, N]
            
            # 应用像素系数：系数[j] 作用于第j列
            coeffs = pixel_coefficients[b].unsqueeze(0)  # [1, N, N]
            attn_inter = attn_inter * coeffs  # 广播乘法 [num_head, N, N]
            
            # 簇外位置使用 attn_inter
            attn_scores[b] = torch.where(same_cluster_mask, attn_scores[b], attn_inter)
        
        # 缩放并应用 softmax
        attn_scores = attn_scores / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_head, N, N]
        
        # 计算加权 V [B, num_head, N, head_channel]
        out_multi = torch.matmul(attn_weights, v_multi)  # [B, num_head, N, head_channel]
        
        # 合并多头 [B, N, C]
        out = out_multi.permute(0, 2, 1, 3).reshape(B, N, C)
        
        # 残差连接和 MLP
        v_spatial = v.reshape(B, H, W, C).permute(0, 3, 1, 2)
        rs1 = v_spatial + self.mlp_1(out).permute(0, 2, 1).reshape(B, C, H, W)
        rs1_flat = rs1.permute(0, 2, 3, 1).reshape(B, -1, C)
        rs2 = rs1 + self.mlp_2(rs1_flat).permute(0, 2, 1).reshape(B, C, H, W)
        return rs2

# =========================================================================
# 5. GDFN (保持不变)
# =========================================================================
class GDFN(nn.Module):
    def __init__(self, dim, expansion_ratio=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.f1 = nn.Sequential(
            nn.Conv2d(dim, dim * expansion_ratio, kernel_size=1),
            nn.GroupNorm(4, dim * expansion_ratio),
            nn.Dropout2d(0.3),
            nn.Conv2d(dim * expansion_ratio, dim * expansion_ratio, 
                      kernel_size=3, padding=1, groups=dim * expansion_ratio),
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(dim, dim * expansion_ratio, kernel_size=1),
            nn.GroupNorm(4, dim * expansion_ratio),
            nn.Dropout2d(0.3),
            nn.Conv2d(dim * expansion_ratio, dim * expansion_ratio, 
                      kernel_size=3, padding=1, groups=dim * expansion_ratio),
        )
        self.gate_act = nn.GELU()
        self.map = nn.Sequential(
            nn.Conv2d(dim * expansion_ratio, dim, kernel_size=1),
        )
    def forward(self, x):
        x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f1 = self.f1(x_norm)
        f2 = self.f2(x_norm)
        gated = self.gate_act(f1) * f2
        out = self.map(gated)
        return out

# =========================================================================
# 6. 组合模块: AdaptiveDWCBlock & FreqProcessingUnit (双中心逻辑)
# =========================================================================
class AdaptiveDWCBlock(nn.Module):
    def __init__(self, channel, n_clusters=16):
        super(AdaptiveDWCBlock, self).__init__()
        self.clustering = DifferentiableSoftKMeans(channel, n_clusters=n_clusters)
        
        self.dwc_ll = PWAC(channel, channel, n_clusters=n_clusters)
        self.dwc_lh = PWAC(channel, channel, n_clusters=n_clusters)
        self.dwc_hl = PWAC(channel, channel, n_clusters=n_clusters)
        self.dwc_hh = PWAC(channel, channel, n_clusters=n_clusters)

    def forward(self, ll, lh, hl, hh):
        # 1. 聚类: 只返回 labels
        labels = self.clustering(ll)
        
        # 2. PWAC: 残差连接
        ll_out = self.dwc_ll(ll, labels) + ll 
        lh_out = self.dwc_lh(lh, labels) + lh
        hl_out = self.dwc_hl(hl, labels) + hl
        hh_out = self.dwc_hh(hh, labels) + hh
        
        return ll_out, lh_out, hl_out, hh_out, labels

class FreqProcessingUnit(nn.Module):
    def __init__(self, pan_channel, ms_channel, head_channel, dropout, n_clusters=16, max_pixels=4096):
        super(FreqProcessingUnit, self).__init__()
        self.WD = DWT_2D()
        self.clustering = DifferentiableSoftKMeans(pan_channel, n_clusters=n_clusters)
        self.n_clusters = n_clusters
        self.max_pixels = max_pixels
        
        self.v_ll_attn = AdaptiveAttention(ms_channel, head_channel, dropout, n_clusters)
        self.v_lh_attn = AdaptiveAttention(ms_channel, head_channel, dropout, n_clusters)
        self.v_hl_attn = AdaptiveAttention(ms_channel, head_channel, dropout, n_clusters)
        self.v_hh_attn = AdaptiveAttention(ms_channel, head_channel, dropout, n_clusters)
        
        self.conv_idwt_up = conv_IDWT(channel=ms_channel)
        self.combine = combine(channel=ms_channel)
        self.resblock = resblock(channel=ms_channel)
        self.mlp_back = FFN(in_channel=ms_channel, FFN_channel=ms_channel // 2, out_channel=ms_channel)
        
        # 每个 query-key 对的可学习系数矩阵 [max_pixels, max_pixels]
        self.pixel_coefficients_ll = nn.Parameter(torch.ones(max_pixels, max_pixels))
        self.pixel_coefficients_lh = nn.Parameter(torch.ones(max_pixels, max_pixels))
        self.pixel_coefficients_hl = nn.Parameter(torch.ones(max_pixels, max_pixels))
        self.pixel_coefficients_hh = nn.Parameter(torch.ones(max_pixels, max_pixels))

    def forward(self, pan_bands, ms_img, back_img, external_labels=None):
        wd_ll, wd_lh, wd_hl, wd_hh = pan_bands
        
        pre_v = self.combine(x1=wd_ll, x2=ms_img, x3=self.mlp_back(back_img))
        v = self.resblock(pre_v) # MS 域特征
        
        # 1. 获取分组结构
        if external_labels is not None:
            labels = external_labels
        else:
            labels = self.clustering(wd_ll)
            
        B, C, H, W = v.shape
        N = H * W
        
        # 2. 获取每个 query-key 对的系数矩阵 [N, N]，然后扩展到 [B, N, N]
        pixel_coeff_ll = self.pixel_coefficients_ll[:N, :N]  # [N, N]
        pixel_coeff_ll = pixel_coeff_ll.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
        pixel_coeff_lh = self.pixel_coefficients_lh[:N, :N]  # [N, N]
        pixel_coeff_lh = pixel_coeff_lh.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
        pixel_coeff_hl = self.pixel_coefficients_hl[:N, :N]  # [N, N]
        pixel_coeff_hl = pixel_coeff_hl.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
        pixel_coeff_hh = self.pixel_coefficients_hh[:N, :N]  # [N, N]
        pixel_coeff_hh = pixel_coeff_hh.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        # 3. Attention - 在 AdaptiveAttention 内部计算 centers_k
        v_ll = self.v_ll_attn(wd_ll, wd_ll, v, labels, pixel_coeff_ll)
        v_lh = self.v_lh_attn(wd_lh, wd_ll, v, labels, pixel_coeff_lh)
        v_hl = self.v_hl_attn(wd_hl, wd_ll, v, labels, pixel_coeff_hl)
        v_hh = self.v_hh_attn(wd_hh, wd_ll, v, labels, pixel_coeff_hh)
        
        v_idwt = self.conv_idwt_up(torch.cat([v_ll, v_lh, v_hl, v_hh], dim=1))
        return v_idwt

# =========================================================================
# 7. 主网络架构 (保持不变)
# =========================================================================
class S_MWiT(nn.Module):
    def __init__(self, pan_ll_channel, L_up_channel, head_channel, dropout):
        super(S_MWiT, self).__init__()
        self.pan_ll_channel = pan_ll_channel
        self.WD = DWT_2D()
        
        self.dwc_block1 = AdaptiveDWCBlock(pan_ll_channel)
        self.dwc_block2 = AdaptiveDWCBlock(pan_ll_channel)
        self.dwc_block3 = AdaptiveDWCBlock(pan_ll_channel)
        
        self.conv_idwt_pan = conv_IDWT(channel=pan_ll_channel)
        
        self.freq_unit_1 = FreqProcessingUnit(pan_ll_channel, L_up_channel, head_channel, dropout)
        self.gdfn = GDFN(L_up_channel)
        self.freq_unit_2 = FreqProcessingUnit(pan_ll_channel, L_up_channel, head_channel, dropout)
        
        self.conv_x = FFN_2(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.conv_v = FFN_2(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.resblock_1 = resblock(channel=L_up_channel)

    def forward(self, pan_ll, L_up, back_img):
        wd_ll, wd_lh, wd_hl, wd_hh = torch.split(self.WD(pan_ll), [self.pan_ll_channel]*4, dim=1)
        pan_bands = (wd_ll, wd_lh, wd_hl, wd_hh)

        # 空间域处理: 提取 labels_1
        s_ll, s_lh, s_hl, s_hh, labels_1 = self.dwc_block1(wd_ll, wd_lh, wd_hl, wd_hh) 
        s_ll, s_lh, s_hl, s_hh, _        = self.dwc_block2(s_ll, s_lh, s_hl, s_hh) 
        s_ll, s_lh, s_hl, s_hh, _        = self.dwc_block3(s_ll, s_lh, s_hl, s_hh) 
        
        x_idwt = self.conv_idwt_pan(torch.cat([s_ll, s_lh, s_hl, s_hh], dim=1))
        
        # 频率域处理 Round 1: 复用 labels_1
        v_idwt_1 = self.freq_unit_1(pan_bands, L_up, back_img, external_labels=labels_1)
        
        gdfn_out = self.gdfn(v_idwt_1)
        gdfn_ll, gdfn_lh, gdfn_hl, gdfn_hh = torch.split(self.WD(gdfn_out), [self.pan_ll_channel]*4, dim=1)
        gdfn_bands = (gdfn_ll, gdfn_lh, gdfn_hl, gdfn_hh)
        
        # 频率域处理 Round 2: 独立计算
        v_idwt_2 = self.freq_unit_2(gdfn_bands, L_up, back_img, external_labels=None)

        x_1 = self.conv_x(x_idwt) + self.conv_v(v_idwt_2)
        x = self.resblock_1(x_1)
        return x

class L_MWiT(S_MWiT):
    def __init__(self, pan_ll_channel, L_up_channel, head_channel, dropout):
        super(L_MWiT, self).__init__(pan_ll_channel, L_up_channel, head_channel, dropout)
    
    def forward(self, pan_ll, back_img, L_up):
        return super().forward(pan_ll, L_up, back_img)

# =========================================================================
# 8. 辅助类 (保持不变)
# =========================================================================
class FFN(nn.Module):
    def __init__(self, in_channel, FFN_channel, out_channel):
        super(FFN, self).__init__()
        self.FFN_channel, self.out_channel = FFN_channel, out_channel
        self.linear_1 = nn.Linear(in_channel, FFN_channel)
        self.conv1 = nn.Conv2d(FFN_channel, FFN_channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(FFN_channel, FFN_channel, 1, 1, 0, bias=True)
        self.linear_2 = nn.Linear(FFN_channel, out_channel)
        self.act = nn.PReLU(num_parameters=FFN_channel, init=0.01)
    def forward(self, x):
        B, C, H, W = x.shape
        rs1 = self.linear_1(x.permute(0, 2, 3, 1).reshape(B, -1, C)).permute(0, 2, 1).reshape(B, self.FFN_channel, H, W)
        rs2 = self.act(self.conv1(rs1))
        rs3 = self.conv2(rs2) + rs1
        rs4 = self.linear_2(rs3.permute(0, 2, 3, 1).reshape(B, -1, self.FFN_channel)).permute(0, 2, 1).reshape(B, self.out_channel, H, W)
        return rs4

class FFN_2(nn.Module):
    def __init__(self, in_channel, FFN_channel, out_channel):
        super(FFN_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, FFN_channel, 3, 1, 2, bias=True, dilation=2)
        self.conv2 = nn.Conv2d(FFN_channel, FFN_channel, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(FFN_channel, FFN_channel, 1, 1, 0, bias=True)
        self.conv4 = nn.Conv2d(FFN_channel, out_channel, 3, 1, 1, bias=True)
        self.act = nn.PReLU(num_parameters=FFN_channel, init=0.01)
    def forward(self, x):
        rs1 = self.conv1(x)
        rs2 = self.act(self.conv2(rs1))
        rs3 = self.conv3(rs2) + rs1
        rs4 = self.conv4(rs3)
        return rs4

class conv_IDWT(nn.Module):
    def __init__(self, channel):
        super(conv_IDWT, self).__init__()
        self.res_block = resblock(channel=channel)
        self.IDWT = IDWT_2D()
    def forward(self, x):
        rs1 = self.IDWT(x)
        rs2 = self.res_block(rs1)
        return rs2

class resblock(nn.Module):
    def __init__(self, channel):
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.act = nn.PReLU(num_parameters=channel, init=0.01)
    def forward(self, x):
        rs1 = self.act(self.conv1(x))
        rs2 = self.conv2(rs1) + x
        return rs2

class combine(nn.Module):
    def __init__(self, channel):
        super(combine, self).__init__()
        self.resblock = resblock(channel=channel)
        self.a = nn.Parameter(torch.tensor(0.33), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(0.33), requires_grad=True)
    def forward(self, x1, x2, x3):
        rs1 = self.a * x1 + self.b * x2 + (1 - self.a - self.b) * x3
        rs2 = self.resblock(rs1)
        return rs2

class raise_channel(nn.Module):
    def __init__(self, in_channel, target_channel):
        super(raise_channel, self).__init__()
        self.raise_conv = nn.Sequential(
            nn.Conv2d(in_channel, target_channel, 5, 1, 2, bias=True),
            nn.PReLU(num_parameters=target_channel, init=0.01),
            nn.Conv2d(target_channel, target_channel, 3, 1, 1, bias=True),
        )
    def forward(self, x): return self.raise_conv(x)

class reduce_channel(nn.Module):
    def __init__(self, ms_target_channel, L_up_channel):
        super(reduce_channel, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(ms_target_channel, ms_target_channel, 3, 1, 1, bias=True),
            nn.PReLU(num_parameters=ms_target_channel, init=0.01),
            nn.Conv2d(ms_target_channel, L_up_channel, 3, 1, 1, bias=True),
            nn.Conv2d(L_up_channel, L_up_channel, 3, 1, 1, bias=True),
        )
    def forward(self, x): return self.reduce_conv(x)

# =========================================================================
# 9. HWViT (网络入口)
# =========================================================================
class HWViT(nn.Module):
    def __init__(self, L_up_channel=32, pan_channel=1, pan_target_channel=32, ms_target_channel=32, head_channel=8, dropout=0):
        super(HWViT, self).__init__()
        self.pan_channel = pan_channel
        self.lms = nn.Sequential(
            nn.Conv2d(L_up_channel, L_up_channel * 16, 3, 1, 1, bias=True),
            nn.PixelShuffle(4),
        )
        self.pan_raise_channel = raise_channel(in_channel=pan_channel, target_channel=pan_target_channel)
        self.lms_raise_channel = raise_channel(in_channel=L_up_channel, target_channel=ms_target_channel)
        self.ms_raise_channel = raise_channel(in_channel=L_up_channel, target_channel=ms_target_channel)
        self.reduce_channel = reduce_channel(ms_target_channel=ms_target_channel, L_up_channel=L_up_channel)
        
        # Layer 1: Small Scale
        self.L_MWiT_block = L_MWiT(L_up_channel=ms_target_channel, pan_ll_channel=pan_target_channel, head_channel=head_channel, dropout=dropout)
        # Layer 2: Full Scale
        self.F_MWiT_block = S_MWiT(pan_ll_channel=pan_target_channel, L_up_channel=ms_target_channel, head_channel=head_channel, dropout=dropout)
        
        self.lms_down_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lms_down_4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pan_down_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.act_1 = nn.PReLU(num_parameters=L_up_channel, init=0.01)
        self.act_2 = nn.PReLU(num_parameters=L_up_channel, init=0.01)

    def forward(self, pan, ms, lms):
        pan = self.pan_raise_channel(pan)
        lms_1 = self.act_1(self.lms(ms) + lms)
        lms_2 = self.lms_raise_channel(lms_1)
        
        # Layer 1
        back_1 = self.L_MWiT_block(pan_ll=self.pan_down_2(pan), L_up=self.lms_down_4(lms_2), back_img=self.ms_raise_channel(ms))
        
        # Layer 2
        back_2 = self.F_MWiT_block(pan_ll=pan, L_up=self.lms_down_2(lms_2), back_img=back_1)
        
        back = self.reduce_channel(back_2)
        result = self.act_2(back + lms_1)
        return result