import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
import ipdb
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import math
#from hilbert import decode, encode
from pyzorder import ZOrderIndexer

#Done by Gustavo Scheidt
#This file contains ConvBlock3D, VMUNet3D, SS3D_v5, VSSBlock3D, VSSLayer3D, PatchEmbedding/Expanding/Merging3D
    
class ConvBlock3D(nn.Module):           #From MedSegMamba, VM-UNet uses Conv2D
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock3D, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = ConvBlock3D(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

#Adapted from the VM-UNET code
class VSSLayer_up3D(nn.Module):
    """Upsampling VSSLayer3D (mirror of encoder), used in decoder stages."""
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        mlp_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        upsample=None,
        use_checkpoint=False,
        d_state=64,                 #increased from d_16
        expansion_factor=1,
        scan_type='scan',
        size=12,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock3D_v5(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                expansion_factor=expansion_factor,
                mlp_drop_rate=mlp_drop,
                scan_type=scan_type,
                size=size,
                orientation=i % 6,
            )
            for i in range(depth)
        ])

        if upsample is not None:
            self.upsample = upsample(in_channels=dim)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class Final_PatchExpand3D(nn.Module):
    """Final upsampling layer (like in original VM-UNet final_patch_expand)"""
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        #Adapted from nn.Linear to nn.ConvTranspose3d to make sure the volumetric data is correctly processed
        self.up = nn.ConvTranspose3d(dim, dim // dim_scale, kernel_size=dim_scale, stride=dim_scale)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.up(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)
        x = self.norm(x)
        return x

#VSSM class from the original VM-UNet with the necessary modifications
class VSSM_3D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=64, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed3D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        self.ape = False
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer3D(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up3D(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand3D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand3D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv3d(dims_decoder[-1]//4, num_classes, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
           
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
        
    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])
        return x

    def forward_final(self, x):
        x = self.final_up(x)
        # changed the permutation to adapt to the 3D
        x = x.permute(0, 4, 1, 2, 3)
        x = self.final_conv(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        return x


class VMUNet_3D(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM_3D(in_chans=input_channels,
                              num_classes=num_classes,
                              depths=depths,
                              depths_decoder=depths_decoder,
                              drop_path_rate=drop_path_rate)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)
        logits = self.vmunet(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits
        
	#Erased the function load_from because we don't use pre-initialized weights
    

#Blocks HSCANS, SS3D, VSSBlock, VSSLayer, FeedFoward 
class HSCANS(nn.Module):
    """""
    HSCANS reorders scan indices of 2D or 3D patches (e.g., Hilbert, zigzag),
    allowing the model to process input in different spatial orders to better capture local dependencies.
    """""
    #I removed all uses of the hilbert scan due to library incompatibility issues, something to re-add in the future

    def __init__(self, size=12, dim=3, scan_type='scan', ):
        super().__init__()
        size = int(size)
        max_num = size ** dim
        indexes = np.arange(max_num)
        self.dim=dim
        if 'sweep' == scan_type:  # ['sweep', 'scan', 'zorder', 'zigzag', 'hilbert']
            locs_flat = indexes
        elif 'scan' == scan_type:
            if dim == 2:
                indexes = indexes.reshape(size, size)
                for i in np.arange(1, size, step=2):
                    indexes[i, :] = indexes[i, :][::-1]
                locs_flat = indexes.reshape(-1)
            elif dim==3:
                indexes = indexes.reshape(size, size, size)
                for i in np.arange(1, size, step=2):
                    indexes[:, i, :] = np.flip(indexes[:, i, :], axis=1) # flipping y
                for j in np.arange(1, size, step=2):
                    indexes[j, :, :] = np.flip(indexes[j, :, :], axis=(0,1))
                locs_flat = indexes.reshape(-1)
        elif 'zorder' == scan_type:
            zi = ZOrderIndexer((0, size - 1), (0, size - 1))
            locs_flat = []
            for z in indexes:
                r, c = zi.rc(int(z))
                locs_flat.append(c * size + r)
            locs_flat = np.array(locs_flat)
        elif 'zigzag' == scan_type:
            indexes = indexes.reshape(size, size)
            locs_flat = []
            for i in range(2 * size - 1):
                if i % 2 == 0:
                    start_col = max(0, i - size + 1)
                    end_col = min(i, size - 1)
                    for j in range(start_col, end_col + 1):
                        locs_flat.append(indexes[i - j, j])
                else:
                    start_row = max(0, i - size + 1)
                    end_row = min(i, size - 1)
                    for j in range(start_row, end_row + 1):
                        locs_flat.append(indexes[j, i - j])
            locs_flat = np.array(locs_flat)
        else:
            raise Exception('invalid encoder mode')
        locs_flat_inv = np.argsort(locs_flat)
        index_flat = torch.LongTensor(locs_flat.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        index_flat_inv = torch.LongTensor(locs_flat_inv.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        self.index_flat = nn.Parameter(index_flat, requires_grad=False)
        self.index_flat_inv = nn.Parameter(index_flat_inv, requires_grad=False)
    def flat_locs_hilbert(self, locs, num_dim, num_bit):
        ret = []
        l = 2 ** num_bit
        for i in range(len(locs)):
            loc = locs[i]
            loc_flat = 0
            for j in range(num_dim):
                loc_flat += loc[j] * (l ** j)
            ret.append(loc_flat)
        return np.array(ret).astype(np.uint64)
    def __call__(self, img):
        img_encode = self.encode(img)
        return img_encode
    def encode(self, img):
        img_encode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat_inv.expand(img.shape), img)
        return img_encode
    def decode(self, img):
        img_decode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(2, self.index_flat.expand(img.shape), img)
        return img_decode

class SS3D_v5(nn.Module): #no multiplicative path, the better version described in VMamba
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1, #2
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=None,
            einsum=True,
            size=12, 
            scan_type='scan',#size needs to be a power of 2 to use hilbert
            num_direction = 8,
            orientation = 0, #0, 1, 2
            **kwargs,
            ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.orientation = orientation
        self.d_model = d_model #channel dim, 512 or 1024, gets expanded
        self.d_state = d_state
        
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        if einsum:
            self.x_proj = (
                    nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for i in range(num_direction)
            )
            self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=8, N, inner) = (K=8, new_c = self.dt_rank + self.d_state * 2, C)
            del self.x_proj
        else:
            #print('no einsum for x_proj')
            raise Exception('have to use einsum for now lol')
        # figure out how to do dts without einsum
        self.dt_projs = [
                self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for i in range(num_direction)
                ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=num_direction, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=num_direction, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
        self.scans = HSCANS(size=size, scan_type=scan_type)

        # self.scans.encode = lambda x: x
        # self.scans.decode = lambda x: x

        self.num_direction = num_direction

        if (orientation%3) == 0:
            self.transp = lambda x: x
        elif (orientation%3) == 1:
            self.transp = lambda x: torch.transpose(x, dim0=2, dim1=3) # change to 3 4 if hilbert
        elif (orientation%3) == 2:
            self.transp = lambda x: torch.transpose(x, dim0=2, dim1=4) # scan goes across first dim
        self.transp2 = lambda x: x
        if (orientation%6) > 2: # 3, 4, 5
            self.transp2 = lambda x: torch.transpose(x, dim0=3, dim1=4)

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
    def A_log_init(d_state, d_inner, copies=8, device=None, merge=True):
        # S4D real initialization
        A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
        ).contiguous()
        #('A', A.shape)
        A_log = torch.log(A)    # Keep A_log in fp32

        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=8, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)    # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        #0,1, 2, 3, 4
        B, C, H, W, D = x.shape
        L = H * W * D
        K = self.num_direction
        xs = []

        xs.append(self.scans.encode(self.transp2(self.transp(x)).contiguous().view(B, -1, L)))
        
        xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(torch.rot90(x, k=1, dims=(3,4)), k=1, dims=(2,4)))).contiguous().view(B, -1, L)))
        xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(2,4)))).contiguous().view(B, -1, L)))
        xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(torch.rot90(x, k=-1, dims=(2,4)), k=1, dims=(2,3)))).contiguous().view(B, -1, L)))
        
        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(2,3)))).contiguous().view(B, -1, L)))
        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(2,4)))).contiguous().view(B, -1, L)))
        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(3,4)))).contiguous().view(B, -1, L)))
        
        xs = torch.stack(xs,dim=1).view(B, K // 2, -1, L)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (b, k, c, l)
                
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        
        xs = xs.float().view(B, -1, L) # (b, k * d, l)

        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)    # (k * d, d_state)
        
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
        ).view(B, K, -1, L)
        
        assert out_y.dtype == torch.float

        # out_y = xs.view(B, K, -1, L) # for testing

        inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
        ys = []

        # xs.append(self.scans.encode(self.transp2(self.transp(x)).contiguous().view(B, -1, L)))        
        ys.append(self.transp(self.transp2(self.scans.decode(out_y[:, 0]).view(B, -1, H, W, D))).contiguous().view(B, -1, L))
        ys.append(self.transp(self.transp2(self.scans.decode(inv_y[:, 0]).view(B, -1, H, W, D))).contiguous().view(B, -1, L))

        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(torch.rot90(x, k=1, dims=(3,4)), k=1, dims=(2,4)))).contiguous().view(B, -1, L)))
        ys.append(torch.rot90(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 1]).view(B, -1, H, W, D))), k=-1, dims=(2,4)), k=-1, dims=(3,4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 1]).view(B, -1, H, W, D))), k=-1, dims=(2,4)), k=-1, dims=(3,4)).contiguous().view(B, -1, L))

        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(x, k=2, dims=(2,4)))).contiguous().view(B, -1, L)))
        ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 2]).view(B, -1, H, W, D))), k=2, dims=(2,4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 2]).view(B, -1, H, W, D))), k=2, dims=(2,4)).contiguous().view(B, -1, L))

        # xs.append(self.scans.encode(self.transp2(self.transp(torch.rot90(torch.rot90(x, k=3, dims=(2,4)), k=1, dims=(2,3)))).contiguous().view(B, -1, L)))
        ys.append(torch.rot90(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 3]).view(B, -1, H, W, D))), k=-1, dims=(2,3)), k=1, dims=(2,4)).contiguous().view(B, -1, L))
        ys.append(torch.rot90(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 3]).view(B, -1, H, W, D))), k=-1, dims=(2,3)), k=1, dims=(2,4)).contiguous().view(B, -1, L))
        
        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 1]).view(B, -1, H, W, D))), k=2, dims=(2,3)).contiguous().view(B, -1, L))
        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 1]).view(B, -1, H, W, D))), k=2, dims=(2,3)).contiguous().view(B, -1, L))

        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 2]).view(B, -1, H, W, D))), k=2, dims=(2,4)).contiguous().view(B, -1, L))
        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 2]).view(B, -1, H, W, D))), k=2, dims=(2,4)).contiguous().view(B, -1, L))

        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(out_y[:, 3]).view(B, -1, H, W, D))), k=2, dims=(3,4)).contiguous().view(B, -1, L))
        # ys.append(torch.rot90(self.transp(self.transp2(self.scans.decode(inv_y[:, 3]).view(B, -1, H, W, D))), k=2, dims=(3,4)).contiguous().view(B, -1, L))
        
        # for y in ys:
        #     print(torch.all(y==x.view(B, -1, L)))
        return sum(ys)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, D, C = x.shape #!!!

        x = self.in_proj(x)

        x = x.permute(0, 4, 1, 2, 3).contiguous()        
        x = self.act(self.conv3d(x)) # (b, d, h, w)
        y = self.forward_core(x) # 1 1024 1728
        
        assert y.dtype == torch.float32
                
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, D, -1) #bcl > blc > bhwdc
        
        y = self.out_norm(y)
        
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out

#almost a 2D->3D version of the original VM-UNet
class VSSBlock3D_v5(nn.Module): #no multiplicative path, added MLP. more like transformer block used in TABSurfer now
  def __init__(
      self,
      hidden_dim: int = 0,
      drop_path: float = 0,
      norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
      attn_drop_rate: float = 0,
      d_state: int = 16,
      expansion_factor = 1, # can only be 1 for v3, no linear projection to increase channels
      mlp_drop_rate=0.,
      orientation = 0,
      scan_type = 'scan',
      size = 12,
      **kwargs,
      ):
    super().__init__()
    print(orientation, end='')
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = SS3D_v5(d_model=hidden_dim, 
                                  dropout=attn_drop_rate, 
                                  d_state=d_state, 
                                  expand=expansion_factor, 
                                  orientation=orientation, 
                                  scan_type=scan_type, 
                                  size=size,
                                  **kwargs)
    #the difference is that here we add a second layer norm for the MLP and the FeedFoward, like in a Transformer
    self.ln_2 = norm_layer(hidden_dim)
    #block MLP like a Transformer
    self.mlp = FeedForward(dim = hidden_dim, hidden_dim=expansion_factor*hidden_dim, dropout_rate = mlp_drop_rate)

    self.drop_path = DropPath(drop_path)

  def forward(self, input: torch.Tensor):
    x = input + self.drop_path(self.self_attention(self.ln_1(input)))
    #new block MLP post-attention
    x = x + self.drop_path(self.mlp(self.ln_2(x)))
    return x

#Exactly the same as the VSSLayer in the VM-UNet, just added the VSSBlock3D
class VSSLayer3D(nn.Module):
    """ A basic layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        mlp_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=64,                         #Increase the d_state (16 in the VM-UNet)
        expansion_factor = 1,               #Controls internal MLP width, not used in 2D
        scan_type = 'scan',                 #Defines 3D spatial scan 
        #orientation_order = None,          #Block-wise spatial orientation setting (we don't use any special config)
        size = 12,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        #Here we substitute the original VSSBlock for the 3D version (3D SSM + MLP + orientation)
        self.blocks = nn.ModuleList([
            VSSBlock3D_v5(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                expansion_factor=expansion_factor,
                mlp_drop_rate=mlp_drop,
                scan_type=scan_type,
                size = size,
                orientation=i%6, # 0 1 2 3 4 5 6 7 8 > 0 1 2 3 4 5 0 1 2
            )
            for i in range(depth)])
        
        #Custom init for projection weights (matches VM-UNet but never triggered by name)
        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
    
class FeedForward(nn.Module): #used in the VSSBlock3D 
    def __init__(self, dim, dropout_rate, hidden_dim = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim=dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)
#PatchExpanding, PatchMerging, PatchEmbedding
class PatchEmbed3D(nn.Module):
    """
    Aqui o PatchEmbed3d converte uma entrada volumétrica (B, C, D, H, W) para (B, C_out, D_out, H_out, W_out)
    usando Conv3d com kernel e stride = patch_size (default: 4)
    """
    def __init__(self, patch_size=(4, 4, 4), in_chans=4, embed_dim=32):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class PatchMerging3D(nn.Module):
    """
    Downsample por meio de concatenação de vizinhos e projeção linear:
    (B, C, D, H, W) -> (B, 2C, D/2, H/2, W/2)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.reduction = nn.Conv3d(in_channels * 8, in_channels * 2, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, "Input dimensions must be even"

        x0 = x[:, :, 0::2, 0::2, 0::2]
        x1 = x[:, :, 0::2, 0::2, 1::2]
        x2 = x[:, :, 0::2, 1::2, 0::2]
        x3 = x[:, :, 0::2, 1::2, 1::2]
        x4 = x[:, :, 1::2, 0::2, 0::2]
        x5 = x[:, :, 1::2, 0::2, 1::2]
        x6 = x[:, :, 1::2, 1::2, 0::2]
        x7 = x[:, :, 1::2, 1::2, 1::2]

        x_cat = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=1)
        x_out = self.reduction(x_cat)
        return x_out


class PatchExpand3D(nn.Module):
    """
    Upsample via ConvTranspose3d (trilinear alternativa):
    (B, C, D, H, W) -> (B, C/2, 2D, 2H, 2W)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.expand = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x):
        return self.expand(x)

