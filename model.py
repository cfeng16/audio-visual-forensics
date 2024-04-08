import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class AViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., visual_len=0, audio_len=0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        #assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, 1, image_height, image_width))
        self.visual_modality_embeding = nn.Parameter(torch.randn(1, dim, 1, 1, 1))
        self.audio_modality_embeding = nn.Parameter(torch.randn(1, dim, 1))
        self.temporal_visual_embedding = nn.Parameter(torch.randn(1, dim, visual_len, 1, 1))
        self.temporal_audio_embedding = nn.Parameter(torch.randn(1, dim, audio_len))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, video, audio):
        video += self.pos_embedding
        video += self.temporal_visual_embedding
        video += self.visual_modality_embeding
        audio += self.temporal_audio_embedding
        audio += self.audio_modality_embeding
        b, d, _, _, _ = video.size()
        #video = video.reshape(b, -1, d) 
        video = video.permute([0, 2, 1])
        #audio = audio.reshape(b, -1, d)
        audio = audio.permute([0,2,1])
        x = torch.cat((audio, video), dim=1)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class MP_AViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., max_visual_len=0, max_audio_len=0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        #assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        #num_patches = (image_height // patch_height) * (image_width // patch_width)
        #patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        #self.to_patch_embedding = nn.Sequential(
        #    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #    nn.Linear(patch_dim, dim),
        #)
        #self.pos_embedding = nn.Parameter(torch.randn(1, dim, 1))
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, 1, image_height, image_width))
        self.visual_modality_embeding = nn.Parameter(torch.randn(1, dim, 1))
        self.audio_modality_embeding = nn.Parameter(torch.randn(1, dim, 1))
        self.temporal_visual_embedding = nn.Parameter(torch.randn(1, dim, max_visual_len))
        self.temporal_audio_embedding = nn.Parameter(torch.randn(1, dim, max_audio_len))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video, audio):
        video_ = []
        _, _, t_len, _, _ = video.size()
        _,_,aud_t_len = audio.size()
        video += self.pos_embedding
        for i in range(len(video)):
            dim_, t_, _, _ = video[i].size()
            video_.append(F.max_pool2d(video[i], kernel_size=video.size()[3:]).reshape(dim_, t_)[None,:])
        video = torch.cat(video_, dim=0)
        video += self.temporal_visual_embedding[:, :, :t_len]
        video += self.visual_modality_embeding
        audio += self.temporal_audio_embedding[:, :, :aud_t_len]
        audio += self.audio_modality_embeding
        b, d, _ = video.size()
        video = video.reshape(b, -1, d) 
        #video = video.permute([0, 2, 1])
        audio = audio.reshape(b, -1, d)
        #audio = audio.permute([0, 2, 1])
        x = torch.cat((audio, video), dim=1)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class MP_av_feature_AViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., max_visual_len=0, max_audio_len=0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        #assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        #num_patches = (image_height // patch_height) * (image_width // patch_width)
        #patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        #self.to_patch_embedding = nn.Sequential(
        #    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #    nn.Linear(patch_dim, dim),
        #)
        #self.pos_embedding = nn.Parameter(torch.randn(1, dim, 1))
        self.pos_embedding = nn.Parameter(torch.randn(1, dim, 1, image_height, image_width))
        self.visual_modality_embeding = nn.Parameter(torch.randn(1, dim, 1))
        self.audio_modality_embeding = nn.Parameter(torch.randn(1, dim, 1))
        self.temporal_visual_embedding = nn.Parameter(torch.randn(1, dim, max_visual_len))
        self.temporal_audio_embedding = nn.Parameter(torch.randn(1, dim, max_audio_len))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video, audio):
        video_ = []
        _, _, t_len, _, _ = video.size()
        _,_,aud_t_len = audio.size()
        video += self.pos_embedding
        for i in range(len(video)):
            dim_, t_, _, _ = video[i].size()
            video_.append(F.max_pool2d(video[i], kernel_size=video.size()[3:]).reshape(dim_, t_)[None,:])
        video = torch.cat(video_, dim=0)
        video += self.temporal_visual_embedding[:, :, :t_len]
        video += self.visual_modality_embeding
        audio += self.temporal_audio_embedding[:, :, :aud_t_len]
        audio += self.audio_modality_embeding
        b, d, _ = video.size()
        video = video.reshape(b, -1, d) 
        #video = video.permute([0, 2, 1])
        audio = audio.reshape(b, -1, d)
        #audio = audio.permute([0, 2, 1])

        # add maxpool1d for PCA dimension reduction
        video = video.permute([0, 2, 1])
        audio = audio.permute([0, 2, 1])
        video = F.max_pool1d(video, kernel_size=video.shape[2])
        video = video.squeeze(-1)
        audio = F.max_pool1d(audio, kernel_size=audio.shape[2])
        audio = audio.squeeze(-1)
        # added

        
        # x = torch.cat((audio, video), dim=1)

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = self.dropout(x)
        # x = self.transformer(x)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        # return self.mlp_head(x)
        
        return torch.cat((video, audio), dim=1)