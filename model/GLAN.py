import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange
import numbers
# from model.layer import TIS
from model.block import TIS
from visualization import visualize_feature_map
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class MS_CAM(nn.Module):
    """
     多尺度通道注意力，结合全局注意力和局部注意力，使得模型不会丢失小区域的细节信息
    """
    def __init__(self,in_channels,ratio=16,bias=False):
        super().__init__()
        inter_channels = int(in_channels//ratio)
        self.local_attn = nn.Sequential(
            nn.Conv2d(in_channels,inter_channels,1,1,0,bias=bias),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels,in_channels,1,1,0,bias=bias),
            
        )
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,inter_channels,1,1,0,bias=bias),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels,in_channels,1,1,0,bias=bias),
        )
       
    def forward(self,x):
        xl = self.local_attn(x)
        xg = self.global_attn(x)
        xlg = xl + xg
        
        return x*torch.sigmoid(xlg)

class PAM(nn.Module):

    def __init__(self, n_feats, k_size=1):
        super(PAM, self).__init__()
        self.k1 = nn.Conv2d(n_feats, n_feats, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k2 = nn.Conv2d(n_feats, n_feats, 1) 
        self.k3 = nn.Conv2d(n_feats, n_feats, 1) 

    def forward(self, x):
        y = self.k1(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k2(x), y)
        out = self.k3(out)

        return out    

class MSCFM(nn.Module):
    def __init__(self, n_feats,kSize=3 ,scales=4):
        super(MSCFM, self).__init__()
        
        self.confusion_head = nn.Conv2d(n_feats, n_feats, 1, stride=1)
        
        # the first branch
        self.conv3_1_1 = nn.Conv2d(n_feats//scales, n_feats//scales, kSize, stride=1, padding=(kSize-1)//2)
        self.conv3_1_lrelu_1 = nn.LeakyReLU(negative_slope = 0.05,inplace=True)
        # the second branch
        self.conv3_2_1 = nn.Conv2d(n_feats//scales, n_feats//scales, kSize, stride=1, padding=(kSize-1)//2)
        self.conv3_2_lrelu_1 = nn.LeakyReLU(negative_slope = 0.05,inplace=True)
        self.conv3_2_2 = nn.Conv2d(n_feats//scales, n_feats//scales, kSize, stride=1, padding=(kSize-1)//2)
        self.conv3_2_lrelu_2 = nn.LeakyReLU(negative_slope = 0.05,inplace=True)
        
        # the third branch
        self.conv3_3_1 = nn.Conv2d(n_feats//scales, n_feats//scales, kSize, stride=1, padding=(kSize-1)//2)
        self.conv3_3_lrelu_1 = nn.LeakyReLU(negative_slope = 0.05,inplace=True)
        self.conv3_3_2 = nn.Conv2d(n_feats//scales, n_feats//scales, kSize, stride=1, padding=(kSize-1)//2)
        self.conv3_3_lrelu_2 = nn.LeakyReLU(negative_slope = 0.05,inplace=True)
        self.conv3_3_3 = nn.Conv2d(n_feats//scales, n_feats//scales, kSize, stride=1, padding=(kSize-1)//2)
        self.conv3_3_lrelu_3 = nn.LeakyReLU(negative_slope = 0.05,inplace=True)
        # the fourth branch
        
        self.conv3_4_1 = nn.Conv2d(n_feats//scales, n_feats//scales, kSize, stride=1, padding=(kSize-1)//2)
        self.conv3_4_lrelu_1 = nn.LeakyReLU(negative_slope = 0.05,inplace=True)
        self.conv3_4_2 = nn.Conv2d(n_feats//scales, n_feats//scales, kSize, stride=1, padding=(kSize-1)//2)
        self.conv3_4_lrelu_2 = nn.LeakyReLU(negative_slope = 0.05,inplace=True)
        self.conv3_4_3 = nn.Conv2d(n_feats//scales, n_feats//scales, kSize, stride=1, padding=(kSize-1)//2)
        self.conv3_4_lrelu_3 = nn.LeakyReLU(negative_slope = 0.05,inplace=True)
        self.conv3_4_4 = nn.Conv2d(n_feats//scales, n_feats//scales, kSize, stride=1, padding=(kSize-1)//2)
        self.conv3_4_lrelu_4 = nn.LeakyReLU(negative_slope = 0.05,inplace=True)
        
        # self.lrelu = nn.LeakyReLU(negative_slope = 0.05,inplace=True)
        self.pam = PAM(n_feats=n_feats,k_size=1)
        self.cca = MS_CAM(in_channels=n_feats)
        self.confusion_tail = nn.Conv2d(n_feats, n_feats, 1,stride = 1,padding=0)

    def forward(self, x):
        Low = self.confusion_head(x)
        X = torch.split(Low,16,dim=1)
        x1 = X[0]
        x2 = X[1]
        x3 = X[2]
        x4 = X[3]

        #the first branch x1 ->y1
        y1 = self.conv3_1_lrelu_1(self.conv3_1_1(x1))
        # y1 = self.pam(y1)
        # the second branch: x2+y1 ->Conv() -> y2 
        y2 = self.conv3_2_lrelu_1(self.conv3_2_1(x2))
        y2 = self.conv3_2_lrelu_2(self.conv3_2_2(y2+y1))
        # y2 = self.pam(y2)
        # the third branch: x3 -> Conv() -> y3+y2 -> Conv() -> y3
        y3 = self.conv3_3_lrelu_1(self.conv3_3_1(x3))
        y3 = self.conv3_3_lrelu_2(self.conv3_3_2(y3))
        y3 = self.conv3_3_lrelu_3(self.conv3_3_3(y3+y2))
        # y3 = self.pam(y3)
        # the fourth branch: x4 ->Conv() ->y4 -> Conv() -> y4+y3 -> Conv() -> y4
        y4 = self.conv3_4_lrelu_1(self.conv3_4_1(x4))
        y4 = self.conv3_4_lrelu_2(self.conv3_4_2(y4))
        y4 = self.conv3_4_lrelu_3(self.conv3_4_3(y4))
        y4 = self.conv3_4_lrelu_4(self.conv3_4_4(y4+y3)) #local residual connection
        # y4 = self.pam(y4)
        # concat multi-scale feature   
        out = torch.cat([y1,y2,y3,y4], 1)
        
        # out = channel_shuffle(out, 4)
        out = self.confusion_tail(self.cca(out))
        return  self.pam(out+x)   
 
# LayerNorm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


## Gated-Dconv Feed-Forward Network (GDFN)
class GFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GFeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


    


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim=48, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = GFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    
class GLAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #c2wh= dict([(64,56), (128,28), (256,14) ,(512,7)])
        self.fm1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
        
        self.msfb1 = MSCFM(n_feats=64)
        self.msfb2 = MSCFM(n_feats=64)
        self.msfb3 = MSCFM(n_feats=64)
        self.msfb4 = MSCFM(n_feats=64)
        self.msfb5 = MSCFM(n_feats=64)
        self.msfb6 = MSCFM(n_feats=64)


        self.transformer1 = TransformerBlock(dim=64)
        self.transformer2 = TransformerBlock(dim=64)
        self.transformer3 = TransformerBlock(dim=64)
        self.transformer4 = TransformerBlock(dim=64)
        self.transformer5 = TransformerBlock(dim=64)
        self.transformer6 = TransformerBlock(dim=64)
 
        # refer to fcanet dict([(64,56), (128,28), (256,14) ,(512,7)])   https://github.com/cfzd/FcaNet.git
        # our feature channel is set 64, f_h and f_h is set 56 according to the above dict
        self.tis1 = TIS(n_feats=64)
        self.tis2 = TIS(n_feats=64)
        self.tis3 = TIS(n_feats=64)
        self.tis4 = TIS(n_feats=64)
        self.tis5 = TIS(n_feats=64)
        self.tis6 = TIS(n_feats=64)

       
        self.fm_aggre =  nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        
        self.out_enhanced= nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,stride=1,padding=1)


    def forward(self, x):
        fm1 = self.fm1(x) 

        out1 = self.msfb1(fm1) # high-frequency
        out2 = self.transformer1(fm1)# low-frequency
        out1,out2 = self.tis1(out1,out2)
        
        out1 = self.msfb2(out1) # high-frequency
        out2 = self.transformer2(out2)# low-frequency
        out1,out2 = self.tis2(out1,out2)
        
        out1 = self.msfb3(out1) # high-frequency
        out2 = self.transformer3(out2)# low-frequency
        out1,out2 = self.tis3(out1,out2)
        
        out1 = self.msfb4(out1) # high-frequency
        out2 = self.transformer4(out2)# low-frequency
        out1,out2 = self.tis4(out1,out2)
        

        out1 = self.msfb5(out1) # high-frequency
        out2 = self.transformer5(out2)# low-frequency
        out1,out2 = self.tis5(out1,out2)
        
        out1 = self.msfb6(out1) # high-frequency
        out2 = self.transformer6(out2)# low-frequency
        
        out1,out2 = self.tis6(out1,out2)
        out = self.fm_aggre(out1+out2)
        out = self.out_enhanced(out)
        return out
