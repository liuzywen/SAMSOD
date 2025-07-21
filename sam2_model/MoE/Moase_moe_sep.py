# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class BiasedDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(BiasedDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, x, p2, largest, dim=2):
        if dim >= len(x.shape):
            raise ValueError("Dimension {} out of range for input with shape {}".format(dim, x.shape))

        values_2, indices_2 = torch.topk(x, int(x.shape[dim] * p2), dim=dim, largest=largest)
        result_2 = torch.zeros_like(x)
        result_2.scatter_(dim, indices_2, values_2)
        return result_2


class Adapter(nn.Module):
    def __init__(self,
                 in_dim,
                 mid_dim,
                 scale_value,
                 init_option="lora",
                 drop="our"):
        super().__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.scale = scale_value
        self.down_proj = nn.Linear(self.in_dim, self.mid_dim)
        self.non_linear_func = nn.GELU()
        self.up_proj = nn.Linear(self.mid_dim, self.in_dim)
        self.drop = drop

        self.ourdropout = BiasedDropout(p=0.5)
        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, p2, largest=False):

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = self.ourdropout(down, p2, largest)

        up = self.up_proj(down)
        up = up * self.scale
        output = up

        return output


class KeepTopK(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        # x: [b, n, d]
        if self.top_k == 0:
            return x

        filter_value = -float('Inf')
        indices_to_remove = x < torch.topk(x, self.top_k)[0][..., -1, None]  # topk返回value的最内层大小比较
        x[indices_to_remove] = filter_value

        return x

class Moe_Adapter(nn.Module):
    def __init__(self, in_dim, mid_dim, scale_value, drop="our"):
        super(Moe_Adapter, self).__init__()
        TopK_Function = KeepTopK(top_k=int(2))
        self.router = nn.Sequential(
            nn.Linear(in_dim, 2),
            TopK_Function,
            nn.Softmax(dim=-1)
        )
        self.router_2 = nn.Sequential(
            nn.Linear(in_dim, 2),
            TopK_Function,
            nn.Softmax(dim=-1)
        )
        self.adaptmlp = Adapter(in_dim=in_dim, mid_dim=mid_dim, scale_value=scale_value, drop=drop)
        self.adaptmlp1 = Adapter(in_dim=in_dim, mid_dim=mid_dim, scale_value=scale_value, drop=drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.resize(B, H * W, C)
        weight_idx2 = self.router_2(x)

        weights = self.router(x)
        weight_idx0 = weights[:, :, 0].unsqueeze(dim=-1)
        biased_threshold0 = torch.mean(weight_idx0)
        weight_idx1 = weights[:, :, 1].unsqueeze(dim=-1)
        biased_threshold1 = torch.mean(weight_idx1)

        hign_expert_value0 = weight_idx2[:, :, 0].unsqueeze(dim=-1)*self.adaptmlp(x, p2=0.45 + 0.1*biased_threshold0, largest=True)

        low_expert_value1 = weight_idx2[:, :, 1].unsqueeze(dim=-1)*self.adaptmlp1(x, p2=0.35 + 0.1*biased_threshold1, largest=False)

        expert_value = low_expert_value1 + hign_expert_value0

        expert_value = expert_value.resize(B, H, W, C)
        return expert_value
if __name__ == '__main__':
    x = torch.randn(1,14,14,96)
    model = Moe_Adapter(96,32,0.5)
    y = model(x)
    print(y)