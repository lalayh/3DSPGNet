# original version from
# https://github.com/asyml/vision-transformer-pytorch/

import torch


class MlpBlock(torch.nn.Module):
    """Transformer Feed-Forward Block"""

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.0):
        super().__init__()

        # init layers
        self.fc1 = torch.nn.Linear(in_dim, mlp_dim)
        self.fc2 = torch.nn.Linear(mlp_dim, out_dim)
        self.act = torch.nn.ReLU(True)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class EncoderBlock(torch.nn.Module):
    def __init__(
        self, in_dim, num_heads, mlp_dim, dropout_rate=0.0, attn_dropout_rate=0.0  # 80,2,160,0,0
    ):
        super().__init__()

        # self.norm1 = torch.nn.LayerNorm(in_dim)
        # self.attn = torch.nn.MultiheadAttention(in_dim, num_heads)
        # if dropout_rate > 0:
        #     self.dropout = torch.nn.Dropout(dropout_rate)
        # else:
        #     self.dropout = None
        self.norm2 = torch.nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x, mask=None):
        residual = x  # 形状(1,1000,80)

        # x = self.norm1(x)  # 形状(1,1000,80)  # 注意力机制开始
        # x, attn_weights = self.attn(x, x, x, attn_mask=mask, need_weights=False)  # 形状(1,1000,80),
        # # print(x.shape)
        # if self.dropout is not None:
        #     x = self.dropout(x)
        # x += residual
        # residual = x  # 注意力机制结束

        x = self.norm2(x)
        x = self.mlp(x)
        x += residual
        return x  # 形状(1,1000,80)


class Transformer(torch.nn.Module):
    def __init__(
        self,
        emb_dim,  # 80
        mlp_dim,  # 160
        num_layers=1,  # 2
        num_heads=2,  # 2
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
    ):
        super().__init__()

        in_dim = emb_dim
        self.encoder_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(
                in_dim, num_heads, mlp_dim, dropout_rate, attn_dropout_rate  # 80,2,160,0,0
            )
            self.encoder_layers.append(layer)

        # self.num_heads = num_heads

    def forward(self, x, mask=None):  # 形状(1,1000,80)
        # if self.num_heads > 1:
        #     b, s, t = mask.shape  # 1000,1,1
        #     mask_n = mask.repeat(1, self.num_heads, 1).reshape(b * self.num_heads, s, t)  # 形状（2000,1,1）
        # else:
        #     mask_n = mask

        for layer in self.encoder_layers:
            # x = layer(x, mask=mask_n)  # 形状(1,1000,80)
            x = layer(x)

        return x
