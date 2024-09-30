import tinygrad.nn as nn
from tinygrad import Tensor
import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class MultiHeadAttention2D:
    def __init__(self, in_channels, out_channels, n_heads, kernel_size=1, stride=1, padding=0):
        assert in_channels % n_heads == 0, "in_channels must be divisible by n_heads"
        self.n_heads = n_heads
        self.k_channels = in_channels // n_heads  # Channels per head

        # Conv2D layers for Q, K, V
        self.conv_q = Tensor.randn(in_channels, in_channels, kernel_size, kernel_size)
        self.conv_k = Tensor.randn(in_channels, in_channels, kernel_size, kernel_size)
        self.conv_v = Tensor.randn(in_channels, in_channels, kernel_size, kernel_size)

        # Output projection with configurable out_channels
        self.conv_o = Tensor.randn(in_channels, out_channels, kernel_size, kernel_size)

        # Stride and padding for output spatial dimension control
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Get the input dimensions
        B, C, W, H = x.shape

        # Apply Conv2D for Q, K, V (with padding and stride)
        q = x.conv2d(self.conv_q, stride=self.stride, padding=self.padding)
        k = x.conv2d(self.conv_k, stride=self.stride, padding=self.padding)
        v = x.conv2d(self.conv_v, stride=self.stride, padding=self.padding)

        # Reshape Q, K, V for multi-head attention
        B, C_q, W_q, H_q = q.shape
        q = q.reshape(B, self.n_heads, self.k_channels, W_q * H_q)
        k = k.reshape(B, self.n_heads, self.k_channels, W_q * H_q)
        v = v.reshape(B, self.n_heads, self.k_channels, W_q * H_q)

        # Compute attention weights
        attention_scores = q @ k.transpose(-2, -1) / np.sqrt(self.k_channels)
        attention_weights = attention_scores.softmax(axis=-1)

        # Apply attention weights to V
        attn_output = attention_weights @ v
        attn_output = attn_output.reshape(B, C_q, W_q, H_q)

        # Project the output to the desired number of output channels
        out = attn_output.conv2d(self.conv_o, stride=1, padding=0)

        return out


class ResidualAttentionBlock:
    def __init__(self, in_channels, out_channels, n_heads, kernel_size=3, stride=1, padding=0, use_residual=True):
        self.attention = MultiHeadAttention2D(in_channels, out_channels, n_heads, kernel_size=kernel_size, stride=stride, padding=padding)

        self.use_residual = use_residual  # New parameter to control residual connections

        # Adjust for residual if shapes don't match (e.g., different channels, spatial sizes)
        if in_channels != out_channels or stride > 1:
            # 1x1 convolution to match channels, stride to match spatial dimensions
            self.residual_conv = Tensor.randn(in_channels, out_channels, 1, 1)
            self.needs_residual_conv = True
        else:
            self.needs_residual_conv = False

    def forward(self, x):
        # Multi-head attention layer
        attn_out = self.attention.forward(x)

        if self.use_residual:
            # Residual connection (skip connection)
            if self.needs_residual_conv:
                # Apply 1x1 convolution to match channels and dimensions
                res = x.conv2d(self.residual_conv, stride=self.attention.stride, padding=0)
            else:
                res = x  # If input and output shapes match, use identity
            return attn_out + res
        else:
            # Skip residual connection if not needed
            return attn_out


class CANNEncoder:
    def __init__(self, config):
        # Config is a list of dictionaries for each stage
        self.blocks = []
        for stage in config:
            self.blocks.append(ResidualAttentionBlock(
                in_channels=stage['in_channels'],
                out_channels=stage['out_channels'],
                n_heads=stage['n_heads'],
                kernel_size=stage.get('kernel_size', 3),
                stride=stage.get('stride', 1),
                padding=stage.get('padding', 1)
            ))

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x

class CANNDecoder:
    def __init__(self, config):
        # Config is a list of dictionaries for each stage (reverse of encoder)
        self.blocks = []
        for stage in config:
            self.blocks.append(ResidualAttentionBlock(
                in_channels=stage['in_channels'],
                out_channels=stage['out_channels'],
                n_heads=stage['n_heads'],
                kernel_size=stage.get('kernel_size', 3),
                stride=stage.get('stride', 1),
                padding=stage.get('padding', 1)
            ))

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x

class CANNAutoencoder:
    def __init__(self, encoder_config, decoder_config):
        self.encoder = CANNEncoder(encoder_config)
        self.decoder = CANNDecoder(decoder_config)

    def forward(self, x):
        # Encode the input
        latent_rep = self.encoder.forward(x)
        # Decode to reconstruct the input
        output = self.decoder.forward(latent_rep)
        return output


# class MultiHeadAttention:
#     def __init__(self, dim, num_heads=8):
#         super(MultiHeadAttention, self).__init__()
#         assert dim % num_heads == 0, "Dimension must be divisible by num_heads"
# 
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
# 
#         self.qkv = nn.Linear(dim, dim * 3, bias=False)  # Q, K, V in one projection
#         self.fc_out = nn.Linear(dim, dim)
# 
#     def forward(self, x):
#         B, N, C = x.shape  # Batch size, sequence length (H*W), channels
#         QKV = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
#         Q, K, V = QKV.permute(2, 0, 3, 1, 4)  # separate Q, K, V
# 
#         # Scaled dot-product attention
#         energy = torch.einsum('bhqd, bhkd -> bhqk', Q, K)  # QK^T
#         scaling = energy / (self.head_dim ** 0.5)
#         attention = torch.softmax(scaling, dim=-1)
# 
#         out = torch.einsum('bhqk, bhvd -> bhqd', attention, V)
#         out = out.reshape(B, N, C)
# 
#         # Apply the final linear layer
#         return self.fc_out(out)

# class MultiHeadAttention:
#   def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
#     assert channels % n_heads == 0
#     self.channels, self.out_channels, self.n_heads, self.p_dropout, self.window_size, self.heads_share, self.block_length, self.proximal_bias, self.proximal_init = channels, out_channels, n_heads, p_dropout, window_size, heads_share, block_length, proximal_bias, proximal_init
#     self.attn, self.k_channels  = None, channels // n_heads
#     self.conv_q, self.conv_k, self.conv_v = [nn.Conv1d(channels, channels, 1) for _ in range(3)]
#     self.conv_o = nn.Conv1d(channels, out_channels, 1)
#     if window_size is not None: self.emb_rel_k, self.emb_rel_v = [Tensor.randn(1 if heads_share else n_heads, window_size * 2 + 1, self.k_channels) * (self.k_channels ** -0.5) for _ in range(2)]
# 
#   def forward(self, x, c, attn_mask=None):
#     q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
#     x, self.attn = self.attention(q, k, v, mask=attn_mask)
#     return self.conv_o(x)
# 
#   def attention(self, query: Tensor, key: Tensor, value: Tensor, mask=None):# reshape [b, d, t] -> [b, n_h, t, d_k]
#     b, d, t_s, t_t = key.shape[0], key.shape[1], key.shape[2], query.shape[2]
#     query = query.reshape(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
#     key = key.reshape(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
#     value = value.reshape(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
#     scores = (query / math.sqrt(self.k_channels)) @ key.transpose(-2, -1)
#     if self.window_size is not None:
#       assert t_s == t_t, "Relative attention is only available for self-attention."
#       key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
#       rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
#       scores = scores + self._relative_position_to_absolute_position(rel_logits)
#     if mask is not None:
#       scores = Tensor.where(mask, scores, -1e4)
#       if self.block_length is not None:
#         assert t_s == t_t, "Local attention is only available for self-attention."
#         scores = Tensor.where(Tensor.ones_like(scores).triu(-self.block_length).tril(self.block_length), scores, -1e4)
#     p_attn = scores.softmax(axis=-1)  # [b, n_h, t_t, t_s]
#     output = p_attn.matmul(value)
#     if self.window_size is not None:
#       relative_weights = self._absolute_position_to_relative_position(p_attn)
#       value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
#       output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
#     output = output.transpose(2, 3).contiguous().reshape(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
#     return output, p_attn
# 
#   def _matmul_with_relative_values(self, x, y): return x.matmul(y.unsqueeze(0))                 # x: [b, h, l, m], y: [h or 1, m, d], ret: [b, h, l, d]
# 
#   def _matmul_with_relative_keys(self, x, y): return x.matmul(y.unsqueeze(0).transpose(-2, -1)) # x: [b, h, l, d], y: [h or 1, m, d], re, : [b, h, l, m]
# 
#   def _get_relative_embeddings(self, relative_embeddings, length):
#     pad_length, slice_start_position = max(length - (self.window_size + 1), 0), max((self.window_size + 1) - length, 0)
#     padded_relative_embeddings = relative_embeddings if pad_length <= 0\
#       else relative_embeddings.pad(convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
#     return padded_relative_embeddings[:, slice_start_position:(slice_start_position + 2 * length - 1)] #used_relative_embeddings
# 
#   def _relative_position_to_absolute_position(self, x: Tensor): # x: [b, h, l, 2*l-1] -> [b, h, l, l]
#     batch, heads, length, _ = x.shape
#     x = x.pad(convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))
#     x_flat = x.reshape([batch, heads, length * 2 * length]).pad(convert_pad_shape([[0,0],[0,0],[0,length-1]]))
#     return x_flat.reshape([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
# 
#   def _absolute_position_to_relative_position(self, x: Tensor): # x: [b, h, l, l] -> [b, h, l, 2*l-1]
#     batch, heads, length, _ = x.shape
#     x = x.pad(convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
#     x_flat = x.reshape([batch, heads, length**2 + length*(length -1)]).pad(convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
#     return x_flat.reshape([batch, heads, length, 2*length])[:,:,:,1:]